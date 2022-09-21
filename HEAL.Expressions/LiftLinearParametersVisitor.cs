using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using static HEAL.Expressions.Expr;

namespace HEAL.Expressions {
  /// <summary>
  /// Lifts parameters out of linear expressions (p1 * fx + p2 * fy +p3 * fz + p4) -> (p1 fx + p2 fy + fz) * p3 + p4
  /// </summary>
  /// 
  public class LiftLinearParametersVisitor : ExpressionVisitor {
    private readonly ParameterExpression theta;
    private readonly List<double> thetaValues;

    private LiftLinearParametersVisitor(ParameterExpression theta, double[] thetaValues) {
      this.theta = theta;
      this.thetaValues = thetaValues.ToList();
    }

    public static Expression<ParametricFunction> LiftParameters(Expression<ParametricFunction> expr, ParameterExpression theta, double[] thetaValues, out double[] newThetaValues) {
      var v = new LiftLinearParametersVisitor(theta, thetaValues);
      var newExpr = (Expression<ParametricFunction>)v.Visit(expr);
      var updatedThetaValues = v.thetaValues.ToArray();

      // replace parameters with the value 1.0, -1.0, 0.0 with constants
      var selectedIdx = Enumerable.Range(0, updatedThetaValues.Length)
        .Where(idx => updatedThetaValues[idx] == 1.0
        || updatedThetaValues[idx] == -1.0
        || updatedThetaValues[idx] == 0.0)
        .ToArray();

      var replV = new ReplaceParameterWithNumberVisitor(theta, updatedThetaValues, selectedIdx);
      newExpr = (Expression<ParametricFunction>)replV.Visit(newExpr);

      // remove unused parameters
      var collectVisitor = new CollectParametersVisitor(theta, updatedThetaValues);
      newExpr = (Expression<ParametricFunction>)collectVisitor.Visit(newExpr);
      newThetaValues = collectVisitor.GetNewParameterValues;
      return newExpr;
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      var left = Visit(node.Left);
      var right = Visit(node.Right);
      if (IsParameter(right)) {
        if (node.NodeType == ExpressionType.Add) {
          var leftParameters = CollectTermsVisitor.CollectTerms(left).Where(IsParameter).ToArray();
          var sum = 0.0;
          foreach (var leftParam in leftParameters) {
            sum += ParameterValue(leftParam);
            thetaValues[ParameterIndex(leftParam)] = 0.0;
          }
          thetaValues[ParameterIndex(right)] += sum;
          return node.Update(left, null, right);
        } else if (node.NodeType == ExpressionType.Subtract) {
          throw new NotSupportedException("should be handled by ConvertSubToAddVisitor");
        } else if (node.NodeType == ExpressionType.Multiply) {
          var terms = CollectTermsVisitor.CollectTerms(left).ToArray();
          if (terms.All(HasScalingParameter)) {
            // multiply outer parameter into inner parameters
            var p0 = ParameterValue(right);
            foreach (var t in terms) {
              thetaValues[ParameterIndex(FindScalingParameter(t))] *= p0;
            }
            return left;
          } else {
            // lift parameters (p1 + ...) * p2 -> (0 + ...) * p2 + p1*p2
            var leftParameters = terms.Where(IsParameter).ToArray();
            var sum = 0.0;
            foreach (var leftParam in leftParameters) {
              sum += ParameterValue(leftParam);
              thetaValues[ParameterIndex(leftParam)] = 0.0;
            }

            if (sum != 0.0) {
              return Expression.Add(node.Update(left, null, right), CreateParameter(sum * ParameterValue(right)));
            } else return node.Update(left, null, right);
          }

        } else if (node.NodeType == ExpressionType.Divide) {
          throw new NotSupportedException("should be handled by ConvertDivToMulVisitor");
        }
      } else {
        // right is no parameter -> try to extract scale and intercept parameter
        if (node.NodeType == ExpressionType.Add) {
          var terms = CollectTermsVisitor.CollectTerms(left).Concat(CollectTermsVisitor.CollectTerms(right)).ToArray();
          ExtractScaleAndIntercept(terms, out var scale, out var intercept);
          var newNode = node.Update(left, null, right);
          if (scale != 1.0) {
            newNode = Expression.Multiply(newNode, CreateParameter(scale));
          }
          if (intercept != 0.0) {
            newNode = Expression.Add(newNode, CreateParameter(intercept));
          }
          return newNode;
        } else if (node.NodeType == ExpressionType.Subtract) throw new NotSupportedException("should by handled by ConvertSubToAddVisitor");
        else if (node.NodeType == ExpressionType.Multiply) {
          var leftTerms = CollectTermsVisitor.CollectTerms(left);
          var rightTerms = CollectTermsVisitor.CollectTerms(right);
          ExtractScale(leftTerms, out var leftScale);
          ExtractScale(rightTerms, out var rightScale);
          var scale = leftScale * rightScale;

          var newNode = node.Update(left, null, right);
          if (scale != 1.0) newNode = Expression.Multiply(newNode, CreateParameter(scale));
          return newNode;
        } else if (node.NodeType == ExpressionType.Divide) {
          var leftTerms = CollectTermsVisitor.CollectTerms(left);
          var rightTerms = CollectTermsVisitor.CollectTerms(right);
          ExtractScale(leftTerms, out var leftScale);
          ExtractScale(rightTerms, out var rightScale);

          var newNode = node.Update(left, null, right);
          var scale = leftScale / rightScale;
          if (scale != 1.0) newNode = Expression.Multiply(newNode, CreateParameter(scale));
          return newNode;
        }
      }
      return node.Update(left, null, right);
    }

    private void ExtractScale(IEnumerable<Expression> terms, out double scale) {
      var termsArr = terms.ToArray();
      scale = 1.0;
      if (termsArr.All(HasScalingParameter)) {
        scale = ParameterValue(FindScalingParameter(termsArr.First()));
        foreach (var t in termsArr) {
          thetaValues[ParameterIndex(FindScalingParameter(t))] /= scale;
        }
      }
    }
    private void ExtractScaleAndIntercept(IEnumerable<Expression> terms, out double scale, out double intercept) {
      var termsArr = terms.ToArray();
      var additiveParameters = termsArr.Where(IsParameter).ToArray();
      intercept = 0.0;
      foreach (var addParam in additiveParameters) {
        intercept += ParameterValue(addParam);
        thetaValues[ParameterIndex(addParam)] = 0.0;
      }
      ExtractScale(termsArr.Except(additiveParameters), out scale);
    }

    private Expression CreateParameter(double value) {
      var pExpr = Expression.ArrayIndex(theta, Expression.Constant(thetaValues.Count));
      thetaValues.Add(value);
      return pExpr;
    }

    private Expression FindScalingParameter(Expression expr) {
      var factors = CollectFactorsVisitor.CollectFactors(expr);
      return factors.FirstOrDefault(IsParameter);
    }

    private bool HasScalingParameter(Expression expr) {
      return FindScalingParameter(expr) != null;
    }
    private double ParameterValue(Expression expr) {
      return thetaValues[ParameterIndex(expr)];
    }
    private int ParameterIndex(Expression expr) {
      if (!IsParameter(expr)) throw new InvalidProgramException("internal error");
      var binExpr = (BinaryExpression)expr;
      return (int)((ConstantExpression)binExpr.Right).Value;
    }

    private bool IsParameter(Expression expr) {
      return expr.NodeType == ExpressionType.ArrayIndex &&
        ((BinaryExpression)expr).Left == theta;
    }
  }
}

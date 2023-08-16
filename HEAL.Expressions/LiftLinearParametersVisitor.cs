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

    public static ParameterizedExpression LiftParameters(ParameterizedExpression expr) {
      var v = new LiftLinearParametersVisitor(expr.p, expr.pValues);
      var newExpr = (Expression<ParametricFunction>)v.Visit(expr.expr);
      var updatedThetaValues = v.thetaValues.ToArray();

      // replace parameters with the value 1.0, -1.0, 0.0 with constants
      // var selectedIdx = Enumerable.Range(0, updatedThetaValues.Length)
      //   .Where(idx => updatedThetaValues[idx] == 1.0
      //   || updatedThetaValues[idx] == -1.0
      //   || updatedThetaValues[idx] == 0.0)
      //   .ToArray();
      // 
      // var replV = new ReplaceParameterWithNumberVisitor(expr.p, updatedThetaValues, selectedIdx);
      // newExpr = (Expression<ParametricFunction>)replV.Visit(newExpr);

      // remove unused parameters
      var collectVisitor = new CollectParametersVisitor(expr.p, updatedThetaValues);
      newExpr = (Expression<ParametricFunction>)collectVisitor.Visit(newExpr);
      return new ParameterizedExpression(newExpr, expr.p, collectVisitor.GetNewParameterValues);
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      var left = Visit(node.Left);
      var right = Visit(node.Right);
      if (IsParameter(right)) {
        if (node.NodeType == ExpressionType.Add) {
          // sum up all parameters from left into right
          var terms = CollectTermsVisitor.CollectTerms(left);
          var leftParameters = terms.Where(IsParameter);
          var remainingTerms = terms.Where(t => !IsParameter(t));
          return Expression.Add(remainingTerms.Aggregate(Expression.Add), NewParam(leftParameters.Sum(ParameterValue) + ParameterValue(right)));
        } else if (node.NodeType == ExpressionType.Subtract) {
          throw new NotSupportedException("should be handled by ConvertSubToAddVisitor");
        } else if (node.NodeType == ExpressionType.Multiply) {
          var terms = CollectTermsVisitor.CollectTerms(left).ToArray();
          if (terms.All(HasScalingParameter)) {
            // multiply outer parameter into inner parameters
            var p0 = ParameterValue(right);
            return terms.Select(t => ScaleTerm(t, p0)).Aggregate(Expression.Add);
          } else {
            // extract all additive parameters
            var leftParameters = terms.Where(IsParameter);
            var remainingTerms = terms.Where(t => !IsParameter(t));
            var sum = leftParameters.Sum(ParameterValue);

            if (leftParameters.Any()) {
              return Expression.Add(node.Update(remainingTerms.Aggregate(Expression.Add), null, right), NewParam(sum * ParameterValue(right)));
            } else return node.Update(left, null, right);
          }

        } else if (node.NodeType == ExpressionType.Divide) {
          throw new NotSupportedException("should be handled by ConvertDivToMulVisitor");
        }
      } else {
        // right is no parameter -> try to extract scale and intercept parameter
        if (node.NodeType == ExpressionType.Add) {
          var allTerms = CollectTermsVisitor.CollectTerms(left).Concat(CollectTermsVisitor.CollectTerms(right)).ToArray();
          var scale = 1.0;

          // ExtractScaleAndIntercept(allTerms, out var scale, out var intercept);
          var parameters = allTerms.Where(IsParameter);
          var intercept = parameters.Sum(ParameterValue);
          var remaining = allTerms.Where(t => !IsParameter(t));
          if (remaining.All(HasScalingParameter)) {
            scale = ParameterValue(FindScalingParameter(remaining.First()));
            remaining = remaining.Select(t => ScaleTerm(t, 1.0 / scale));
          }

          var newNode = remaining.Aggregate(Expression.Add);
          if (scale != 1.0) {
            newNode = Expression.Multiply(newNode, NewParam(scale));
          }
          if (intercept != 0.0) {
            newNode = Expression.Add(newNode, NewParam(intercept));
          }
          return newNode;
        } else if (node.NodeType == ExpressionType.Subtract) throw new NotSupportedException("should by handled by ConvertSubToAddVisitor");
        else if (node.NodeType == ExpressionType.Multiply) {
          var leftTerms = CollectTermsVisitor.CollectTerms(left);
          var rightTerms = CollectTermsVisitor.CollectTerms(right);
          var leftScale = 1.0;
          var rightScale = 1.0;
          if (leftTerms.All(HasScalingParameter)) {
            leftScale = ParameterValue(FindScalingParameter(leftTerms.First()));
            leftTerms = leftTerms.Select(t => ScaleTerm(t, 1.0 / leftScale));
          }
          if (rightTerms.All(HasScalingParameter)) {
            rightScale = ParameterValue(FindScalingParameter(rightTerms.First()));
            rightTerms = rightTerms.Select(t => ScaleTerm(t, 1.0 / rightScale));
          }
          var scale = leftScale * rightScale;

          var newNode = Expression.Multiply(leftTerms.Aggregate(Expression.Add), rightTerms.Aggregate(Expression.Add));
          if (scale != 1.0) newNode = Expression.Multiply(newNode, NewParam(scale));
          return newNode;
        } else if (node.NodeType == ExpressionType.Divide) {
          var leftTerms = CollectTermsVisitor.CollectTerms(left);
          var rightTerms = CollectTermsVisitor.CollectTerms(right);
          var leftScale = 1.0;
          var rightScale = 1.0;
          if (leftTerms.All(HasScalingParameter)) {
            leftScale = ParameterValue(FindScalingParameter(leftTerms.First()));
            leftTerms = leftTerms.Select(t => ScaleTerm(t, 1.0 / leftScale));
          }
          if (rightTerms.All(HasScalingParameter)) {
            rightScale = ParameterValue(FindScalingParameter(rightTerms.First()));
            rightTerms = rightTerms.Select(t => ScaleTerm(t, 1.0 / rightScale));
          }

          var newNode = Expression.Divide(leftTerms.Aggregate(Expression.Add), rightTerms.Aggregate(Expression.Add));
          var scale = leftScale / rightScale;
          if (scale != 1.0) newNode = Expression.Multiply(newNode, NewParam(scale));
          return newNode;
        }
      }
      return node.Update(left, null, right);
    }

    private Expression ScaleTerm(Expression term, double scale) {
      // return expr;
      if (scale == 1.0) return term; // nothing to do
                                     // TODO handle x * 0.0?

      var factors = CollectFactorsVisitor.CollectFactors(term);
      // product of parameter values
      var currentScalingFactor = factors.Where(IsParameter).Aggregate(1.0, (agg, f) => agg * ParameterValue(f)); // fold product
      var otherFactors = factors.Where(f => !IsParameter(f));
      if (otherFactors.Count() == factors.Count()) throw new InvalidProgramException(); // we must have at least one scaling parameter
      if (otherFactors.Any())
        return Expression.Multiply(otherFactors.Aggregate(Expression.Multiply), NewParam(scale * currentScalingFactor));
      else
        return NewParam(scale * currentScalingFactor);
    }

    private Expression NewParam(double val) {
      thetaValues.Add(val);
      return Expression.ArrayIndex(theta, Expression.Constant(thetaValues.Count - 1));
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

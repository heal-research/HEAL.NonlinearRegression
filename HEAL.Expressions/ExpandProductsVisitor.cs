using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using static HEAL.Expressions.Expr;

namespace HEAL.Expressions {
  /// <summary>
  /// Expands products (a + b + ...) * (x + y + ...) = ax + ay + a... + bx + by + b... +....
  /// The expansion is careful to not increase the number of parameters.
  /// This visitor can be used in a last step of simplification to reduce the number of dependent parameters
  /// </summary>
  /// 
  public class ExpandProductsVisitor : ExpressionVisitor {
    private readonly ParameterExpression theta;
    private readonly List<double> thetaValues;

    private ExpandProductsVisitor(ParameterExpression theta, double[] thetaValues) {
      this.theta = theta;
      this.thetaValues = thetaValues.ToList();
    }

    public static Expression<ParametricFunction> Expand(Expression<ParametricFunction> expr, ParameterExpression theta, double[] thetaValues, out double[] newThetaValues) {
      var v = new ExpandProductsVisitor(theta, (double[])thetaValues.Clone());
      var newExpr = (Expression<ParametricFunction>)v.Visit(expr);

      newThetaValues = v.thetaValues.ToArray();
      return newExpr;
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      var left = Visit(node.Left);
      var right = Visit(node.Right);
      if (node.NodeType == ExpressionType.Multiply) {
        var leftTerms = CollectTermsVisitor.CollectTerms(left).ToArray();
        var rightParamCount = CountParametersVisitor.Count(right, theta);
        if (rightParamCount == 0) {
          // -> multiply into each left term
          var result = Expression.Multiply(leftTerms.First(), right);
          foreach (var lt in leftTerms.Skip(1)) {
            result = Expression.Add(result, Expression.Multiply(lt, right));
          }
          return result;
        } else if (IsParameter(right)) {
          // right is only a parameter
          Expression sumOfTermsWithScaling = null;
          Expression sumOfTermsWithoutScaling = null;
          var rightIndex = ParameterIndex(right);
          var p0 = thetaValues[rightIndex];
          foreach (var lt in leftTerms) {
            if (HasScalingParameter(lt)) {
              thetaValues[ParameterIndex(FindScalingParameter(lt))] *= p0;
              if (sumOfTermsWithScaling == null) sumOfTermsWithScaling = lt;
              else sumOfTermsWithScaling = Expression.Add(sumOfTermsWithScaling, lt);
            } else {
              // left term has no scaling parameter
              if (sumOfTermsWithoutScaling == null) sumOfTermsWithoutScaling = lt;
              else sumOfTermsWithoutScaling = Expression.Add(sumOfTermsWithoutScaling, lt);
            }
          }
          if (sumOfTermsWithoutScaling == null) return sumOfTermsWithScaling;
          else if (sumOfTermsWithScaling == null) return Expression.Multiply(sumOfTermsWithoutScaling, right);
          else return Expression.Add(sumOfTermsWithScaling, Expression.Multiply(sumOfTermsWithoutScaling, right));
        } else if (rightParamCount == 1 && HasScalingParameter(right)) {
          // a single scaling parameter
          // -> multiply into each left term with a scaling parameter and explicitly multiply to sum of remaining terms without scaling parameters
          // if there is only a single term remaining we can include it with the other terms
          Expression sumOfTermsWithScaling = null;
          Expression sumOfTermsWithoutScaling = null;
          var rightFactors = CollectFactorsVisitor.CollectFactors(right).ToArray();
          var rightScalingFactors = rightFactors.Where(IsParameter);
          var remainingFactors = rightFactors.Except(rightScalingFactors);
          var rightWithoutScaling = remainingFactors.Aggregate(Expression.Multiply);
          var p0 = rightScalingFactors.Aggregate(1.0, (prod, f) => prod * thetaValues[ParameterIndex(f)]);

          foreach (var lt in leftTerms) {
            if (HasScalingParameter(lt)) {
              thetaValues[ParameterIndex(FindScalingParameter(lt))] *= p0;
              if (sumOfTermsWithScaling == null) sumOfTermsWithScaling = Expression.Multiply(lt, rightWithoutScaling);
              else sumOfTermsWithScaling = Expression.Add(sumOfTermsWithScaling, Expression.Multiply(lt, rightWithoutScaling));
            } else {
              // left term has no scaling parameter
              if (sumOfTermsWithoutScaling == null) sumOfTermsWithoutScaling = lt;
              else sumOfTermsWithoutScaling = Expression.Add(sumOfTermsWithoutScaling, lt);
            }
          }

          if (sumOfTermsWithoutScaling == null) return sumOfTermsWithScaling;
          else if (sumOfTermsWithScaling == null) return Expression.Multiply(sumOfTermsWithoutScaling, right);
          else return Expression.Add(sumOfTermsWithScaling, Expression.Multiply(sumOfTermsWithoutScaling, right));
        }
      }
      return node.Update(left, null, right);
    }

    private Expression CreateParameter(double p0) {
      var paramExpr = Expression.ArrayIndex(theta, Expression.Constant(thetaValues.Count));
      thetaValues.Add(p0);
      return paramExpr;
    }

    private Expression FindScalingParameter(Expression expr) {
      var factors = CollectFactorsVisitor.CollectFactors(expr);
      return factors.FirstOrDefault(IsParameter);
    }
    private int ParameterIndex(Expression expr) {
      if (!IsParameter(expr)) throw new InvalidProgramException("internal error");
      var binExpr = (BinaryExpression)expr;
      return (int)((ConstantExpression)binExpr.Right).Value;
    }


    private bool HasScalingParameter(Expression expr) {
      return FindScalingParameter(expr) != null;
    }

    private bool IsParameter(Expression expr) {
      return expr.NodeType == ExpressionType.ArrayIndex &&
        ((BinaryExpression)expr).Left == theta;
    }
  }
}

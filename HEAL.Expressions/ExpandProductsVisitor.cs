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

    public static ParameterizedExpression Expand(ParameterizedExpression expr) {
      var v = new ExpandProductsVisitor(expr.p, expr.pValues);
      var newExpr = (Expression<ParametricFunction>)v.Visit(expr.expr);

      return new ParameterizedExpression(newExpr, v.theta, v.thetaValues.ToArray());
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      var left = Visit(node.Left);
      var right = Visit(node.Right);
      if (node.NodeType == ExpressionType.Multiply) {
        var leftTerms = CollectTermsVisitor.CollectTerms(left).ToArray();
        var rightParamCount = CountParametersVisitor.Count(right, theta);
        if (rightParamCount == 0) {
          // -> there are no parameters in right factor. We may multiply right into each left term because we do not duplicate parameters
          return leftTerms.Select(lt => Expression.Multiply(lt, right)).Aggregate(Expression.Add);
        } else if (rightParamCount == 1 && HasScalingParameter(right)) {
          // right has a single scaling parameter
          // -> multiply into each left term with a scaling parameter and explicitly multiply to sum of remaining terms without scaling parameters
          Expression sumOfTermsWithScaling = null;
          Expression sumOfTermsWithoutScaling = null;
          var rightFactors = CollectFactorsVisitor.CollectFactors(right).ToArray();
          var p0 = rightFactors.Where(IsParameter).Aggregate(1.0, (prod, f) => prod * ParameterValue(f));

          var remainingFactors = rightFactors.Where(f => !IsParameter(f));
          Expression rightWithoutScaling = null;
          if (remainingFactors.Any()) rightWithoutScaling = remainingFactors.Aggregate(Expression.Multiply);

          foreach (var lt in leftTerms) {
            if (HasScalingParameter(lt)) {
              var lt_right = rightWithoutScaling != null ? Expression.Multiply(lt, rightWithoutScaling) : lt;
              if (sumOfTermsWithScaling == null) sumOfTermsWithScaling = ScaleTerm(lt_right, p0);
              else sumOfTermsWithScaling = Expression.Add(sumOfTermsWithScaling, ScaleTerm(lt_right, p0));
            } else {
              // left term has no scaling parameter
              if (sumOfTermsWithoutScaling == null) sumOfTermsWithoutScaling = lt;
              else sumOfTermsWithoutScaling = Expression.Add(sumOfTermsWithoutScaling, lt);
            }
          }

          if (sumOfTermsWithoutScaling == null) return sumOfTermsWithScaling;
          else if (sumOfTermsWithScaling == null) return Multiply(sumOfTermsWithoutScaling, right);
          else return Expression.Add(sumOfTermsWithScaling, Multiply(sumOfTermsWithoutScaling, right));
        }
      }
      return node.Update(left, null, right);
    }

    private Expression Multiply(Expression sumOfTermsWithoutScaling, Expression right) {
      if (sumOfTermsWithoutScaling.NodeType == ExpressionType.Divide) {
        var divExpr = (BinaryExpression)sumOfTermsWithoutScaling;
        return divExpr.Update(Expression.Multiply(divExpr.Left, right), null, divExpr.Right);
      }
      return Expression.Multiply(sumOfTermsWithoutScaling, right);
    }

    private Expression ScaleTerm(Expression expr, double scale) {
      // return expr;
      if (scale == 1.0) return expr; // nothing to do
                                     // TODO handle x * 0.0?

      var factors = CollectFactorsVisitor.CollectFactors(expr);
      // product of parameter values
      var currentScalingFactor = factors.Where(IsParameter).Aggregate(1.0, (agg, f) => agg * ParameterValue(f)); // fold product
      var otherFactors = factors.Where(f => !IsParameter(f));
      if (otherFactors.Count() == factors.Count()) throw new InvalidProgramException(); // we must have at least one scaling parameter
      if (otherFactors.Any())
        return Expression.Multiply(otherFactors.Aggregate(Expression.Multiply), NewParam(scale * currentScalingFactor));
      else
        return NewParam(scale * currentScalingFactor);
    }
    private Expression NewParam(double value) {
      thetaValues.Add(value);
      return Expression.ArrayIndex(theta, Expression.Constant(thetaValues.Count - 1));
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

    private double ParameterValue(Expression expr) {
      return thetaValues[ParameterIndex(expr)];
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

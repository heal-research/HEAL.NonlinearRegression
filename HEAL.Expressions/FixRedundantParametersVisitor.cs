using System;
using System.Linq;
using System.Linq.Expressions;
using static HEAL.Expressions.Expr;

namespace HEAL.Expressions {
  /// <summary>
  /// Checks for linearly dependent parameters and converts redundant parameters to constants
  /// </summary>
  /// 
  // TODO: a visitor that lifts parameters or constants out of non-linear expressions as far as possible.
  public class FixRedundantParametersVisitor : ExpressionVisitor {
    private readonly ParameterExpression theta;
    private readonly double[] thetaValues;

    private FixRedundantParametersVisitor(ParameterExpression theta, double[] thetaValues) {
      this.theta = theta;
      this.thetaValues = thetaValues;
    }

    public static Expression<ParametricFunction> FixRedundantParameters(Expression<ParametricFunction> expr, ParameterExpression theta, double[] thetaValues) {
      var v = new FixRedundantParametersVisitor(theta, thetaValues);
      return (Expression<ParametricFunction>)v.Visit(expr);
    }

    protected override Expression VisitBinary(BinaryExpression node) {

      // here we match two types of patterns
      // 1) the current node is (+/-) and one of the arguments is a parameter
      // 2) the current node is (*/div) and one of the arguments is a parameter

      // in the first case we collect all terms and check if one of them is a parameter
      // in the second case we collect all terms and check if all of them have a scaling parameter (* parameter)
      var left = Visit(node.Left);
      var right = Visit(node.Right);

      if (node.NodeType == ExpressionType.Add || node.NodeType == ExpressionType.Subtract) {
        if (IsParameter(left) && IsParameter(right)) {
          right = ParameterValue(right); // two parameters -> fix one 
        } else if (IsParameter(left)) {
          var terms = CollectTermsVisitor.CollectTerms(right);
          if (terms.Any(IsParameter))
            left = ParameterValue(left);
        } else if (IsParameter(right)) {
          var terms = CollectTermsVisitor.CollectTerms(left);
          if (terms.Any(IsParameter))
            right = ParameterValue(right);
        }
      } else if (node.NodeType == ExpressionType.Multiply || node.NodeType == ExpressionType.Divide) {
        if (IsParameter(left) && IsParameter(right)) {
          right = ParameterValue(right); // two parameters -> fix one 
        } else if (IsParameter(left)) {
          var terms = CollectTermsVisitor.CollectTerms(right);
          if (terms.All(HasScalingParameter))
            left = ParameterValue(left);
        } else if (IsParameter(right)) {
          var terms = CollectTermsVisitor.CollectTerms(left);
          if (terms.All(HasScalingParameter))
            right = ParameterValue(right);
        }
      }
      return node.Update(left, null, right);
    }

    private bool HasScalingParameter(Expression expr) {
      var factors = CollectFactorsVisitor.CollectFactors(expr);
      return factors.Any(IsParameter);
    }

    private Expression ParameterValue(Expression expr) {
      if (!IsParameter(expr)) throw new InvalidProgramException("internal error");
      var binExpr = (BinaryExpression)expr;
      var idx = (int)((ConstantExpression)binExpr.Right).Value;
      return Expression.Constant(thetaValues[idx]);
    }

    private bool IsParameter(Expression expr) {
      return expr.NodeType == ExpressionType.ArrayIndex &&
        ((BinaryExpression)expr).Left == theta;
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      return base.VisitUnary(node);
    }
  }
}

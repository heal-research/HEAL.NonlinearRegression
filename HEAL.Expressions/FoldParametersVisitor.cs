using System;
using System.Collections.Generic;
using System.Data.Common;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using static HEAL.Expressions.Expr;

namespace HEAL.Expressions {

  public class FoldParametersVisitor : ExpressionVisitor {
    private readonly ParameterExpression thetaParam;
    private readonly List<double> thetaValues;

    private MethodInfo aq = typeof(Functions).GetMethod("AQ", new[] { typeof(double), typeof(double) });

    public static Expression FoldParameters(Expression expr, ParameterExpression theta, double[] thetaValues, out double[] newThetaValues) {
      var visitor = new FoldParametersVisitor(theta, thetaValues);
      expr = (Expression<ParametricFunction>)visitor.Visit(expr);
      newThetaValues = visitor.GetNewParameterValues;
      return expr;
    }

    // TODO make thetaValues optional
    public FoldParametersVisitor(ParameterExpression theta, double[] thetaValues) {
      this.thetaParam = theta;
      this.thetaValues = thetaValues.ToList();
    }

    public double[] GetNewParameterValues => thetaValues.ToArray();

    protected override Expression VisitBinary(BinaryExpression node) {
      var left = Visit(node.Left);
      var right = Visit(node.Right);
      var rightIsParam = IsParam(node.Right, out var paramExpr, out var paramIdx);
      var leftBinary = left as BinaryExpression;
      if (rightIsParam && leftBinary != null) {
        // no inner parameter
        switch (node.NodeType) {
          case ExpressionType.Add: {
              var terms = CollectTermsVisitor.CollectTerms(leftBinary);
              foreach (var t in terms) {
                // if another parameter is found then we can remove the outer parameter
                if (IsParam(t, out var _, out var leftParamIdx)) {
                  thetaValues[leftParamIdx] += thetaValues[paramIdx];
                  return left;
                }
              }
              // no other parameter found -> return all terms
              return node.Update(left, null, right);
            }
          case ExpressionType.Multiply: {
              // first extract additive parameters
              var terms = CollectTermsVisitor.CollectTerms(leftBinary).ToArray();
              if (terms.All(HasScalingParameter)) { // TODO problematic with constants
                foreach (var t in terms) {
                  var leftParamIdx = FindScalingParameterIndex(t);
                  thetaValues[leftParamIdx] *= thetaValues[paramIdx];
                }
                return left;
              }

              // not all terms have a scaling parameter
              return node.Update(left, null, right);
            }
          case ExpressionType.Subtract: {
              throw new NotSupportedException("should be handled in ArrangeParametersRightVisitor");
            }
          case ExpressionType.Divide: {
              throw new NotSupportedException("should be handled in ArrangeParametersRightVisitor");
            }
        }
      }

      return node.Update(left, null, right);
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      if (node.NodeType == ExpressionType.UnaryPlus) {
        return Visit(node.Operand);
      } else if (node.NodeType == ExpressionType.Negate) {
        var opd = Visit(node.Operand);
        if (IsParam(opd, out var _, out var paramIdx)) {
          thetaValues[paramIdx] *= -1;
          return opd;
        } else {
          return node.Update(opd);
        }
      } else
        return base.Visit(node);
    }

    protected override Expression VisitMethodCall(MethodCallExpression node) {
      var args = node.Arguments.Select(arg => Visit(arg)).ToArray();
      // if all arguments to this function are parameters then we can remove the function and use a parameter instead
      if (args.All(arg => IsParam(arg, out _, out _))) {
        var argValues = new object[args.Length];
        IsParam(args[0], out _, out var firstParamIdx);
        for (int i = 0; i < args.Length; i++) {
          IsParam(args[i], out _, out var paramIdx);
          argValues[i] = thetaValues[paramIdx];
        }
        thetaValues[firstParamIdx] = (double)node.Method.Invoke(node.Object, argValues); // override the first parameter with the result value of the function
        return Expression.ArrayIndex(thetaParam, Expression.Constant(firstParamIdx));
      } else if (node.Method == aq && IsParam(args[1], out var arrIdxExpr, out var paramIdx)) {
        // aq(x, p) = x / sqrt(1 + p²) = x * 1/sqrt(1+p²) = x * p'
        thetaValues[paramIdx] = 1.0 / Math.Sqrt(1 + thetaValues[paramIdx] * thetaValues[paramIdx]);
        return Expression.Multiply(args[0], arrIdxExpr);
      } else {
        return node.Update(node.Object, args);
      }
    }

    private bool IsParam(Expression expr, out BinaryExpression arrayIdxExpr, out int paramIdx) {
      arrayIdxExpr = null;
      paramIdx = -1;
      if (expr.NodeType == ExpressionType.ArrayIndex) {
        arrayIdxExpr = (BinaryExpression)expr;
        if (arrayIdxExpr.Left == thetaParam) {
          paramIdx = (int)((ConstantExpression)arrayIdxExpr.Right).Value;
          return true;
        }

        arrayIdxExpr = null;
      }
      return false;
    }

    private bool HasScalingParameter(Expression expr) {
      var factors = CollectFactorsVisitor.CollectFactors(expr);
      return factors.Any(f => IsParam(f, out _, out _));
    }

    private int FindScalingParameterIndex(Expression expr) {
      var factors = CollectFactorsVisitor.CollectFactors(expr);
      foreach (var f in factors) {
        if (IsParam(f, out _, out var paramIdx)) return paramIdx;
      }
      throw new InvalidProgramException();
    }
  }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using static HEAL.Expressions.Expr;

namespace HEAL.Expressions {
  /// <summary>
  /// Push negation operator down the tree.
  /// -(a + b) -> -a + -b
  /// -(a - b) -> -a - -b
  /// -(a * b) -> a * -b
  /// -(a / b) -> a / -b
  /// -sin(x) -> sin(-x)
  /// -cos(x) -> sin(-x-pi/2)  (optional)
  /// -cbrt(x) -> cbrt(-x)
  /// -(-x) -> x
  ///   
  /// </summary>
  /// 
  public class LowerNegationVisitor : ExpressionVisitor {
    private readonly ParameterExpression theta;
    private readonly List<double> thetaValues;

    private readonly MethodInfo sin = typeof(Math).GetMethod("Sin", new[] { typeof(double) });

    private LowerNegationVisitor(ParameterExpression theta, double[] thetaValues) {
      this.theta = theta;
      this.thetaValues = thetaValues.ToList();
    }

    public static ParameterizedExpression LowerNegation(ParameterizedExpression expr) {
      var v = new LowerNegationVisitor(expr.p, expr.pValues);
      var newExpr = (Expression<ParametricFunction>)v.Visit(expr.expr);
      return new ParameterizedExpression(newExpr, expr.p, v.thetaValues.ToArray());
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      // lift parameters out of negation if possible
      var opd = Visit(node.Operand);
      if (node.NodeType == ExpressionType.Negate) {
        if (opd is UnaryExpression unaryOpd && unaryOpd.NodeType == ExpressionType.Negate) {
          return ((UnaryExpression)opd).Operand;
        } else if (opd is BinaryExpression binOpd) {
          if (binOpd.NodeType == ExpressionType.Add || binOpd.NodeType == ExpressionType.Subtract) {
            return binOpd.Update(Visit(Expression.Negate(binOpd.Left)), null, Visit(Expression.Negate(binOpd.Right)));
          } else if (binOpd.NodeType == ExpressionType.Multiply || binOpd.NodeType == ExpressionType.Divide) {
            var left = binOpd.Left;
            var right = binOpd.Right;
            if (left.NodeType == ExpressionType.Constant && right.NodeType != ExpressionType.Constant) {
              return binOpd.Update(Visit(Expression.Negate(left)), null, right);
            } else if (IsParameter(left) && !IsParameter(right)) {
              return binOpd.Update(Visit(Expression.Negate(left)), null, right);
            } else {
              return binOpd.Update(left, null, Visit(Expression.Negate(right))); // default: negate right
            }
          } else if (IsParameter(binOpd)) {
            return NewParam(-ParameterValue(binOpd));
          }
        } else if (opd is MethodCallExpression methodCall) {
          if (methodCall.Method.Name == "Sin" || methodCall.Method.Name == "Cbrt") {
            return methodCall.Update(methodCall.Object, new[] { Visit(Expression.Negate(methodCall.Arguments[0])) });
          } else if (methodCall.Method.Name == "Cos") {
            var newArg = Expression.Add(Visit(Expression.Negate(methodCall.Arguments[0])), Expression.Constant(-Math.PI / 2));
            return Expression.Call(sin, newArg);
          }
        } else if (opd is ConstantExpression constExpr) {
          return Expression.Constant(-(double)constExpr.Value);
        }
      }
      return node.Update(opd);
    }


    private Expression NewParam(double val) {
      thetaValues.Add(val);
      return Expression.ArrayIndex(theta, Expression.Constant(thetaValues.Count - 1));
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

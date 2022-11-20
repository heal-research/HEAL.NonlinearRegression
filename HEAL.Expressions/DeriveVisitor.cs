using System;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;

namespace HEAL.Expressions {
  // TODO: use the fact that parameters occur only once to speedup code / simplify derived expressions
  public class DeriveVisitor : ExpressionVisitor {
    private readonly int dxIdx;
    private readonly ParameterExpression param;

    public DeriveVisitor(ParameterExpression param, int dxIdx) {
      this.param = param;
      this.dxIdx = dxIdx;
    }

    protected override Expression VisitConstant(ConstantExpression node) {
      return Expression.Constant(0.0);
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      switch (node.NodeType) {
        case ExpressionType.Add: {
            return Expression.Add(Visit(node.Left), Visit(node.Right));
          }
        case ExpressionType.Subtract: {
            return Expression.Subtract(Visit(node.Left), Visit(node.Right));
          }
        case ExpressionType.Multiply: {
            // f g' + g f'
            return Expression.Add(
              Expression.Multiply(node.Left, Visit(node.Right)),
              Expression.Multiply(node.Right, Visit(node.Left)));
          }
        case ExpressionType.Divide: {
            // (g f' - f g') / g²
            var f = node.Left;
            var g = node.Right;
            var df = Visit(node.Left);
            var dg = Visit(node.Right);
            return Expression.Divide(
              Expression.Subtract(
                Expression.Multiply(g, df),
                Expression.Multiply(f, dg)
                ),
              Expression.Call(pow, g, Expression.Constant(2.0))
              );
          }

        case ExpressionType.ArrayIndex: {
            if (node.Left == param) {
              var firstIndex = node.Right;
              if (firstIndex.NodeType == ExpressionType.Constant) {
                var idx = (int)((ConstantExpression)firstIndex).Value;
                if (idx == dxIdx) return Expression.Constant(1.0);
                else return Expression.Constant(0.0);
              } else throw new NotSupportedException("only constant indices for parameter are allowed");
            } else {
              return Expression.Constant(0.0); // return base.VisitBinary(node); // array index for other variables or parameters ok
            }
          }
        default: throw new NotSupportedException(node.ToString());
      }
    }

    private readonly MethodInfo abs = typeof(Math).GetMethod("Abs", new[] { typeof(double) });
    private readonly MethodInfo sin = typeof(Math).GetMethod("Sin", new[] { typeof(double) });
    private readonly MethodInfo cos = typeof(Math).GetMethod("Cos", new[] { typeof(double) });
    private readonly MethodInfo exp = typeof(Math).GetMethod("Exp", new[] { typeof(double) });
    private readonly MethodInfo log = typeof(Math).GetMethod("Log", new[] { typeof(double) });
    private readonly MethodInfo tanh = typeof(Math).GetMethod("Tanh", new[] { typeof(double) });
    private readonly MethodInfo cosh = typeof(Math).GetMethod("Cosh", new[] { typeof(double) });
    private readonly MethodInfo sqrt = typeof(Math).GetMethod("Sqrt", new[] { typeof(double) });
    private readonly MethodInfo cbrt = typeof(Math).GetMethod("Cbrt", new[] { typeof(double) });
    private readonly MethodInfo pow = typeof(Math).GetMethod("Pow", new[] { typeof(double), typeof(double) });
    private readonly MethodInfo sign = typeof(Math).GetMethod("Sign", new[] { typeof(double) }); // for deriv abs(x)

    private readonly MethodInfo plog = typeof(Functions).GetMethod("plog", new[] { typeof(double) });

    //private readonly MethodInfo exp = typeof(Math).GetMethod("Exp", new[] { typeof(double) });
    //private readonly MethodInfo exp = typeof(Math).GetMethod("Exp", new[] { typeof(double) });
    //private readonly MethodInfo exp = typeof(Math).GetMethod("Exp", new[] { typeof(double) });

    // TODO: unit tests
    protected override Expression VisitMethodCall(MethodCallExpression node) {
      var x = node.Arguments.First();
      var dx = Visit(x);
      Expression dfx = null;
      if (node.Method == sin) {
        dfx = Expression.Call(cos, x);
      } else if (node.Method == cos) {
        dfx = Expression.Negate(Expression.Call(sin, x));
      } else if (node.Method == exp) {
        dfx = node;
      } else if (node.Method == log || node.Method == plog) {
        // protected log: log(abs(x))
        dfx = Expression.Divide(Expression.Constant(1.0), x);
      } else if (node.Method == tanh) {
        dfx = Expression.Divide(
          Expression.Constant(2.0),
          Expression.Add(
            Expression.Call(cosh,
              Expression.Multiply(Expression.Constant(2.0), x)),
            Expression.Constant(1.0)));
      } else if (node.Method == sqrt) {
        dfx = Expression.Multiply(Expression.Constant(0.5), Expression.Divide(Expression.Constant(1.0), node));
      } else if (node.Method == cbrt) {
        // 1/3 * 1/cbrt(...)^2
        dfx = Expression.Divide(Expression.Constant(1.0 / 3.0),
          Expression.Call(pow, node, Expression.Constant(2.0)));
      } else if (node.Method == pow) {
        var b = node.Arguments[0];
        var exponent = node.Arguments[1];
        if (exponent.NodeType != ExpressionType.Constant) throw new NotSupportedException("only constant exponents are allowed");
        var expVal = (double)((ConstantExpression)exponent).Value;
        dfx = Expression.Multiply(exponent, Expression.Call(pow, b, Expression.Constant(expVal - 1)));
      } else if (node.Method == abs) {
        dfx = Expression.Convert(Expression.Call(sign, x), typeof(double));
      } else throw new NotSupportedException($"Unsupported method call {node.Method.Name}");

      return Expression.Multiply(dfx, dx);
    }
  }
}

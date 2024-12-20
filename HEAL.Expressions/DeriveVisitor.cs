using System;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;

namespace HEAL.Expressions {
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
              var index = node.Right;
              if (index.NodeType == ExpressionType.Constant) {
                var idx = (int)((ConstantExpression)index).Value;
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
    private readonly MethodInfo log10 = typeof(Math).GetMethod("Log10", new[] { typeof(double) });
    private readonly MethodInfo tanh = typeof(Math).GetMethod("Tanh", new[] { typeof(double) });
    private readonly MethodInfo cosh = typeof(Math).GetMethod("Cosh", new[] { typeof(double) });
    private readonly MethodInfo sqrt = typeof(Math).GetMethod("Sqrt", new[] { typeof(double) });
    private readonly MethodInfo cbrt = typeof(Functions).GetMethod("Cbrt", new[] { typeof(double) });
    private readonly MethodInfo pow = typeof(Math).GetMethod("Pow", new[] { typeof(double), typeof(double) });
    private readonly MethodInfo powabs = typeof(Functions).GetMethod("PowAbs", new[] { typeof(double), typeof(double) });
    
    private readonly MethodInfo sign = typeof(Functions).GetMethod("Sign", new[] { typeof(double) }); // for deriv abs(x)
    private readonly MethodInfo logistic = typeof(Functions).GetMethod("Logistic", new[] { typeof(double) });
    private readonly MethodInfo invlogistic = typeof(Functions).GetMethod("InvLogistic", new[] { typeof(double) });
    private readonly MethodInfo logisticPrime = typeof(Functions).GetMethod("LogisticPrime", new[] { typeof(double) }); // deriv of logistic
    private readonly MethodInfo logisticPrimePrime = typeof(Functions).GetMethod("LogisticPrimePrime", new[] { typeof(double) }); // deriv of logistic
    private readonly MethodInfo invlogisticPrime = typeof(Functions).GetMethod("InvLogisticPrime", new[] { typeof(double) });
    private readonly MethodInfo invlogisticPrimePrime = typeof(Functions).GetMethod("InvLogisticPrimePrime", new[] { typeof(double) });

    //private readonly MethodInfo exp = typeof(Math).GetMethod("Exp", new[] { typeof(double) });
    //private readonly MethodInfo exp = typeof(Math).GetMethod("Exp", new[] { typeof(double) });
    //private readonly MethodInfo exp = typeof(Math).GetMethod("Exp", new[] { typeof(double) });

    // TODO: unit tests
    protected override Expression VisitMethodCall(MethodCallExpression node) {
      var x = node.Arguments.First();
      var dx = Visit(x);
      Expression dfx;
      if (node.Method == sin) {
        dfx = Expression.Call(cos, x);
      } else if (node.Method == cos) {
        dfx = Expression.Negate(Expression.Call(sin, x));
      } else if (node.Method == exp) {
        dfx = node;
      } else if (node.Method == log) {
        dfx = Expression.Divide(Expression.Constant(1.0), x);
      } else if (node.Method == log10) {
        dfx = Expression.Divide(Expression.Divide(Expression.Constant(1.0), x), Expression.Constant(Math.Log(10)));
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
        var exponent = node.Arguments[1];
        if (exponent.NodeType == ExpressionType.Constant) {
          // for efficiency
          var expVal = (double)((ConstantExpression)exponent).Value;
          dfx = Expression.Multiply(exponent, Expression.Call(pow, x, Expression.Constant(expVal - 1)));
        } else {
          // d/dx f(x)^g(x) = f(x)^(g(x)-1) (g(x) f'(x) + f(x) (log f(x)) g'(x))
          var dgx = Visit(exponent);
          return Expression.Multiply(
            Expression.Call(pow, x, Expression.Subtract(exponent, Expression.Constant(1.0))),
            Expression.Add(Expression.Multiply(exponent, dx), 
                           Expression.Multiply(x, 
                              Expression.Multiply(
                                Expression.Call(log, x), 
                                dgx))));
        }
      } else if (node.Method == powabs) {
        var exponent = node.Arguments[1];
        if (exponent.NodeType == ExpressionType.Constant) {
          // for efficiency
          var expVal = (double)((ConstantExpression)exponent).Value;
          dfx = Expression.Multiply(exponent, 
            Expression.Multiply(x,
              Expression.Call(powabs, x, Expression.Constant(expVal - 2))));
        } else {
          // d/dx (pow[abs(f(x))^g(x)]) = abs(f(x))^g(x) (g'(x) log(abs(f(x))) + (g(x) f'(x))/f(x))          
          var dgx = Visit(exponent);
          return Expression.Multiply(
            Expression.Call(powabs, x, exponent),
            Expression.Add(Expression.Divide(Expression.Multiply(exponent, dx), x),
                           Expression.Multiply(
                             Expression.Call(log, Expression.Call(abs, x)),
                             dgx)));
        }
      } else if (node.Method == abs) {
        dfx = Expression.Call(sign, x);
      } else if (node.Method == sign) {
        dfx = Expression.Constant(0.0);
      } else if (node.Method == logistic) {
        dfx = Expression.Call(logisticPrime, x);
      } else if (node.Method == invlogistic) {
        dfx = Expression.Call(invlogisticPrime, x);
      } else if (node.Method == logisticPrime) {
        dfx = Expression.Call(logisticPrimePrime, x);
      } else if (node.Method == invlogisticPrime) {
        dfx = Expression.Call(invlogisticPrimePrime, x);
      } else throw new NotSupportedException($"Unsupported method call {node.Method.Name}");

      return Expression.Multiply(dfx, dx);
    }
  }
}

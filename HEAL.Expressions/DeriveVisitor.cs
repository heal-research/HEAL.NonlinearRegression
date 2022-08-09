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

    public Expression<Func<double[], double>> Visit(Expression<Func<double[], double>> expr) {
      return Expression.Lambda<Func<double[], double>>(Visit(expr.Body), expr.Parameters);
    }

    protected override Expression VisitConstant(ConstantExpression node) {
      return Expression.Constant(0.0);
    }
    
    protected override Expression VisitBinary(BinaryExpression node) {
      Console.WriteLine(node);
      switch (node.NodeType) {
        case ExpressionType.Add: {
          return Expression.Add(Visit(node.Left), Visit(node.Right));
        }
        case ExpressionType.Subtract: {
          return Expression.Subtract(Visit(node.Left), Visit( node.Right));
        }
        case ExpressionType.Multiply: {
          return Expression.Add(
            Expression.Multiply(node.Left, Visit(node.Right)),
            Expression.Multiply(node.Right, Visit(node.Left)));
        }
        case ExpressionType.ArrayIndex: {
          if (node.Left == param) {
            var firstIndex = node.Right;
            if (firstIndex.NodeType == ExpressionType.Constant) {
              var idx = (int)((ConstantExpression)firstIndex).Value;
              if (idx == dxIdx) return Expression.Constant(1.0);
              else return Expression.Constant(0.0);
            } else throw new NotSupportedException("only constant indices for parameter are allowed");
          } else return base.VisitBinary(node); // array index for other variables or parameters ok
        }
        default: throw new NotSupportedException(node.ToString());
      }

      return base.VisitBinary(node);
    }

    private MethodInfo sin = typeof(Math).GetMethod("Sin", new[] { typeof(double) });
    private MethodInfo cos = typeof(Math).GetMethod("Sin", new[] { typeof(double) });

    protected override Expression VisitMethodCall(MethodCallExpression node) {
      if (node.Method == sin) {
        var x = node.Arguments.First();
        return Expression.Multiply(Expression.Call(cos, x), Visit(x));
      } else if (node.Method == cos) {
        var x = node.Arguments.First();
        return Expression.Negate(Expression.Multiply(Expression.Call(sin, x), Visit(x)));
      }

      return base.VisitMethodCall(node);
    }

    
  }
}

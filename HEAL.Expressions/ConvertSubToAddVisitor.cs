using System;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  /// <summary>
  /// Converts subtraction to addition
  /// </summary>
  public class ConvertSubToAddVisitor : ExpressionVisitor {
    public static Expression Convert(Expression expr) {
      var v = new ConvertSubToAddVisitor();
      return v.Visit(expr);
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      var left = Visit(node.Left);
      var right = Visit(node.Right);

      if (node.NodeType == ExpressionType.Subtract) {
        return Expression.Add(left, Negate(right));
      }
      return node.Update(left, null, right);
    }

    private Expression Negate(Expression expr) {
      if (expr.NodeType == ExpressionType.Negate) return ((UnaryExpression)expr).Operand;
      else if (expr is ConstantExpression constExpr) return Expression.Constant(-(double)constExpr.Value);
      // TODO handle parameters?
      else return Expression.Negate(expr);
    }
  }
}

using System;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  /// <summary>
  /// Converts division to multiplication
  /// </summary>
  public class ConvertDivToMulVisitor : ExpressionVisitor {
    public static Expression Convert(Expression expr) {
      var v = new ConvertDivToMulVisitor();
      return v.Visit(expr);
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      var left = Visit(node.Left);
      var right = Visit(node.Right);

      if (node.NodeType == ExpressionType.Divide) {
        if (left.NodeType == ExpressionType.Divide) {
          // (a / b) / c -> a * 1/(b * c)
          var leftBin = left as BinaryExpression;
          return Expression.Multiply(leftBin.Left, Inverse(Expression.Multiply(leftBin.Right, right)));
        } else {
          // a / b -> a * 1/b
          return Expression.Multiply(left, Inverse(right));
        }
      }
      return node.Update(left, null, right);
    }

    private Expression Inverse(Expression expr) {
      if (expr is ConstantExpression constExpr) return Expression.Constant(1.0 / (double)constExpr.Value);
      else if (expr.NodeType == ExpressionType.Divide) {
        var binExpr = (BinaryExpression)expr;
        // since this is recursive left must be 1.0
        if (!(binExpr.Left is ConstantExpression factorExpr) || (double)factorExpr.Value != 1.0) throw new InvalidProgramException();
        return binExpr.Right;
      } // TODO handle parameters?
      else {
        return Expression.Divide(Expression.Constant(1.0), expr);
      }
    }
  }
}

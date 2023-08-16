using System.Linq.Expressions;

namespace HEAL.Expressions {
  /// <summary>
  /// Translates 
  /// (a / b) / (c / d) -->  (a * d) / (b * c)
  /// (a / b) / c -->  a / (b * c)
  /// a / (b / c) -->  a * c / b 
  /// </summary>
  /// 
  public class SimplifyDivisionVisitor : ExpressionVisitor {


    public static Expression Simplify(Expression expr) {
      var v = new SimplifyDivisionVisitor();
      return v.Visit(expr);
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      var left = Visit(node.Left);
      var right = Visit(node.Right);
      if (node.NodeType == ExpressionType.Divide) {
        if (left.NodeType == ExpressionType.Divide && right.NodeType == ExpressionType.Divide) {
          /// (a / b) / (c / d) -->  (a * d) / (b * c)
          var leftDiv = left as BinaryExpression;
          var rightDiv = right as BinaryExpression;
          return Expression.Divide(Expression.Multiply(leftDiv.Left, rightDiv.Right), Expression.Multiply(leftDiv.Right, rightDiv.Left));
        } else if (left.NodeType == ExpressionType.Divide) {
          /// (a / b) / c -->  a / (b * c)
          var leftDiv = left as BinaryExpression;
          return node.Update(leftDiv.Left, null, Expression.Multiply(leftDiv.Right, right));
        } else if (right.NodeType == ExpressionType.Divide) {
          /// a / (b / c) -->  a * c / b 
          var rightDiv = right as BinaryExpression;
          return node.Update(Expression.Multiply(left, rightDiv.Right), null, rightDiv.Left);
        } else return node.Update(left, null, right);
      }
      return node.Update(left, null, right);
    }
  }
}

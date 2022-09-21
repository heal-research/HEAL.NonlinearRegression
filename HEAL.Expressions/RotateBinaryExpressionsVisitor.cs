using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  /// <summary>
  /// Transforms add, mul expressions to correspond to left-associative execution. 
  /// a, b, c are reordered by size
  /// a ° (b ° c) -> (a ° b) ° c
  /// (a ° b) ° (c ° d) -> ((a ° b) ° c) ° d
  /// a ° b is also reordered if size of a < size of b
  /// </summary>
  public class RotateBinaryExpressionsVisitor : ExpressionVisitor {
    private Dictionary<Expression, int> len = new Dictionary<Expression, int>();
    int nodeCount = 0;

    public static Expression Rotate(Expression expr) {
      var v = new RotateBinaryExpressionsVisitor();
      return v.Visit(expr);
    }

    public override Expression Visit(Expression node) {
      var oldNodeCount = nodeCount;
      var newExpr = base.Visit(node);
      if (newExpr != null) {
        nodeCount++;
        len[newExpr] = nodeCount - oldNodeCount;
      }
      return newExpr;
    }
    protected override Expression VisitBinary(BinaryExpression node) {
      var left = Visit(node.Left);
      var leftNodeCount = len[left];
      var right = Visit(node.Right);
      var rightNodeCount = len[right];
      var leftBinary = left as BinaryExpression;
      var rightBinary = right as BinaryExpression;

      if (node.NodeType == ExpressionType.Add || node.NodeType == ExpressionType.Multiply) {
        if (rightBinary != null
          && rightBinary.NodeType == node.NodeType) {
          /// a ° (b ° c) -> (a ° b) ° c,   recursively for a = (al ° ar)
          var terms = new List<Expression>() { rightBinary.Left, rightBinary.Right };
          while (leftBinary != null && leftBinary.NodeType == node.NodeType) {
            terms.Add(leftBinary.Right);
            left = leftBinary.Left;
            leftBinary = left as BinaryExpression;
          }
          terms.Add(left);
          return terms.OrderByDescending(t => len[t]).Aggregate((l, r) => {
            var newBin = Expression.MakeBinary(node.NodeType, l, r);
            len[newBin] = 1 + len[l] + len[r];
            return newBin;
          });
        } else if (len[left] < len[right]) {
          return node.Update(right, null, left);
        } else {
          return node.Update(left, null, right);
        }
      } else {
        return node.Update(left, null, right);
      }
    }
  }
}

using System;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  // finds the index of the offset parameter if it exists
  internal class FindOffsetParameterVisitor : ExpressionVisitor {
    private readonly ParameterExpression theta;
    private int paramIdx = -1;

    private FindOffsetParameterVisitor(ParameterExpression theta) {
      this.theta = theta;
    }

    internal static int FindOffsetParameter(Expression expr, ParameterExpression theta) {
      var v = new FindOffsetParameterVisitor(theta);
      v.Visit(expr);
      return v.paramIdx;
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      if (node.NodeType == ExpressionType.Add || node.NodeType == ExpressionType.Subtract) {
        if (TryGetParameterIndex(node.Left, out var index) || TryGetParameterIndex(node.Right, out index)) {
          this.paramIdx = index;
          return node;
        }

        // both sides of Add/Sub can still contain the offset parameter
        Visit(node.Left);
        Visit(node.Right);
      }
      return node;
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      if (node.NodeType == ExpressionType.UnaryPlus || node.NodeType == ExpressionType.Negate) return Visit(node.Operand);
      else return node;
    }

    private bool TryGetParameterIndex(Expression expr, out int index) {
      if (expr is BinaryExpression binaryExpression && binaryExpression.NodeType == ExpressionType.ArrayIndex && binaryExpression.Left == theta) {
        index = (int)((ConstantExpression)binaryExpression.Right).Value;
        return true;
      }
      index = -1;
      return false;
    }
  }
}
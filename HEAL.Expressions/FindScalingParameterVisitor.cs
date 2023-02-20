using System;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  // finds the index of the parameter which scales the whole expression if it exists
  internal class FindScalingParameterVisitor : ExpressionVisitor {
    private readonly ParameterExpression theta;
    private int paramIdx = -1;

    private FindScalingParameterVisitor(ParameterExpression theta) {
      this.theta = theta;
    }

    internal static int FindScalingParameter(Expression expr, ParameterExpression theta) {
      var v = new FindScalingParameterVisitor(theta);
      v.Visit(expr);
      return v.paramIdx;
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      if (node.NodeType == ExpressionType.Multiply) {
        if (TryGetParameterIndex(node.Left, out var index) || TryGetParameterIndex(node.Right, out index)) {
          this.paramIdx = index;
          return node;
        }

        // both sides of multiply can still contain the scaling parameter
        Visit(node.Left);
        Visit(node.Right);
      } else if (node.NodeType == ExpressionType.Divide) {
        if (TryGetParameterIndex(node.Left, out var index)) {
          this.paramIdx = index;
          return node;
        }
        Visit(node.Left); // left can still contain a scaling parameter
      }
      return node;
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      if (node.NodeType == ExpressionType.UnaryPlus || node.NodeType == ExpressionType.Negate) return Visit(node.Operand);
      else return node;
    }

    private bool TryGetParameterIndex(Expression expr, out int index) {
      if(expr is BinaryExpression binaryExpression && binaryExpression.NodeType == ExpressionType.ArrayIndex && binaryExpression.Left == theta) {
        index = (int) ((ConstantExpression)binaryExpression.Right).Value;
        return true;
      }
      index = -1;
      return false;
    }
  }
}
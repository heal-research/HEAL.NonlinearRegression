using System.Linq.Expressions;

namespace HEAL.Expressions {
  internal class ReplaceParameterWithNumberVisitor : ExpressionVisitor {
    private readonly ParameterExpression theta;
    private readonly double[] thetaValues;

    public ReplaceParameterWithNumberVisitor(ParameterExpression theta, double[] thetaValues) {
      this.theta = theta;
      this.thetaValues = thetaValues;
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      if (node.NodeType == ExpressionType.ArrayIndex && node.Left == theta) {
        return Expression.Constant(thetaValues[(int)((ConstantExpression)node.Right).Value]);
      } else {
        return base.VisitBinary(node);
      }
    }
  }
}
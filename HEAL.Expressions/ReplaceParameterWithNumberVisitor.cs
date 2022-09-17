using System.Linq;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  internal class ReplaceParameterWithNumberVisitor : ExpressionVisitor {
    private readonly ParameterExpression theta;
    private readonly double[] thetaValues;
    private readonly int[] paramIndexes;

    public ReplaceParameterWithNumberVisitor(ParameterExpression theta, double[] thetaValues, int[] paramIndexes = null) {
      this.theta = theta;
      this.thetaValues = thetaValues;
      this.paramIndexes = paramIndexes;
    }

    public static Expression Replace(Expression expr, ParameterExpression theta, double[] thetaValues, int[] paramIndexes = null) {
      var v = new ReplaceParameterWithNumberVisitor(theta, thetaValues, paramIndexes);
      return v.Visit(expr);
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      if (node.NodeType == ExpressionType.ArrayIndex && node.Left == theta) {
        var idx = (int)((ConstantExpression)node.Right).Value;
        if (paramIndexes == null || paramIndexes.Contains(idx)) {
          return Expression.Constant(thetaValues[idx]);
        }
      }

      return base.VisitBinary(node);
    }
  }
}
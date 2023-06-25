using System.Linq.Expressions;

namespace HEAL.Expressions {
  public class ReplaceParameterWithConstantVisitor : ExpressionVisitor {
    private readonly int paramIdx;
    private readonly double constVal;
    private readonly ParameterExpression p;

    public ReplaceParameterWithConstantVisitor(ParameterExpression p, int paramIdx, double constVal) {
      this.p = p;
      this.paramIdx = paramIdx;
      this.constVal = constVal;
    }

    public static Expression<Expr.ParametricFunction> Execute(Expression<Expr.ParametricFunction> expr, ParameterExpression p, int paramIdx, double constVal) {
      var v = new ReplaceParameterWithConstantVisitor(p, paramIdx, constVal);
      return (Expression<Expr.ParametricFunction>)v.Visit(expr);
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      if (node.NodeType == ExpressionType.ArrayIndex) {
        // is it a reference to p[varIdx]?
        if (node.Left == p && (int)((ConstantExpression)node.Right).Value == paramIdx) {
          return Expression.Constant(constVal);
        }
      }
      return base.VisitBinary(node);
    }
  }
}

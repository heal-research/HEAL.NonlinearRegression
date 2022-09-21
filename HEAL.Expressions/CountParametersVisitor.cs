using System.Linq.Expressions;

namespace HEAL.Expressions {
  public class CountParametersVisitor : ExpressionVisitor {
    private readonly ParameterExpression p;
    private int nParam = 0;

    public CountParametersVisitor(ParameterExpression p) {
      this.p = p;
    }


    public static int Count(Expression expr, ParameterExpression param) {
      var v = new CountParametersVisitor(param);
      v.Visit(expr);
      return v.nParam;
    }


    protected override Expression VisitBinary(BinaryExpression node) {
      if (IsParam(node)) {
        nParam++;
        return node;
      } else {
        return base.VisitBinary(node);
      }
    }

    private bool IsParam(Expression expr) {
      return expr.NodeType == ExpressionType.ArrayIndex && ((BinaryExpression)expr).Left == p;
    }
  }
}

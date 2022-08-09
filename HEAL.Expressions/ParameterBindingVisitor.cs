using System;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  internal class ParameterBindingVisitor : ExpressionVisitor {
    private ParameterExpression oldParam;
    private ParameterExpression newParam;

    public Expression BindParameter(Expression<Func<double[], double>> expr, ParameterExpression oldParam,
      ParameterExpression newParam) {
      this.oldParam = oldParam;
      this.newParam = newParam;
      return Visit(expr.Body);
    }

    protected override Expression VisitParameter(ParameterExpression node) {
      return node == oldParam ? newParam : base.VisitParameter(node);
    }
  }
}

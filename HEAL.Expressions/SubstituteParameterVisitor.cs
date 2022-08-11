using System;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  // used in Broadcast method to replace a double[] parameter with a double[,] parameter in an expression
  internal class SubstituteParameterVisitor : ExpressionVisitor {
    private ParameterExpression[] oldParam;
    private ParameterExpression[] newParam;

    public Expression ReplaceParameter(Expression expr, ParameterExpression oldParam,
      ParameterExpression newParam) {
      this.oldParam = new [] { oldParam };
      this.newParam = new [] { newParam };
      return Visit(expr);
    }

    public Expression ReplaceParameters(Expression expr, ParameterExpression[] oldParam,
      ParameterExpression[] newParam) {
      this.oldParam = oldParam;
      this.newParam = newParam;
      return Visit(expr);
    }

    protected override Expression VisitParameter(ParameterExpression node) {
      var idx = Array.IndexOf(oldParam, node);
      return idx >= 0 ? newParam[idx] : base.VisitParameter(node);
    }
  }
}

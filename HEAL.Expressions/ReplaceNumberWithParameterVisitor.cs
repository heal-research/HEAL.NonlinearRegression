using System.Collections.Generic;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  internal class ReplaceNumberWithParameterVisitor : ExpressionVisitor {
    private readonly ParameterExpression theta;
    public List<double> paramValues = new List<double>();
    public double[] ParameterValues => paramValues.ToArray();

    public ReplaceNumberWithParameterVisitor(ParameterExpression theta) {
      this.theta = theta;
    }

    protected override Expression VisitConstant(ConstantExpression node) {
      if (node.Value is double val) {
        return NewParameter(val);
      } else {
        return base.VisitConstant(node);
      }
    }

    private Expression NewParameter(double value) {
      paramValues.Add(value);
      return Expression.ArrayIndex(theta, Expression.Constant(paramValues.Count - 1));
    }
  }
}
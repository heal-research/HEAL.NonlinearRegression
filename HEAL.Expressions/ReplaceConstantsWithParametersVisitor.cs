using System.Collections.Generic;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  public class ReplaceConstantsWithParametersVisitor : ExpressionVisitor {
    private readonly ParameterExpression theta;
    public List<double> paramValues = new List<double>();
    public double[] ParameterValues => paramValues.ToArray();

    private ReplaceConstantsWithParametersVisitor(ParameterExpression theta) {
      this.theta = theta;
    }

    public static ParameterizedExpression Replace(ParameterizedExpression expr) {
      var v = new ReplaceConstantsWithParametersVisitor(expr.p);
      var newExpr = (Expression<Expr.ParametricFunction>)v.Visit(expr.expr);
      return new ParameterizedExpression(newExpr, expr.p, v.ParameterValues);
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
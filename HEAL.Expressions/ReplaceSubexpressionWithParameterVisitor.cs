using System;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  // TODO ReplaceSubexpressionWithParameterVisitor and ReplaceVariableWithParameterVisitor and SubstituteParameterVisitor
  // could potentially be replaced by a visitor that has a dictionary of replacements with a filter on the FlattenExpressionVisitor
  public class ReplaceSubexpressionWithParameterVisitor : ExpressionVisitor {
    private readonly double replVal;
    private readonly Expression subExpr;
    private readonly int numParam;
    private readonly ParameterExpression p;

    private ReplaceSubexpressionWithParameterVisitor(Expression subExpr, ParameterExpression p, int numParam) {
      this.subExpr = subExpr;
      this.numParam = numParam;
      this.p = p;
    }

    public static Expression<T> Execute<T>(Expression<T> expr, Expression subExpr, double[] pValues, double replValue, out double[] newPValues) {
      var p = expr.Parameters[0];
      var v = new ReplaceSubexpressionWithParameterVisitor(subExpr, p, pValues.Length);
      
      // the new parameter is the last one
      newPValues = new double[pValues.Length + 1];
      Array.Copy(pValues, 0, newPValues, 0, pValues.Length);
      newPValues[newPValues.Length - 1] = replValue;
      return (Expression<T>)v.Visit(expr);
    }

    public override Expression Visit(Expression node) {
      if (node == subExpr) {
        // new param
        return Expression.ArrayIndex(p, Expression.Constant(numParam));
      } else return base.Visit(node);
    }
  }
}

using System.Collections.Generic;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  public class CollectParametersVisitor : ExpressionVisitor {
    private readonly List<double> newPValues;
    private readonly double[] pValues;
    private readonly ParameterExpression p;

    public CollectParametersVisitor(ParameterExpression p, double[] pValues) {
      this.p = p;
      this.pValues = pValues;
      this.newPValues = new List<double>();
    }

    public double[] GetNewParameterValues => newPValues.ToArray();


    protected override Expression VisitBinary(BinaryExpression node) {
      if (IsParam(node, out var arrIdxExpr, out var pIdx)) {
        return NewParam(pIdx);
      } else {
        return base.VisitBinary(node);
      }
    }

    private Expression NewParam(int paramIdx) {
      newPValues.Add(pValues[paramIdx]);
      return Expression.ArrayIndex(p, Expression.Constant(newPValues.Count - 1));
    }

    private bool IsParam(Expression expr, out BinaryExpression arrayIdxExpr, out int paramIdx) {
      arrayIdxExpr = null;
      paramIdx = -1;
      if (expr.NodeType == ExpressionType.ArrayIndex) {
        arrayIdxExpr = (BinaryExpression)expr;
        if (arrayIdxExpr.Left == p) {
          paramIdx = (int)((ConstantExpression)arrayIdxExpr.Right).Value;
          return true;
        }

        arrayIdxExpr = null;
      }
      return false;
    }
  }
}

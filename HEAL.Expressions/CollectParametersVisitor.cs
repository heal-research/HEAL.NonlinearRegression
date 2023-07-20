using System.Collections.Generic;
using System.Linq.Expressions;
using System.Reflection.Emit;

namespace HEAL.Expressions {
  // collects only the values of the parameters that actually occur in the expression
  public class CollectParametersVisitor : ExpressionVisitor {
    private readonly List<double> newPValues;
    private readonly double[] pValues;
    private Dictionary<int, Expression> newParameters;
    private readonly ParameterExpression p;

    public CollectParametersVisitor(ParameterExpression p, double[] pValues) {
      this.p = p;
      this.pValues = pValues;
      this.newPValues = new List<double>();
      this.newParameters = new Dictionary<int, Expression>();
    }

    public double[] GetNewParameterValues => newPValues.ToArray();


    public static Expression<Expr.ParametricFunction> Execute(Expression<Expr.ParametricFunction> expr, double[] theta, out double[] newTheta) {
      var v = new CollectParametersVisitor(expr.Parameters[0], theta);
      var newExpr = v.Visit(expr);
      newTheta = v.GetNewParameterValues;
      return (Expression<Expr.ParametricFunction>)newExpr;
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      if (IsParam(node, out var _, out var pIdx)) {
        return NewParam(pIdx);
      } else {
        return base.VisitBinary(node);
      }
    }

    private Expression NewParam(int paramIdx) {
      if (newParameters.TryGetValue(paramIdx, out var newParam)) {
        return newParam;
      } else {
        newPValues.Add(pValues[paramIdx]);
        newParam = Expression.ArrayIndex(p, Expression.Constant(newPValues.Count - 1));
        newParameters[paramIdx] = newParam;
        return newParam;
      }
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

using System.Collections.Generic;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  public class CountParametersVisitor : ExpressionVisitor {
    private readonly ParameterExpression p;
    private HashSet<int> paramIdx = new HashSet<int>();

    public CountParametersVisitor(ParameterExpression p) {
      this.p = p;
    }


    public static int Count(Expression expr, ParameterExpression param) {
      var v = new CountParametersVisitor(param);
      v.Visit(expr);
      return v.paramIdx.Count;
    }


    protected override Expression VisitBinary(BinaryExpression node) {
      if (node.NodeType == ExpressionType.ArrayIndex && node.Left == p) {
        paramIdx.Add((int)((ConstantExpression)node.Right).Value);
        return node;
      } else {
        return base.VisitBinary(node);
      }
    }
  }
}

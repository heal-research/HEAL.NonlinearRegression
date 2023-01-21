using System.Collections.Generic;
using System.Linq.Expressions;
using static HEAL.Expressions.Expr;

namespace HEAL.Expressions {
  public class CollectConstantsVisitor : ExpressionVisitor {
    private List<double> Constants = new List<double>();

    private CollectConstantsVisitor() : base() { }

    public static double[] CollectConstants(Expression<ParametricFunction> expr) {
      var v = new CollectConstantsVisitor();
      v.Visit(expr);
      return v.Constants.ToArray();
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      // do not recurse into array index expressions (do not count array indices)
      if (node.NodeType == ExpressionType.ArrayIndex) return node;
      else {
        base.Visit(node.Left);
        base.Visit(node.Right);
        return node;
      }
    }

    protected override Expression VisitConstant(ConstantExpression node) {
      Constants.Add((double)node.Value);
      return node;
    }
  }
}

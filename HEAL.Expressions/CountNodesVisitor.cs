using System.Linq.Expressions;
using static HEAL.Expressions.Expr;

namespace HEAL.Expressions {
  public class CountNodesVisitor : ExpressionVisitor {
    private int numNodes = 0;

    private CountNodesVisitor() : base() { }
    public static int Count(Expression<ParametricFunction> expr) {
      var v = new CountNodesVisitor();
      v.Visit(expr);
      return v.numNodes;
    }


    protected override Expression VisitBinary(BinaryExpression node) {
      numNodes++;
      // do not recurse into array index expressions (variables and parameters are counted as one node)
      if (node.NodeType == ExpressionType.ArrayIndex) {
        return node;
      } else {
        base.Visit(node.Left);
        base.Visit(node.Right);
        return node;
      }
    }

    protected override Expression VisitConstant(ConstantExpression node) {
      numNodes++;
      return node;
    }

    protected override Expression VisitMethodCall(MethodCallExpression node) {
      numNodes++;
      foreach (var arg in node.Arguments) base.Visit(arg);
      return node;
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      numNodes++;
      base.Visit(node.Operand);
      return node;
    }
  }
}

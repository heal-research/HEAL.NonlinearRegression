using System.Collections.Generic;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  public class CollectTermsVisitor : ExpressionVisitor {
    public List<Expression> Terms { get; private set; } = new List<Expression>();

    public static IEnumerable<Expression> CollectTerms(Expression expr) {
      var v = new CollectTermsVisitor();
      v.Visit(expr);
      return v.Terms;
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      if(node.NodeType == ExpressionType.Add || node.NodeType == ExpressionType.Subtract) {
        Visit(node.Left);
        Visit(node.Right);
        return node;
      } else {
        Terms.Add(node);
        return node;  
      }
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      return Visit(node.Operand);
    }

    protected override Expression VisitConstant(ConstantExpression node) {
      Terms.Add(node);
      return node;
    }
    protected override Expression VisitMethodCall(MethodCallExpression node) {
      Terms.Add(node);
      return node;
    }
  }
}

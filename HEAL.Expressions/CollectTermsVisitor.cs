using System;
using System.Collections.Generic;
using System.Linq;
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
      if (node.NodeType == ExpressionType.Add) {
        Visit(node.Left);
        Visit(node.Right);
        return node;
      } else if (node.NodeType == ExpressionType.Subtract) {
        Visit(node.Left);
        Terms.AddRange(CollectTerms(node.Right).Select(Negate));
        return node;
      } else {
        Terms.Add(node);
        return node;
      }
    }

    private Expression Negate(Expression arg) {
      if (arg is ConstantExpression constExpr) {
        return Expression.Constant(-(double)constExpr.Value);
      } else if (arg is UnaryExpression unaryExpr && unaryExpr.NodeType == ExpressionType.Negate) {
        return unaryExpr.Operand;
      } else return Expression.Negate(arg);
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      if (node.NodeType == ExpressionType.Negate) {
        Terms.AddRange(CollectTerms(node.Operand).Select(Negate));
        return node;
      } else if (node.NodeType == ExpressionType.UnaryPlus) {
        return base.Visit(node.Operand);
      } else throw new NotSupportedException();
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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  public class CollectFactorsVisitor : ExpressionVisitor {
    public List<Expression> Factors { get; private set; } = new List<Expression>();

    public static IEnumerable<Expression> CollectFactors(Expression expr) {
      var v = new CollectFactorsVisitor();
      v.Visit(expr);
      return v.Factors;
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      if (node.NodeType == ExpressionType.Multiply) {
        Visit(node.Left);
        Visit(node.Right);
        return node;
      } else if (node.NodeType == ExpressionType.Divide) {
        Visit(node.Left);
        var invFactors = CollectFactors(node.Right);
        foreach (var invFactor in invFactors) {
          Factors.Add(Inverse(invFactor));
        }
        return node;
      } else {
        Factors.Add(node);
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
        Factors.AddRange(CollectFactors(node.Operand).Select(Negate));
        return node;
      } else if (node.NodeType == ExpressionType.UnaryPlus) {
        return base.Visit(node.Operand);
      } else throw new NotSupportedException();
    }

    private Expression Inverse(Expression expr) {
      if (expr.NodeType == ExpressionType.Divide) {
        var binExpr = (BinaryExpression)expr;
        return binExpr.Update(binExpr.Right, null, binExpr.Left);
      } else {
        return Expression.Divide(Expression.Constant(1.0), expr);
      }
    }

    protected override Expression VisitConstant(ConstantExpression node) {
      Factors.Add(node);
      return node;
    }

    protected override Expression VisitMethodCall(MethodCallExpression node) {
      Factors.Add(node);
      return node;
    }
  }
}

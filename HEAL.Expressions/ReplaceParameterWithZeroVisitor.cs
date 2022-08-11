using System;
using System.Collections.Generic;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  // TODO can be replaced by generic subexpression replacement visitor
  public class ReplaceParameterWithZeroVisitor : ExpressionVisitor {
    private readonly int paramIdx;
    private readonly ParameterExpression p;

    public ReplaceParameterWithZeroVisitor(ParameterExpression p, int paramIdx) {
      this.p = p;
      this.paramIdx = paramIdx;
    }
    
    protected override Expression VisitBinary(BinaryExpression node) {
      if (node.NodeType == ExpressionType.ArrayIndex) {
        // is it a reference to p[varIdx]?
        if (node.Left == p && (int)((ConstantExpression)node.Right).Value == paramIdx) {
          return Expression.Constant(0.0);
        }
      } 
      return base.VisitBinary(node);
    }
  }
}

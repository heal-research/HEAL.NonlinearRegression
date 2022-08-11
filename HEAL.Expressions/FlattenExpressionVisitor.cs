using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  public class FlattenExpressionVisitor : ExpressionVisitor {
    private readonly List<Expression> exprs;

    // for convenience extension methods for Expression to simplify calling the visitors
    private FlattenExpressionVisitor() {
      this.exprs = new List<Expression>();
    }

    public static IEnumerable<Expression> Execute(Expression expr) {
      var v = new FlattenExpressionVisitor();
      v.Visit(expr);
      return v.exprs;
    }

    public override Expression Visit(Expression node) {
      var res = base.Visit(node);
      if(node != null)
        exprs.Add(node);
      return res;
    }
  }
}

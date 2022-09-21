using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  // for each non-linear unary function creates a new expression where this function is replaced by id
  // To analyse if unary functions can be removed without loss in accuracy.
  // Additionally, for checking nested functions tanh(tanh(tanh())). 
  // bottom up: for each unary function create two expressions one with the function and one with the function removed
  public class NestedFunctionsVisitor : ExpressionVisitor {
    private readonly HashSet<Expression> done;
    private MethodCallExpression removedCall;

    private NestedFunctionsVisitor(HashSet<Expression> done) {
      this.done = done;
    }

    
    public static void Execute<T>(Expression<T> expr,
      out List<Expression<T>> reducedExpressions) {
      reducedExpressions = new List<Expression<T>>();
      // start with expr 
      List<Expression<T>> currentExpressions = new List<Expression<T>>() {expr};
      // TODO this code now produces expressions multiple times via different paths.
      // TODO generate all combinations first and then for each combination remove the functions
      do {
        Execute(currentExpressions, out currentExpressions);
        reducedExpressions.AddRange(currentExpressions);
      } while (currentExpressions.Count > 0); // repeatedly reduce the newly generated expressions
    }

    public static void Execute<T>(IList<Expression<T>> exprs,
      out List<Expression<T>> reducedExpressions) {

      // we repeatedly call the visitor. Each time the visitor replaces a method call that is not on the 
      // done list.

      reducedExpressions = new List<Expression<T>>();
      foreach (var expr in exprs) {
        var done = new HashSet<Expression>();
        bool newExprGenerated = false;
        do {
          var visitor = new NestedFunctionsVisitor(done);
          var newExpr = (Expression<T>)visitor.Visit(expr);
          newExprGenerated = visitor.removedCall != null;
          if (newExprGenerated) {
            reducedExpressions.Add(newExpr);
            done.Add(visitor.removedCall);
          }

        } while (newExprGenerated);
      }
    }

    // the first function that is not in 
    protected override Expression VisitMethodCall(MethodCallExpression node) {
      if (node.Arguments.Count == 1 && !done.Contains(node)) {
        this.removedCall = node;
        return node.Arguments[0];
      } else return base.VisitMethodCall(node);
    }
  }
}

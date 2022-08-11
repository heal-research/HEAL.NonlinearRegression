using System;
using System.Data;
using System.Linq;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  // creates all nested models for subexpression relevance and model selection
  // bottom up:
  // for each subexpression recursively: return a tree which has the subexpression and one in which it is replaced by a new parameter
  // the difficult part is how we are going to connect the parameters in the different models for efficient restarting of optimization
  public class NestedModelsVisitor : ExpressionVisitor {
    private readonly ParameterExpression p;
    private readonly double[] pValues;

    private NestedModelsVisitor(ParameterExpression p, double[] pValues) {
      this.p = p;
      this.pValues = pValues;
    }

    public static void Execute<T>(Expression<T> expr,
      ParameterExpression p, double[] pValues) {


      // for each sub-expression in the tree (which is not a parameter expression):
      // evaluate the sub-expression and calculate mean evaluation 
      // create a new expression with the sub-expression replaced by a parameter
      // re-optimize
      (new NestedModelsVisitor(p, pValues)).Visit(expr);
    }
  }
}

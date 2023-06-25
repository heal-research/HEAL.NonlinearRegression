using System;
using System.Data;
using System.Linq;
using System.Linq.Expressions;
using NUnit.Framework;

namespace HEAL.Expressions.Tests {
  public class FunctionLinearityCheckVisitorTests {
    [SetUp]
    public void Setup() {
    }

   
    [Test]
    public void NodeImpacts() {
      double[] theta = new[] { -1.0 };
      var X = new double[,] {
        { 1.0 },
        { 2.0 }
      };
      var m = X.GetLength(0);
      Expression<Expr.ParametricFunction> expr = (p, x) => Math.Exp(p[0] * x[0]);
      var p = expr.Parameters[0];
      var x = expr.Parameters[1];
      var subexpressions = FlattenExpressionVisitor.Execute(expr.Body);
      foreach (var subExpr in subexpressions) {
        var subExprForEval = Expr.Broadcast(Expression.Lambda<Expr.ParametricFunction>(subExpr, p, x)).Compile();
        var eval = new double[m];
        subExprForEval(theta, X, eval);
        var replValue = eval.Average();
        var reducedExpression = ReplaceSubexpressionWithParameterVisitor.Execute(expr, subExpr, theta, replValue, out var newTheta);
        Console.WriteLine($"{subExpr} {reducedExpression} {string.Join(", ", newTheta.Select(ti => ti.ToString("g4")))}");
      }

    }
  }
}

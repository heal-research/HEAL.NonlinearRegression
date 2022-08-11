using System;
using System.Data;
using System.Linq;
using System.Linq.Expressions;
using HEAL.Expressions;
using NUnit.Framework;

namespace HEAL.Expressions.Tests {
  public class FunctionLinearityCheckVisitorTests {
    [SetUp]
    public void Setup() {
    }

    [Test]
    public void NestedFunctions() {
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => Math.Log(x[0]);
        NestedFunctionsVisitor.Execute(expr, out var reducedExpressions);
        Assert.AreEqual(1, reducedExpressions.Count);
        var exprStr = reducedExpressions.Select(e => e.ToString()).ToArray();
        Assert.IsTrue(exprStr.Contains("(p, x) => x[0]"));
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => Math.Sin(Math.Cos(Math.Log(x[0])));
        NestedFunctionsVisitor.Execute(expr, out var reducedExpressions);
        // TODO generate combinations of functions to remove
        Assert.AreEqual(7, reducedExpressions.Count); // 2^3 variants - 1 (original) 
        var exprStr = reducedExpressions.Select(e => e.ToString()).ToArray();
        Assert.IsTrue(exprStr.Contains("(p, x) => x[0]"));
        Assert.IsTrue(exprStr.Contains("(p, x) => Log(x[0])"));
        Assert.IsTrue(exprStr.Contains("(p, x) => Sin(Log(x[0]))"));
        Assert.IsTrue(exprStr.Contains("(p, x) => Sin(Cos(x[0]))"));
        Assert.IsTrue(exprStr.Contains("(p, x) => Cos(x[0])"));
      }
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
      var expressions = FlattenExpressionVisitor.Execute(expr.Body);
      var subexpressions = expressions.Where(e => !IsParameter(e, p) && 
                                                  e is not ParameterExpression && 
                                                  e is not ConstantExpression);
      foreach (var subExpr in subexpressions) {
        var subExprForEval = Expr.Broadcast(Expression.Lambda<Expr.ParametricFunction>(subExpr, p, x)).Compile();
        var eval = new double[m];
        subExprForEval(theta, X, eval);
        var replValue = eval.Average();
        var reducedExpression = RemoveSubexpressionVisitor.Execute(expr, subExpr, theta, replValue, out var newTheta);
        Console.WriteLine($"{subExpr} {reducedExpression} {string.Join(", ", newTheta.Select(ti => ti.ToString("g4")))}");
      }

    }

    private bool IsParameter(Expression expr, ParameterExpression p) {
      return expr is BinaryExpression binExpr && binExpr.Left == p;
    }
  }
}

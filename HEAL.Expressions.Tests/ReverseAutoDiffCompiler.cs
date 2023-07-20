using System;
using System.Linq.Expressions;
using System.Reflection;
using NUnit.Framework;

namespace HEAL.Expressions.Tests {
  public class ReverseAutoDiffCompiler {
    [SetUp]
    public void Setup() {
    }


    [Test]
    public void Forward() {
      int N = 10;
      var theta = new double[] { 1.0 };
      var X = new double[N, 1];
      for (int i = 0; i < N; i++) X[i, 0] = i;
      var f = new double[N];
      var Jac = new double[N, theta.Length];

      {
        Expression<Expr.ParametricFunction> expr = (p, x) => 1.0;
        var newExpr = ReverseAutoDiffVisitor.GenerateJacobianExpression(expr, nRows: N);
        System.Console.WriteLine(GetDebugView(newExpr));
        newExpr.Compile()(theta, X, f, Jac);
        // jac should be zero
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => x[0] * p[0];
        var newExpr = ReverseAutoDiffVisitor.GenerateJacobianExpression(expr, nRows: N);
        System.Console.WriteLine(GetDebugView(newExpr));
        newExpr.Compile()(theta, X, f, Jac);
        // jac should by x[0]
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => x[0] + p[0];
        var newExpr = ReverseAutoDiffVisitor.GenerateJacobianExpression(expr, nRows: N);
        System.Console.WriteLine(GetDebugView(newExpr));
        newExpr.Compile()(theta, X, f, Jac);
        // jac should be 1
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => -x[0];
        var newExpr = ReverseAutoDiffVisitor.GenerateJacobianExpression(expr, nRows: N);
        System.Console.WriteLine(GetDebugView(newExpr));
        newExpr.Compile()(theta, X, f, Jac);
        // jac should be zero
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => Math.Exp(x[0]);
        var newExpr = ReverseAutoDiffVisitor.GenerateJacobianExpression(expr, nRows: N);
        System.Console.WriteLine(GetDebugView(newExpr));
        newExpr.Compile()(theta, X, f, Jac);
        // jac should be zero
      }
    }

    public static string GetDebugView(Expression exp) {
      if (exp == null)
        return null;

      var propertyInfo = typeof(Expression).GetProperty("DebugView", BindingFlags.Instance | BindingFlags.NonPublic);
      return propertyInfo.GetValue(exp) as string;
    }
  }
}

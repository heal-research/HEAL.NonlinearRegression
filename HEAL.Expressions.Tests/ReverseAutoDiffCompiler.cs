using System;
using System.Linq.Expressions;
using NUnit.Framework;

namespace HEAL.Expressions.Tests {
  public class ReverseAutoDiffCompiler {
    [SetUp]
    public void Setup() {
    }


    [Test]
    public void TestJacobianEvaluation() {
      int N = 10;
      var theta = new double[] { 1.0 };
      var X = new double[N, 1];
      for (int i = 0; i < N; i++) X[i, 0] = i;
      var f = new double[N];
      var Jac = new double[N, theta.Length];
      var fRef = new double[N];
      var JacRef = new double[N, 1];

      {
        Expression<Expr.ParametricFunction> expr = (p, x) => 1.0;
        var newFunc = ReverseAutoDiffVisitor.GenerateJacobianExpression(expr, nRows: N);
        // System.Console.WriteLine(GetDebugView(newExpr));
        newFunc(theta, X, f, Jac);

        Expr.Jacobian(expr, 1).Compile()(theta, X, fRef, JacRef);
        for (int i = 0; i < N; i++) {
          Assert.AreEqual(fRef[i], f[i]);
          Assert.AreEqual(JacRef[i, 0], Jac[i, 0]);
        }
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => x[0] * p[0];
        var newFunc = ReverseAutoDiffVisitor.GenerateJacobianExpression(expr, nRows: N);
        // System.Console.WriteLine(GetDebugView(newExpr));
        newFunc(theta, X, f, Jac);

        Expr.Jacobian(expr, 1).Compile()(theta, X, fRef, JacRef);
        for (int i = 0; i < N; i++) {
          Assert.AreEqual(fRef[i], f[i]);
          Assert.AreEqual(JacRef[i, 0], Jac[i, 0]);
        }
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => x[0] + p[0];
        var newFunc = ReverseAutoDiffVisitor.GenerateJacobianExpression(expr, nRows: N);
        // System.Console.WriteLine(GetDebugView(newExpr));
        newFunc(theta, X, f, Jac);

        Expr.Jacobian(expr, 1).Compile()(theta, X, fRef, JacRef);
        for (int i = 0; i < N; i++) {
          Assert.AreEqual(fRef[i], f[i]);
          Assert.AreEqual(JacRef[i, 0], Jac[i, 0]);
        }
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => -x[0];
        var newFunc = ReverseAutoDiffVisitor.GenerateJacobianExpression(expr, nRows: N);
        // System.Console.WriteLine(GetDebugView(newExpr));
        newFunc(theta, X, f, Jac);

        Expr.Jacobian(expr, 1).Compile()(theta, X, fRef, JacRef);
        for (int i = 0; i < N; i++) {
          Assert.AreEqual(fRef[i], f[i]);
          Assert.AreEqual(JacRef[i, 0], Jac[i, 0]);
        }
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => Math.Exp(x[0]);
        var newFunc = ReverseAutoDiffVisitor.GenerateJacobianExpression(expr, nRows: N);
        // System.Console.WriteLine(GetDebugView(newExpr));
        newFunc(theta, X, f, Jac);

        Expr.Jacobian(expr, 1).Compile()(theta, X, fRef, JacRef);
        for (int i = 0; i < N; i++) {
          Assert.AreEqual(fRef[i], f[i]);
          Assert.AreEqual(JacRef[i, 0], Jac[i, 0]);
        }
      }
    }

    
  }
}

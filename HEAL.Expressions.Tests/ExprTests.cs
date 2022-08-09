using System;
using System.Linq.Expressions;
using HEAL.Expressions;
using NUnit.Framework;

namespace HEAL.Expressions.Tests {
  public class Tests {
    [SetUp]
    public void Setup() {
    }

    [Test]
    public void Broadcast() {
      CompileAndRun((p, x) => p[0] * x[0]);
      CompileAndRun((a, b) => a[0] * b[0]);
      CompileAndRun((p, x) => p[0] - x[0]);
      CompileAndRun((p, x) => p[0] + x[0]);
      CompileAndRun((p, x) => p[1] + x[1]);
      CompileAndRun((p, x) => Math.Log(p[0] + x[0]));
      CompileAndRun((p, x) => Math.Exp(p[0] * x[0]));
      CompileAndRun((p, x) => Math.Sin(p[0] * x[0]));
      CompileAndRun((p, x) => Math.Cos(p[0] * x[0]));
      Assert.Pass();
    }

    [Test]
    public void BroadcastGradient() {
      CompileAndRun((p, x, g) => p[0] * x[0]);
      CompileAndRun((a, b, g) => a[0] * b[0]);
      CompileAndRun((p, x, g) => p[0] - x[0]);
      CompileAndRun((p, x, g) => p[0] + x[0]);
      CompileAndRun((p, x, g) => p[1] + x[1]);
      CompileAndRun((p, x, g) => Math.Log(p[0] + x[0]));
      CompileAndRun((p, x, g) => Math.Exp(p[0] * x[0]));
      CompileAndRun((p, x, g) => Math.Sin(p[0] * x[0]));
      CompileAndRun((p, x, g) => Math.Cos(p[0] * x[0]));
    }

    [Test]
    public void Derive() {
      {
        var dfx_dx = Expr.Derive((p, x) => p[0] * x[0], 0);
        CompileAndRun(dfx_dx);
        Assert.AreEqual("(p, x) => x[0]", dfx_dx.ToString());
      }
      {
        var dfx_dx = Expr.Derive((p, x) => p[0] * x[0], 1);
        CompileAndRun(dfx_dx);
        Assert.AreEqual("(p, x) => 0", dfx_dx.ToString());
      }

      {
        var dfx_dx = Expr.Derive((p, x) => p[0] * x[0] + p[1] * x[1], 1);
        CompileAndRun(dfx_dx);
        Assert.AreEqual("(p, x) => x[1]", dfx_dx.ToString());
      }
      {
        var dfx_dx = Expr.Derive((p, x) => Math.Pow(p[0] * x[0], 2), 0);
        CompileAndRun(dfx_dx);
        Assert.AreEqual("(p, x) => ((2 * Pow((p[0] * x[0]), 1)) * x[0])", dfx_dx.ToString());
      }
    }


    [Test]
    public void RemoveRedundantParameters() {
      var paramValues = new[] { 1.0, 2.0, 3.0, 4.0 };
      var expr = Expr.RemoveRedundantParameters((p, x) => (p[0] + p[1]), paramValues, out var newParamValues);
      Assert.AreEqual("(p, x) => p[0]", expr.ToString());
      Assert.AreEqual("3.0d", newParamValues[0]);
    }
    
    private void CompileAndRun(Expression<Expr.ParametricFunction> expr) {
      int N = 10;
      var X = new double[N,3];
      var f = new double[N];
      var t = new double[5] { 1.0, 2.0, 3.0, 4.0, 5.0 };
      Expr.Broadcast(expr).Compile()(t, X, f);
    }
    private void CompileAndRun(Expression<Expr.ParametricGradientFunction> expr) {
      int N = 10;
      var X = new double[N,3];
      var f = new double[N];
      var t = new double[5] { 1.0, 2.0, 3.0, 4.0, 5.0 };
      var J = new double[N, 5];
      Expr.Broadcast(expr).Compile()(t, X, f, J);
    }
  }
}

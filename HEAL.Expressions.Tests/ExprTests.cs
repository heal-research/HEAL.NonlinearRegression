using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using HEAL.Expressions;
using NUnit.Framework;

namespace HEAL.Expressions.Tests {
  public class ExprTests {
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
    public void FoldParameters() {
      {
        var paramValues = new[] { 1.0, 2.0, 3.0, 4.0 };
        var expr = Expr.FoldParameters((p, x) => (p[0] + p[1]), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => p[0]", expr.ToString());
        Assert.AreEqual(3.0, newParamValues[0]);
      }
      {
        var paramValues = new[] { 1.0, 2.0, 3.0, 4.0 };
        var expr = Expr.FoldParameters((p, x) => (p[0] * (x[0] * p[1])), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => (x[0] * p[0])", expr.ToString());
        Assert.AreEqual(2.0, newParamValues[0]);
      }
      {
        var paramValues = new[] { 1.0, 2.0, 3.0, 4.0 };
        var expr = Expr.FoldParameters((p, x) => (p[0] + (x[0] + p[1])), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => (x[0] + p[0])", expr.ToString());
        Assert.AreEqual(3.0, newParamValues[0]);
      }
      {
        var paramValues = new[] { 1.0, 2.0, 3.0, 4.0 };
        var expr = Expr.FoldParameters((p, x) => (p[0] / (x[0]*p[1] + p[2])), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => ((1 / ((x[0] * p[0]) + p[1])) * p[2])", expr.ToString()); // TODO: -> 1/(x0 p0 + 1) * p1
        Assert.AreEqual(2.0, newParamValues[0]);
        Assert.AreEqual(3.0, newParamValues[1]);
        Assert.AreEqual(1.0, newParamValues[2]);
      }
      {
        var paramValues = new[] { 1.0, 2.0, 3.0, 4.0 };
        var expr = Expr.FoldParameters((p, x) => Math.Log(p[0]), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => p[0]", expr.ToString());
        Assert.AreEqual(Math.Log(1), newParamValues[0]);
      }
      {
        var paramValues = new[] { 1.0, 2.0, 3.0, 4.0 };
        var expr = Expr.FoldParameters((p, x) => p[2] * (p[0]*x[0] + p[1]*x[1]), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => (((x[0] * p[0]) + (x[1] * p[1])) * p[2])", expr.ToString());  // TODO scaling for linear
        Assert.AreEqual(1.0, newParamValues[0]);
        Assert.AreEqual(2.0, newParamValues[1]);
        Assert.AreEqual(3.0, newParamValues[2]);
      }
      {
        var paramValues = new[] { 1.0, 2.0, 3.0, 4.0 };
        var expr = Expr.FoldParameters((p, x) => 1.0 / (p[0]*x[0] + p[1]*x[1]) * p[2], paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => ((1 / ((x[0] * p[0]) + (x[1] * p[1]))) * p[2])", expr.ToString()); // TODO fold
        Assert.AreEqual(1, newParamValues[0]);
        Assert.AreEqual(2, newParamValues[1]);
        Assert.AreEqual(3, newParamValues[2]);
      }
    }

    [Test]
    public void ReplaceVariable() {
      var theta = new double[] { 1.0, 2.0, 3.0 };
      Expression<Expr.ParametricFunction> f = (p, x) => (p[0] * x[0] + Math.Log(x[1] * p[1]) + x[1] * x[2]);
      {
        var expr = Expr.ReplaceVariableWithParameter(f,  theta, varIdx: 0, replVal: 3.14, out var newTheta);
        Assert.AreEqual("(p, x) => (((p[0] * p[1]) + Log((x[1] * p[2]))) + (x[1] * x[2]))", expr.ToString());
        Assert.AreEqual(3.14, newTheta[1]);
      }
      {
        var expr = Expr.ReplaceVariableWithParameter(f,  theta, varIdx: 1, replVal: 3.14, out var newTheta);
        Assert.AreEqual("(p, x) => (((p[0] * x[0]) + Log((p[1] * p[2]))) + (p[3] * x[2]))", expr.ToString());
        Assert.AreEqual(3.14, newTheta[1]);
        Assert.AreEqual(3.14, newTheta[3]);
      }
      {
        var expr = Expr.ReplaceVariableWithParameter(f,  theta, varIdx: 2, replVal: 3.14, out var newTheta);
        Assert.AreEqual("(p, x) => (((p[0] * x[0]) + Log((x[1] * p[1]))) + (x[1] * p[2]))", expr.ToString());
        Assert.AreEqual(3.14, newTheta[2]);
      }
    }


    [Test]
    public void Graphviz() {
      Expression<Expr.ParametricFunction> expr = (p, x) => 2.0 * x[0] + x[0] * p[0] + x[1] + Math.Log(x[1]*p[1] + 1.0) + 1/(x[1] * p[2]);
      Console.WriteLine(Expr.ToGraphViz(expr));
      Console.WriteLine(Expr.ToGraphViz(expr, new double[] {0.0, 1.0, 2.0}));
      Console.WriteLine(Expr.ToGraphViz(expr, varNames: new []{"a","b"}));

      var sat = new Dictionary<Expression, double>();
      var rand = new Random(1234);
      foreach (var node in FlattenExpressionVisitor.Execute(expr)) {
        if (node.NodeType == ExpressionType.Multiply) {
          sat.Add(node, rand.NextDouble());
        }
      }
      Console.WriteLine(Expr.ToGraphViz(expr, saturation: sat));
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

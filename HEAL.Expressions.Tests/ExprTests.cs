using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Runtime.InteropServices;
using NUnit.Framework;

namespace HEAL.Expressions.Tests {
  public class ExprTests {
    [SetUp]
    public void Setup() {
      System.Threading.Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
      System.Threading.Thread.CurrentThread.CurrentUICulture = System.Globalization.CultureInfo.InvariantCulture;
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
      CompileAndRunJacobian((p, x, g) => p[0] * x[0]);
      CompileAndRunJacobian((a, b, g) => a[0] * b[0]);
      CompileAndRunJacobian((p, x, g) => p[0] - x[0]);
      CompileAndRunJacobian((p, x, g) => p[0] + x[0]);
      CompileAndRunJacobian((p, x, g) => p[1] + x[1]);
      CompileAndRunJacobian((p, x, g) => Math.Log(p[0] + x[0]));
      CompileAndRunJacobian((p, x, g) => Math.Exp(p[0] * x[0]));
      CompileAndRunJacobian((p, x, g) => Math.Sin(p[0] * x[0]));
      CompileAndRunJacobian((p, x, g) => Math.Cos(p[0] * x[0]));
      Assert.Pass();
    }

    [Test]
    public void Autodiff() {
      CompareSymbolicAndAutoDiffJacobian((p, x) => p[0] * x[0]);
      CompareSymbolicAndAutoDiffJacobian((a, b) => a[0] * b[0]);
      CompareSymbolicAndAutoDiffJacobian((p, x) => p[0] - x[0]);
      CompareSymbolicAndAutoDiffJacobian((p, x) => p[0] + x[0]);
      CompareSymbolicAndAutoDiffJacobian((p, x) => p[0] / x[0]);
      CompareSymbolicAndAutoDiffJacobian((p, x) => x[0] / p[0]);
      CompareSymbolicAndAutoDiffJacobian((p, x) => p[1] + x[1]);
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Log(p[0] + x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Exp(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Sqrt(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Functions.Cbrt(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Sin(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Cos(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Abs(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Tanh(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Functions.Logistic(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Functions.InvLogistic(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Pow(x[0], p[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Pow(x[0], 2.0));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Pow(p[0] * x[0], p[1]));
      Assert.Pass();
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
        Assert.AreEqual("(p, x) => (Pow(x[0], 2) * (2 * Pow(p[0], 1)))", dfx_dx.ToString()); // Pow(2*p0, 1) is simplified later
      }
    }

    [Test]
    public void Hessian() {
      CompileAndRunHessian((double[] p, double[] x) => Math.Log(p[0] + x[0]));
      CompileAndRunHessian((double[] p, double[] x) => Math.Exp(p[0] * x[0]));
      CompileAndRunHessian((double[] p, double[] x) => Math.Sin(p[0] * x[0]));
      CompileAndRunHessian((double[] p, double[] x) => Math.Cos(p[0] * x[0]));
      Assert.Pass();
    }

    [Test]
    public void FoldParameters() {
      {
        var paramValues = new[] { 2.0, 2.0, 3.0, 4.0 };
        var expr = Expr.FoldParameters((p, x) => (p[0] + p[1]), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => p[0]", expr.ToString());
        Assert.AreEqual(4.0, newParamValues[0]);
      }
      {
        var paramValues = new[] { 2.0, 2.0, 3.0, 4.0 };
        var expr = Expr.FoldParameters((p, x) => (p[0] * (x[0] * p[1])), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => (x[0] * p[0])", expr.ToString());
        Assert.AreEqual(4.0, newParamValues[0]);
      }
      {
        var paramValues = new[] { 2.0, 2.0, 3.0, 4.0 };
        var expr = Expr.FoldParameters((p, x) => (p[0] + (x[0] + p[1])), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => (x[0] + p[0])", expr.ToString());
        Assert.AreEqual(4.0, newParamValues[0]);
      }
      {
        var paramValues = new[] { 2.0, 3.0, 4.0, 5.0 };
        var expr = Expr.FoldParameters((p, x) => (p[0] / (x[0] * p[1] + p[2])), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => (p[0] / (x[0] + p[1]))", expr.ToString());
        Assert.AreEqual(2.0 / 3.0, newParamValues[0]);
        Assert.AreEqual(4.0 / 3.0, newParamValues[1]);
      }
      {
        var paramValues = new[] { 2.0, 2.0, 3.0, 4.0 };
        var expr = Expr.FoldParameters((p, x) => Math.Log(p[0]), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => p[0]", expr.ToString());
        Assert.AreEqual(Math.Log(2), newParamValues[0]);
      }
      {
        var paramValues = new[] { 2.0, 3.0, 4.0, 5.0 };
        var expr = Expr.FoldParameters((p, x) => p[2] * (p[0] * x[0] + p[1] * x[1]), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => ((x[1] * p[0]) + (x[0] * p[1]))", expr.ToString());
        Assert.AreEqual(12.0, newParamValues[0]);
        Assert.AreEqual(8.0, newParamValues[1]);
      }
      {
        var paramValues = new[] { 2.0, 3.0, 4.0, 5.0 };
        var expr = Expr.FoldParameters((p, x) => 1.0 / (p[0] * x[0] + p[1] * x[1]) * p[2], paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => (p[0] / (x[0] + (x[1] * p[1])))", expr.ToString());
        Assert.AreEqual(4.0 / 2.0, newParamValues[0]);
        Assert.AreEqual(3.0 / 2.0, newParamValues[1]);
      }
      {
        var paramValues = new[] { 2.0 };
        var expr = Expr.FoldParameters((p, x) => Math.Log(p[0]), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => p[0]", expr.ToString());
        Assert.AreEqual(Math.Log(2.0), newParamValues[0]);
      }
      {
        var paramValues = new[] { 2.0, 3.0 };
        var expr = Expr.FoldParameters((p, x) => Math.Pow(p[0], p[1]), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => p[0]", expr.ToString());
        Assert.AreEqual(8, newParamValues[0]);
      }
      {
        var paramValues = new[] { 2.0, 3.0, 4.0, 5.0 };
        var expr = Expr.FoldParameters((p, x) => (p[0] * x[0] + p[1] * x[1]) / (p[2] * x[0] + x[1]) * p[3], paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => (((x[0] + (x[1] * p[0])) / ((x[0] * p[1]) + x[1])) * p[2])", expr.ToString());
        Assert.AreEqual(3.0 / 2.0, newParamValues[0]);
        Assert.AreEqual(4.0, newParamValues[1]);
        Assert.AreEqual(10.0, newParamValues[2]);
      }
      {
        var paramValues = new[] { 2.0, 3.0, 4.0, 5.0 };
        var expr = Expr.FoldParameters((p, x) => (p[0] * x[0] + p[1] * x[1]) / (p[2] * x[0] + p[3] * x[1]), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => (((x[0] + (x[1] * p[0])) / (x[0] + (x[1] * p[1]))) * p[2])", expr.ToString());
        Assert.AreEqual(3.0 / 2.0, newParamValues[0]);
        Assert.AreEqual(5.0 / 4.0, newParamValues[1]);
        Assert.AreEqual(2.0 / 4.0, newParamValues[2]);
      }
    }

    [Test]
    public void ReplaceVariable() {
      var theta = new double[] { 1.0, 2.0, 3.0 };
      Expression<Expr.ParametricFunction> f = (p, x) => (p[0] * x[0] + Math.Log(x[1] * p[1]) + x[1] * x[2]);
      {
        var expr = Expr.ReplaceVariableWithParameter(f, theta, varIdx: 0, replVal: 3.14, out var newTheta);
        Assert.AreEqual("(p, x) => (((p[0] * p[1]) + Log((x[1] * p[2]))) + (x[1] * x[2]))", expr.ToString());
        Assert.AreEqual(3.14, newTheta[1]);
      }
      {
        var expr = Expr.ReplaceVariableWithParameter(f, theta, varIdx: 1, replVal: 3.14, out var newTheta);
        Assert.AreEqual("(p, x) => (((p[0] * x[0]) + Log((p[1] * p[2]))) + (p[3] * x[2]))", expr.ToString());
        Assert.AreEqual(3.14, newTheta[1]);
        Assert.AreEqual(3.14, newTheta[3]);
      }
      {
        var expr = Expr.ReplaceVariableWithParameter(f, theta, varIdx: 2, replVal: 3.14, out var newTheta);
        Assert.AreEqual("(p, x) => (((p[0] * x[0]) + Log((x[1] * p[1]))) + (x[1] * p[2]))", expr.ToString());
        Assert.AreEqual(3.14, newTheta[2]);
      }
    }

    [Test]
    public void LiftParameters() {
      {
        Expression<Expr.ParametricFunction> f = (p, x) => Math.Log(p[0] * x[0] + p[1] * x[1]);
        var theta = new double[] { 2.0, 3.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => (Log(((x[0] * 1) + (x[1] * p[0]))) + p[1])", expr.ToString());
        Assert.AreEqual(3.0 / 2.0, newTheta[0]);
        Assert.AreEqual(Math.Log(2.0), newTheta[1]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => Math.Log(p[0] * x[0] + p[1]);
        var theta = new double[] { 2.0, 3.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => (Log(((x[0] * 1) + p[0])) + p[1])", expr.ToString());
        Assert.AreEqual(3.0 / 2.0, newTheta[0]);
        Assert.AreEqual(Math.Log(2.0), newTheta[1]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => Math.Log(p[0] * x[0] - p[1]);
        var theta = new double[] { 2.0, 3.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => Log(((p[0] * x[0]) - p[1]))", expr.ToString());
        Assert.AreEqual(2.0, newTheta[0]);
        Assert.AreEqual(3.0, newTheta[1]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => Math.Log(p[0] * x[0] + p[1] + 3);
        var theta = new double[] { 2.0, 3.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => Log((((p[0] * x[0]) + p[1]) + 3))", expr.ToString()); // no change
        Assert.AreEqual(2.0, newTheta[0]);
        Assert.AreEqual(3.0, newTheta[1]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => Math.Sqrt(p[0] * x[0] + p[1] * x[1]);
        var theta = new double[] { 2.0, 3.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => (Sqrt(((x[0] * 1) + (x[1] * p[0]))) * p[1])", expr.ToString());
        Assert.AreEqual(3.0 / 2.0, newTheta[0]);
        Assert.AreEqual(Math.Sqrt(2.0), newTheta[1]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => Math.Sqrt(Math.Sqrt(p[0] * x[0]));
        var theta = new double[] { 2.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => (Sqrt((Sqrt((x[0] * 1)) * 1)) * p[0])", expr.ToString());
        Assert.AreEqual(Math.Sqrt(Math.Sqrt(2.0)), newTheta[0]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => Math.Sqrt(x[0] * p[0] + p[1] * x[0]);
        var theta = new double[] { 2.0, -2.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => (Sqrt(((x[0] * 1) + (x[0] * -1))) * p[0])", expr.ToString());
        Assert.AreEqual(Math.Sqrt(2.0), newTheta[0]);
      }

      {
        Expression<Expr.ParametricFunction> f = (p, x) => Math.Sqrt(Math.Sqrt((x[0] * x[0]) * p[0]));
        var theta = new double[] { 2.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => (Sqrt((Sqrt(((x[0] * x[0]) * 1)) * 1)) * p[0])", expr.ToString());
        Assert.AreEqual(Math.Sqrt(Math.Sqrt(2.0)), newTheta[0]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => Math.Sqrt(Math.Sqrt(p[0] * x[0]) + Math.Sqrt(p[1] * x[1]));
        var theta = new double[] { 2.0, 3.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => (Sqrt(((Sqrt((x[0] * 1)) * 1) + (Sqrt((x[1] * 1)) * p[0]))) * p[1])", expr.ToString());
        Assert.AreEqual(Math.Sqrt(3.0) / Math.Sqrt(2.0), newTheta[0]);
        Assert.AreEqual(Math.Sqrt(Math.Sqrt(2.0)), newTheta[1]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => Math.Cbrt(p[0] * x[0] + p[1] * x[1]);
        var theta = new double[] { 2.0, 3.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => (Cbrt(((x[0] * 1) + (x[1] * p[0]))) * p[1])", expr.ToString());
        Assert.AreEqual(3.0 / 2.0, newTheta[0]);
        Assert.AreEqual(Math.Cbrt(2.0), newTheta[1], 1e-8);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => Math.Pow(p[0] * x[0] + p[1] * x[1], 2);
        var theta = new double[] { 2.0, 3.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => (Pow(((x[0] * 1) + (x[1] * p[0])), 2) * p[1])", expr.ToString());
        Assert.AreEqual(3.0 / 2.0, newTheta[0]);
        Assert.AreEqual(4.0, newTheta[1]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => Math.Pow(p[0] * x[0] + p[1] * x[1], 3);
        var theta = new double[] { 2.0, 3.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => (Pow(((x[0] * 1) + (x[1] * p[0])), 3) * p[1])", expr.ToString());
        Assert.AreEqual(3.0 / 2.0, newTheta[0]);
        Assert.AreEqual(8.0, newTheta[1]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => Math.Exp(p[0] * x[0] + p[1]);
        var theta = new double[] { 2.0, 3.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => (Exp((p[0] * x[0])) * p[1])", expr.ToString());
        Assert.AreEqual(2.0, newTheta[0]);
        Assert.AreEqual(Math.Exp(3.0), newTheta[1]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => p[0] / (p[1] * x[0] + p[2]);
        var theta = new double[] { 2.0, 3.0, 4.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => ((1 / ((x[0] * 1) + p[0])) * p[1])", expr.ToString());
        Assert.AreEqual(4.0 / 3.0, newTheta[0]);
        Assert.AreEqual(2.0 / 3.0, newTheta[1]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => p[0] / (p[1] * x[0] + p[2] * x[1]);
        var theta = new double[] { 2.0, 3.0, 4.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => ((1 / ((x[0] * 1) + (x[1] * p[0]))) * p[1])", expr.ToString());
        Assert.AreEqual(4.0 / 3.0, newTheta[0]);
        Assert.AreEqual(2.0 / 3.0, newTheta[1]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => Math.Sin(p[0] * x[0] + p[1]);
        var theta = new double[] { 2.0, 3.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => Sin(((p[0] * x[0]) + p[1]))", expr.ToString());      // no change
        Assert.AreEqual(2.0, newTheta[0]);
        Assert.AreEqual(3.0, newTheta[1]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => Math.Cos(p[0] * x[0] + p[1]);
        var theta = new double[] { 2.0, 3.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => Cos(((p[0] * x[0]) + p[1]))", expr.ToString());      // no change
        Assert.AreEqual(2.0, newTheta[0]);
        Assert.AreEqual(3.0, newTheta[1]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => Math.Tanh(p[0] * x[0] + p[1]);
        var theta = new double[] { 2.0, 3.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => Tanh(((p[0] * x[0]) + p[1]))", expr.ToString());      // no change
        Assert.AreEqual(2.0, newTheta[0]);
        Assert.AreEqual(3.0, newTheta[1]);
      }
      {
        // lift out of negation
        Expression<Expr.ParametricFunction> f = (p, x) => p[0] * x[0] + -(p[1] * x[1] + p[2]);
        var theta = new double[] { 2.0, 3.0, 4.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => ((p[0] * x[0]) + (((x[1] * p[1]) + 1) * p[2]))", expr.ToString());
        Assert.AreEqual(2.0, newTheta[0]);
        Assert.AreEqual(3.0 / 4.0, newTheta[1]);
        Assert.AreEqual(-4.0, newTheta[2]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => p[0] * x[0] - (Math.Log(p[1] * x[1]) - Math.Log(p[2] * x[2]));
        var theta = new double[] { 2.0, 3.0, 4.0 };
        var expr = LiftParameters(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => ((p[0] * x[0]) - ((Log((x[1] * 1)) + p[1]) - (Log((x[2] * 1)) + p[2])))", expr.ToString());
        Assert.AreEqual(2.0, newTheta[0]);
        Assert.AreEqual(Math.Log(3.0), newTheta[1]);
        Assert.AreEqual(Math.Log(4.0), newTheta[2]);
      }
    }

    private static Expression<Expr.ParametricFunction> LiftParameters(Expression<Expr.ParametricFunction> f, double[] theta, out double[] newTheta) {
      var expr = LiftParametersVisitor.LiftParameters(new ParameterizedExpression(f, f.Parameters[0], theta));
      newTheta = expr.pValues;
      return expr.expr;
    }

    [Test]
    public void LowerNegation() {
      {
        Expression<Expr.ParametricFunction> f = (p, x) => -(p[0]);
        var theta = new double[] { 2.0, 3.0, 4.0 };
        var expr = LowerNegation(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => p[3]", expr.ToString());
        Assert.AreEqual(-2.0, newTheta[3]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => -(x[0] * p[0] + p[1] * x[0]);
        var theta = new double[] { 2.0, 3.0, 4.0 };
        var expr = LowerNegation(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => ((x[0] * p[3]) + (p[4] * x[0]))", expr.ToString());
        Assert.AreEqual(-2.0, newTheta[3]);
        Assert.AreEqual(-3.0, newTheta[4]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => -(x[0] * p[0] * -(p[1] * x[0]));
        var theta = new double[] { 2.0, 3.0, 4.0 };
        var expr = LowerNegation(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => ((x[0] * p[0]) * (p[4] * x[0]))", expr.ToString());
        Assert.AreEqual(2.0, newTheta[0]);
        Assert.AreEqual(3.0, newTheta[4]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => -(Math.Sin(p[0]));
        var theta = new double[] { 2.0, 3.0, 4.0 };
        var expr = LowerNegation(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => Sin(p[3])", expr.ToString());
        Assert.AreEqual(-2.0, newTheta[3]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => -(Math.Cbrt(p[0]));
        var theta = new double[] { 2.0, 3.0, 4.0 };
        var expr = LowerNegation(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => Cbrt(p[3])", expr.ToString());
        Assert.AreEqual(-2.0, newTheta[3]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => -(Math.Cos(p[0]));
        var theta = new double[] { 2.0, 3.0, 4.0 };
        var expr = LowerNegation(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => Sin((p[3] + -1.5707963267948966))", expr.ToString());
        Assert.AreEqual(-2.0, newTheta[3]);
      }
    }

    private static Expression<Expr.ParametricFunction> LowerNegation(Expression<Expr.ParametricFunction> f, double[] theta, out double[] newTheta) {
      var expr = LowerNegationVisitor.LowerNegation(new ParameterizedExpression(f, f.Parameters[0], theta));
      newTheta = expr.pValues;
      return expr.expr;
    }

    [Test]
    public void Simplify() {
      {
        Expression<Expr.ParametricFunction> f = (p, x) => p[0] - p[1] * (p[2] * x[0] + p[3]);
        var theta = new double[] { 2.0, 3.0, 4.0, 5.0 };
        var simplifiedExpr = Expr.FoldParameters(f, theta, out var newP);
        Assert.AreEqual("(p, x) => ((x[0] * p[0]) + p[1])", simplifiedExpr.ToString());
        Assert.AreEqual(-12.0, newP[0]);
        Assert.AreEqual(-13.0, newP[1]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => 1.0 / x[0] * x[1] * p[0];
        var theta = new double[] { 2.0, 3.0, 4.0, 5.0 };
        var simplifiedExpr = Expr.FoldParameters(f, theta, out var newP);
        Assert.AreEqual("(p, x) => ((x[1] / x[0]) * p[0])", simplifiedExpr.ToString()); // TODO
        Assert.AreEqual(2.0, newP[0]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => 1 / (x[0] * p[0] + x[1] * p[1] + p[2]) * (x[2] * p[3] + p[4]) * p[5];
        var theta = new double[] { 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
        var simplifiedExpr = Expr.FoldParameters(f, theta, out var newP);
        Assert.AreEqual("(p, x) => (((x[2] + p[0]) / ((x[0] + (x[1] * p[1])) + p[2])) * p[3])", simplifiedExpr.ToString()); // TODO
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => p[0] + p[1] * Math.Sqrt(p[2] * x[0] + Math.Sqrt(Math.Sqrt(Math.Sqrt(p[3] * x[1] * p[4] * x[1]) * p[5] * x[2] * p[6] * x[1] + p[7] * x[3] * p[8] * x[1])));
        var theta = new double[] { 5.42128, -1.12871, -0.230468, 0.531836, 0.693147, 0.531836, 0.693147, 3.14159, 1.77245 };
        Console.WriteLine(f.ToString());
        Console.WriteLine(string.Join(" ", theta.Select(pi => pi.ToString("e4"))));
        var simplifiedExpr = Expr.FoldParameters(f, theta, out var newP);
        var newSimplifiedStr = simplifiedExpr.ToString();
        Console.WriteLine(newSimplifiedStr);
        Console.WriteLine(string.Join(" ", newP.Select(pi => pi.ToString("e4"))));
        string oldSimplifiedStr;
        // simplify until no change (TODO: this shouldn't be necessary if foldParameters is implemented correctly)
        do {
          oldSimplifiedStr = newSimplifiedStr;
          simplifiedExpr = Expr.FoldParameters(simplifiedExpr, newP, out newP);
          newSimplifiedStr = simplifiedExpr.ToString();
          Console.WriteLine(newSimplifiedStr);
          Console.WriteLine(string.Join(" ", newP.Select(pi => pi.ToString("e4"))));
        } while (newSimplifiedStr != oldSimplifiedStr);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => p[0] - p[1] * (p[2] * x[0] + p[3]);
        var theta = new double[] { 2.0, 3.0, 4.0, 5.0 };

        Console.WriteLine(f.ToString());
        Console.WriteLine(string.Join(" ", theta.Select(pi => pi.ToString("e4"))));
        var simplifiedExpr = Expr.FoldParameters(f, theta, out var newP);
        var newSimplifiedStr = simplifiedExpr.ToString();
        Console.WriteLine(newSimplifiedStr);
        Console.WriteLine(string.Join(" ", newP.Select(pi => pi.ToString("e4"))));
        string oldSimplifiedStr;
        // simplify until no change (TODO: this shouldn't be necessary if foldParameters is implemented correctly)
        do {
          oldSimplifiedStr = newSimplifiedStr;
          simplifiedExpr = Expr.FoldParameters(simplifiedExpr, newP, out newP);
          newSimplifiedStr = simplifiedExpr.ToString();
          Console.WriteLine(newSimplifiedStr);
          Console.WriteLine(string.Join(" ", newP.Select(pi => pi.ToString("e4"))));
        } while (newSimplifiedStr != oldSimplifiedStr);

      }
      // {
      //   Expression<Expr.ParametricFunction> f = (p, x) => p[0] + -p[1] * (Math.Sqrt(p[2] * x[0]) - (p[3] * x[1] + (p[4] - (Math.Log(p[5] * x[2]) + Math.Log(Math.Log(p[6] * x[3]))) - Math.Log(p[7] * x[4]) + (Math.Sqrt(Math.Log(p[8] * x[5])) - Math.Sqrt(p[9] * x[6]) - Math.Log(p[10] * x[7])) * p[11])));
      //   var theta = new double[] { 4.11006, 0.141145, 0.240224, 0.355864, 1.68955, 0.258132, 1.11302, 0.554311, 0.182776, 1.11302, 0.182776, 1.86428 };
      //   Console.WriteLine(f.ToString());
      //   Console.WriteLine(string.Join(" ", theta.Select(pi => pi.ToString("e4"))));
      //   var simplifiedExpr = Expr.FoldParameters(f, theta, out var newP);
      //   var newSimplifiedStr = simplifiedExpr.ToString();
      //   Console.WriteLine(newSimplifiedStr);
      //   Console.WriteLine(string.Join(" ", newP.Select(pi => pi.ToString("e4"))));
      //   string oldSimplifiedStr;
      //   // simplify until no change (TODO: this shouldn't be necessary if foldParameters is implemented correctly)
      //   do {
      //     oldSimplifiedStr = newSimplifiedStr;
      //     simplifiedExpr = Expr.FoldParameters(simplifiedExpr, newP, out newP);
      //     newSimplifiedStr = simplifiedExpr.ToString();
      //     Console.WriteLine(newSimplifiedStr);
      //     Console.WriteLine(string.Join(" ", newP.Select(pi => pi.ToString("e4"))));
      //   } while (newSimplifiedStr != oldSimplifiedStr);
      // }
    }


    [Test]
    public void Graphviz() {
      Expression<Expr.ParametricFunction> expr = (p, x) => 2.0 * x[0] + x[0] * p[0] + x[1] + Math.Log(x[1] * p[1] + 1.0) + 1 / (x[1] * p[2]);
      Console.WriteLine(Expr.ToGraphViz(expr));
      Console.WriteLine(Expr.ToGraphViz(expr, new double[] { 0.0, 1.0, 2.0 }));
      Console.WriteLine(Expr.ToGraphViz(expr, varNames: new[] { "a", "b" }));

      var sat = new Dictionary<Expression, double>();
      var rand = new Random(1234);
      foreach (var node in FlattenExpressionVisitor.Execute(expr)) {
        if (node.NodeType == ExpressionType.Multiply) {
          sat.Add(node, rand.NextDouble());
        }
      }
      Console.WriteLine(Expr.ToGraphViz(expr, saturation: sat));
    }

    [Test]
    public void CollectTerms() {
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => p[0] + p[1] + x[0] + x[1] + 3.0;
        var terms = CollectTermsVisitor.CollectTerms(expr);
        Assert.AreEqual(5, terms.Count());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => 3.0 + Math.Log(p[0]) + p[0] * x[0];
        var terms = CollectTermsVisitor.CollectTerms(expr);
        Assert.AreEqual(3, terms.Count());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => 3.0 + (p[0] + x[0]);
        var terms = CollectTermsVisitor.CollectTerms(expr);
        Assert.AreEqual(3, terms.Count());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => (3.0 + p[0]) * x[0];
        var terms = CollectTermsVisitor.CollectTerms(expr);
        Assert.AreEqual(1, terms.Count());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => -(3.0 + p[0]) + -(3.0);
        var terms = CollectTermsVisitor.CollectTerms(expr);
        Assert.AreEqual(3, terms.Count());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => Math.Log(3.0 + p[0]) + 3.0;
        var terms = CollectTermsVisitor.CollectTerms(expr);
        Assert.AreEqual(2, terms.Count());
      }

    }

    [Test]
    public void CollectFactors() {
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => p[0] + p[1];
        var factors = CollectFactorsVisitor.CollectFactors(expr);
        Assert.AreEqual(1, factors.Count());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => p[0] * p[1] * x[0] * x[1] * 3.0;
        var factors = CollectFactorsVisitor.CollectFactors(expr);
        Assert.AreEqual(5, factors.Count());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => 3.0 * Math.Log(p[0]) * (p[0] + x[0]);
        var factors = CollectFactorsVisitor.CollectFactors(expr);
        Assert.AreEqual(3, factors.Count());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => 3.0 * (p[0] * x[0]);
        var factors = CollectFactorsVisitor.CollectFactors(expr);
        Assert.AreEqual(3, factors.Count());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => (3.0 * p[0]) + x[0];
        var factors = CollectFactorsVisitor.CollectFactors(expr);
        Assert.AreEqual(1, factors.Count());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => -(3.0 + p[0]) * -(3.0);
        var factors = CollectFactorsVisitor.CollectFactors(expr);
        Assert.AreEqual(2, factors.Count());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => -(3.0 * p[0]) * -(3.0);
        var factors = CollectFactorsVisitor.CollectFactors(expr);
        Assert.AreEqual(3, factors.Count());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => Math.Log(3.0 * p[0]) * 3.0;
        var factors = CollectFactorsVisitor.CollectFactors(expr);
        Assert.AreEqual(2, factors.Count());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => x[0] / x[1];
        var factors = CollectFactorsVisitor.CollectFactors(expr);
        Assert.AreEqual(2, factors.Count());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => x[0] / (x[1] * x[2]);
        var factors = CollectFactorsVisitor.CollectFactors(expr);
        Assert.AreEqual(3, factors.Count());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => x[0] / (x[1] / x[2]);
        var factors = CollectFactorsVisitor.CollectFactors(expr);
        Assert.AreEqual(3, factors.Count());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => x[0] / (x[1] / x[2] * x[3]);
        var factors = CollectFactorsVisitor.CollectFactors(expr);
        Assert.AreEqual(4, factors.Count());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => (x[0] / x[1]) / (x[1] * x[2] * x[3]);
        var factors = CollectFactorsVisitor.CollectFactors(expr);
        Assert.AreEqual(5, factors.Count());
      }
    }

    [Test]
    public void FixRedundantParameters() {
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => (p[0] + x[1] + p[1]);
        var theta = expr.Parameters[0];
        var thetaValues = new double[2];
        var newExpr = FixRedundantParametersVisitor.FixRedundantParameters(expr, theta, thetaValues);
        Assert.AreEqual("(p, x) => ((p[0] + x[1]) + 0)", newExpr.ToString());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => (p[0] + x[1] + Math.Log(p[1]));
        var theta = expr.Parameters[0];
        var thetaValues = new double[2];
        var newExpr = FixRedundantParametersVisitor.FixRedundantParameters(expr, theta, thetaValues);
        // Assert.AreEqual("", newExpr.ToString()); --> not implemented yet
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => p[0] + (x[1] + Math.Log(p[1]));
        var theta = expr.Parameters[0];
        var thetaValues = new double[2];
        var newExpr = FixRedundantParametersVisitor.FixRedundantParameters(expr, theta, thetaValues);
        // Assert.AreEqual("", newExpr.ToString()); --> not implemented yet
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => (p[0] * p[1] * x[1]);
        var theta = expr.Parameters[0];
        var thetaValues = new double[2];
        var newExpr = FixRedundantParametersVisitor.FixRedundantParameters(expr, theta, thetaValues);
        Assert.AreEqual("(p, x) => ((p[0] * 0) * x[1])", newExpr.ToString());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => (p[0] / p[1] * x[1]);
        var theta = expr.Parameters[0];
        var thetaValues = new double[2];
        var newExpr = FixRedundantParametersVisitor.FixRedundantParameters(expr, theta, thetaValues);
        Assert.AreEqual("(p, x) => ((p[0] / 0) * x[1])", newExpr.ToString());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => (p[0] * (p[1] * x[1] + p[2] * x[1]));
        var theta = expr.Parameters[0];
        var thetaValues = new double[3];
        var newExpr = FixRedundantParametersVisitor.FixRedundantParameters(expr, theta, thetaValues);
        Assert.AreEqual("(p, x) => (0 * ((p[1] * x[1]) + (p[2] * x[1])))", newExpr.ToString());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => (p[0] * (p[1] * x[1] + x[1]));
        var theta = expr.Parameters[0];
        var thetaValues = new double[2];
        var newExpr = FixRedundantParametersVisitor.FixRedundantParameters(expr, theta, thetaValues);
        Assert.AreEqual("(p, x) => (p[0] * ((p[1] * x[1]) + x[1]))", newExpr.ToString());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => (p[0] / (p[1] * x[1] + x[1]));
        var theta = expr.Parameters[0];
        var thetaValues = new double[2];
        var newExpr = FixRedundantParametersVisitor.FixRedundantParameters(expr, theta, thetaValues);
        Assert.AreEqual("(p, x) => (p[0] / ((p[1] * x[1]) + x[1]))", newExpr.ToString());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => (p[0] / (p[1] * x[1] + p[2] * x[1]));
        var theta = expr.Parameters[0];
        var thetaValues = new double[3];
        var newExpr = FixRedundantParametersVisitor.FixRedundantParameters(expr, theta, thetaValues);
        Assert.AreEqual("(p, x) => (0 / ((p[1] * x[1]) + (p[2] * x[1])))", newExpr.ToString());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => (p[1] * x[1] + p[2] * x[1]) / p[0];
        var theta = expr.Parameters[0];
        var thetaValues = new double[3];
        var newExpr = FixRedundantParametersVisitor.FixRedundantParameters(expr, theta, thetaValues);
        Assert.AreEqual("(p, x) => (((p[1] * x[1]) + (p[2] * x[1])) / 0)", newExpr.ToString());
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => Math.Sqrt(p[1] * x[1] + p[2] * x[1]) * p[0];
        var theta = expr.Parameters[0];
        var thetaValues = new double[3];
        var newExpr = FixRedundantParametersVisitor.FixRedundantParameters(expr, theta, thetaValues);
        // Assert.AreEqual("", newExpr.ToString()); --> not implemented yet
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => Math.Log(p[1] * x[1] + p[2] * x[1]) * p[0];
        var theta = expr.Parameters[0];
        var thetaValues = new double[3];
        var newExpr = FixRedundantParametersVisitor.FixRedundantParameters(expr, theta, thetaValues);
        // Assert.AreEqual("", newExpr.ToString()); --> not implemented yet
      }
      {
        Expression<Expr.ParametricFunction> expr = (p, x) => Math.Exp(p[1] * x[1]) * p[0];
        var theta = expr.Parameters[0];
        var thetaValues = new double[2];
        var newExpr = FixRedundantParametersVisitor.FixRedundantParameters(expr, theta, thetaValues);
        Assert.AreEqual("(p, x) => (Exp((p[1] * x[1])) * p[0])", newExpr.ToString());
      }
    }

    private static void CompileAndRun(Expression<Expr.ParametricFunction> expr) {
      int N = 10;
      var X = new double[N, 3];
      var f = new double[N];
      var t = new double[5] { 1.0, 2.0, 3.0, 4.0, 5.0 };
      Expr.Broadcast(expr).Compile()(t, X, f);
    }

    private static void CompileAndRunHessian(Expression<Expr.ParametricFunction> expr) {
      var x = new double[3];
      var t = new double[5] { 1.0, 2.0, 3.0, 4.0, 5.0 };
      var h = new double[5, 5];
      var hessian = Expr.Hessian(expr, 5);
      hessian.Compile()(t, x, h);
    }

    private static void CompileAndRunJacobian(Expression<Expr.ParametricGradientFunction> expr) {
      int N = 10;
      var X = new double[N, 3];
      var f = new double[N];
      var t = new double[5] { 1.0, 2.0, 3.0, 4.0, 5.0 };
      var J = new double[N, 5];
      Expr.Broadcast(expr).Compile()(t, X, f, J);
    }

    private static void CompareSymbolicAndAutoDiffJacobian(Expression<Expr.ParametricFunction> expr) {
      int N = 10;
      var rand = new Random(1234);
      var X = new double[N, 3];
      for (int i = 0; i < N; i++) for (int j = 0; j < 3; j++) X[i, j] = i * 3 + j + 1;
      var f = new double[N];
      var t = new double[5] { 1.0, 2.0, 3.0, 4.0, 5.0 };
      var symJ = new double[N, 5];
      var autoJ = new double[N, 5];
      Expr.Broadcast(Expr.Gradient(expr, expr.Parameters.Count)).Compile()(t, X, f, symJ);
      Expr.Evaluate(expr, t, X, ref autoJ);
      for(int i=0;i<N;i++) {
        for (int j = 0; j < 5; j++)
          Assert.AreEqual(symJ[i, j], autoJ[i, j], 1e-6);
      }
    }   
  }
}

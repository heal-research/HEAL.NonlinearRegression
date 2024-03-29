using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HEAL.Expressions.Tests {
  [TestClass]
  public class ExprTests {
    [TestInitialize]
    public void Setup() {
      System.Threading.Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
      System.Threading.Thread.CurrentThread.CurrentUICulture = System.Globalization.CultureInfo.InvariantCulture;
    }

    [TestMethod]
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
    }

    [TestMethod]
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
    }

    [TestMethod]
    public void Autodiff() {
      CompareSymbolicAndAutoDiffJacobian((p, x) => p[0] * x[0]);
      CompareSymbolicAndAutoDiffJacobian((a, b) => a[0] * b[0]);
      CompareSymbolicAndAutoDiffJacobian((p, x) => p[0] - x[0]);
      CompareSymbolicAndAutoDiffJacobian((p, x) => p[0] + x[0]);
      CompareSymbolicAndAutoDiffJacobian((p, x) => p[0] / x[0]);
      CompareSymbolicAndAutoDiffJacobian((p, x) => x[0] / p[0]);
      CompareSymbolicAndAutoDiffJacobian((p, x) => p[1] + x[1]);
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Abs(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Log(p[0] + x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Exp(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Sqrt(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Sqrt(Math.Abs(p[0] * x[0])));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Sin(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Cos(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Tanh(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Pow(x[0], p[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Pow(x[0], 2.0));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Pow(x[0], 3.0));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Pow(p[0] * x[0], p[1]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Pow(Math.Abs(p[0] + x[0]), p[1]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Functions.PowAbs(p[0] + x[0], p[1]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Functions.PowAbs(p[0], x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => p[0] * (Math.Pow(Math.Abs(p[1] + x[0]), p[2]) + x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Functions.Sign(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Functions.Cbrt(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Functions.Logistic(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Functions.InvLogistic(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Functions.LogisticPrime(p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Functions.InvLogisticPrime(p[0] * x[0]));

      CompareSymbolicAndAutoDiffJacobian((p, x) => p[0] * x[0] / (p[1] * x[1] + p[2]));

      // example with duplicate sub-expressions
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Pow(p[0] * x[0], p[1]) + Math.Pow(p[0] * x[0], p[1]) / (p[0] * x[0]));
      CompareSymbolicAndAutoDiffJacobian((p, x) => Math.Cos(Functions.Cbrt(x[0])* p[0]) * p[1]);
    }

    [TestMethod]
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
        Assert.AreEqual("(p, x) => (((p[0] * x[0]) * 2) * x[0])", dfx_dx.ToString());
        // (p, x) => 
      }
      {
        var dfx_dx = Expr.Derive((p, x) => Functions.Cbrt(p[0] * x[0]), 0);
        CompileAndRun(dfx_dx);
        Assert.AreEqual("(p, x) => ((0.3333333333333333 / Pow(Cbrt((p[0] * x[0])), 2)) * x[0])", dfx_dx.ToString());
        // (p, x) => (((2 * x[0]) * p[0]) * x[0])
      }
    }

    [TestMethod]
    public void Hessian() {
      CompileAndRunHessian((double[] p, double[] x) => Math.Log(p[0] + x[0]));
      CompileAndRunHessian((double[] p, double[] x) => Math.Exp(p[0] * x[0]));
      CompileAndRunHessian((double[] p, double[] x) => Math.Sin(p[0] * x[0]));
      CompileAndRunHessian((double[] p, double[] x) => Math.Cos(p[0] * x[0]));
    }

    [TestMethod]
    public void FoldParameters() {
      {
        var paramValues = new[] { 2.0, 2.0, 3.0, 4.0 };
        var expr = Expr.Simplify((p, x) => (p[0] + p[1]), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => p[0]", expr.ToString());
        Assert.AreEqual(4.0, newParamValues[0]);
      }
      {
        var paramValues = new[] { 2.0, 2.0, 3.0, 4.0 };
        var expr = Expr.Simplify((p, x) => (p[0] * (x[0] * p[1])), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => (x[0] * p[0])", expr.ToString());
        Assert.AreEqual(4.0, newParamValues[0]);
      }
      {
        var paramValues = new[] { 2.0, 2.0, 3.0, 4.0 };
        var expr = Expr.Simplify((p, x) => (p[0] + (x[0] + p[1])), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => (x[0] + p[0])", expr.ToString());
        Assert.AreEqual(4.0, newParamValues[0]);
      }
      {
        var paramValues = new[] { 2.0, 3.0, 4.0, 5.0 };
        var expr = Expr.Simplify((p, x) => (p[0] / (x[0] * p[1] + p[2])), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => (p[0] / (x[0] + p[1]))", expr.ToString());
        Assert.AreEqual(2.0 / 3.0, newParamValues[0]);
        Assert.AreEqual(4.0 / 3.0, newParamValues[1]);
      }
      {
        var paramValues = new[] { 2.0, 2.0, 3.0, 4.0 };
        var expr = Expr.Simplify((p, x) => Math.Log(p[0]), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => p[0]", expr.ToString());
        Assert.AreEqual(Math.Log(2), newParamValues[0]);
      }
      {
        var paramValues = new[] { 2.0, 3.0, 4.0, 5.0 };
        var expr = Expr.Simplify((p, x) => p[2] * (p[0] * x[0] + p[1] * x[1]), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => ((x[0] * p[0]) + (x[1] * p[1]))", expr.ToString());
        Assert.AreEqual(8.0, newParamValues[0]);
        Assert.AreEqual(12.0, newParamValues[1]);
      }
      {
        var paramValues = new[] { 2.0, 3.0, 4.0, 5.0 };
        var expr = Expr.Simplify((p, x) => 1.0 / (p[0] * x[0] + p[1] * x[1]) * p[2], paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => (p[0] / ((x[1] * p[1]) + x[0]))", expr.ToString());
        Assert.AreEqual(4.0 / 2.0, newParamValues[0]);
        Assert.AreEqual(3.0 / 2.0, newParamValues[1]);
      }
      {
        var paramValues = new[] { 2.0 };
        var expr = Expr.Simplify((p, x) => Math.Log(p[0]), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => p[0]", expr.ToString());
        Assert.AreEqual(Math.Log(2.0), newParamValues[0]);
      }
      {
        var paramValues = new[] { 2.0, 3.0 };
        var expr = Expr.Simplify((p, x) => Math.Pow(p[0], p[1]), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => p[0]", expr.ToString());
        Assert.AreEqual(8, newParamValues[0]);
      }
      {
        var paramValues = new[] { 2.0, 3.0, 4.0, 5.0 };
        var expr = Expr.Simplify((p, x) => (p[0] * x[0] + p[1] * x[1]) / (p[2] * x[0] + x[1]) * p[3], paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => (((x[1] * p[0]) + (x[0] * p[1])) / ((x[0] * p[2]) + x[1]))", expr.ToString());
        Assert.AreEqual(3.0 * 5.0, newParamValues[0]);
        Assert.AreEqual(2.0 * 5.0, newParamValues[1]);
        Assert.AreEqual(4.0, newParamValues[2]);
      }
      {
        var paramValues = new[] { 2.0, 3.0, 4.0, 5.0 };
        var expr = Expr.Simplify((p, x) => (p[0] * x[0] + p[1] * x[1]) / (p[2] * x[0] + p[3] * x[1]), paramValues, out var newParamValues);
        Assert.AreEqual("(p, x) => (((x[1] * p[0]) + (x[0] * p[1])) / ((x[1] * p[2]) + x[0]))", expr.ToString());
        Assert.AreEqual(3.0 / 4.0, newParamValues[0]);
        Assert.AreEqual(2.0 / 4.0, newParamValues[1]);
        Assert.AreEqual(5.0 / 4.0, newParamValues[2]);
      }
    }

    [TestMethod]
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

    [TestMethod]
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
        Assert.AreEqual("(p, x) => (x[0] * p[4])", expr.ToString());
        Assert.AreEqual(-5.0, newTheta[4]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => -(x[0] * p[0] * -(p[1] * x[0]));
        var theta = new double[] { 2.0, 3.0, 4.0 };
        var expr = LowerNegation(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => (Pow(x[0], 2) * p[5])", expr.ToString());
        Assert.AreEqual(6.0, newTheta[5]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => -(Math.Sin(p[0]));
        var theta = new double[] { 2.0, 3.0, 4.0 };
        var expr = LowerNegation(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => p[4]", expr.ToString());
        Assert.AreEqual(-Math.Sin(theta[0]), newTheta[4]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => -(Functions.Cbrt(p[0]));
        var theta = new double[] { 2.0, 3.0, 4.0 };
        var expr = LowerNegation(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => p[4]", expr.ToString());
        Assert.AreEqual(-Functions.Cbrt(theta[0]), newTheta[4]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => -(Math.Cos(p[0]));
        var theta = new double[] { 2.0, 3.0, 4.0 };
        var expr = LowerNegation(f, theta, out var newTheta);
        Assert.AreEqual("(p, x) => p[4]", expr.ToString());
        Assert.AreEqual(-Math.Cos(theta[0]), newTheta[4]);
      }
    }

    private static Expression<Expr.ParametricFunction> LowerNegation(Expression<Expr.ParametricFunction> f, double[] theta, out double[] newTheta) {
      var expr = RuleBasedSimplificationVisitor.Simplify(new ParameterizedExpression(f, f.Parameters[0], theta));
      newTheta = expr.pValues;
      return expr.expr;
    }

    [TestMethod]
    public void Simplify() {
      {
        Expression<Expr.ParametricFunction> f = (p, x) => p[0] - p[1] * (p[2] * x[0] + p[3]);
        var theta = new double[] { 2.0, 3.0, 4.0, 5.0 };
        var simplifiedExpr = Expr.Simplify(f, theta, out var newP);
        Assert.AreEqual("(p, x) => ((x[0] * p[0]) + p[1])", simplifiedExpr.ToString());
        Assert.AreEqual(-12.0, newP[0]);
        Assert.AreEqual(-13.0, newP[1]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => 1.0 / x[0] * x[1] * p[0];
        var theta = new double[] { 2.0, 3.0, 4.0, 5.0 };
        var simplifiedExpr = Expr.Simplify(f, theta, out var newP);
        Assert.AreEqual("(p, x) => ((x[1] * p[0]) / x[0])", simplifiedExpr.ToString());
        Assert.AreEqual(2.0, newP[0]);
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => 1 / (x[0] * p[0] + x[1] * p[1] + p[2]) * (x[2] * p[3] + p[4]) * p[5];
        var theta = new double[] { 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
        var simplifiedExpr = Expr.Simplify(f, theta, out var newP);
        Assert.AreEqual("(p, x) => (((x[2] * p[0]) + p[1]) / (((x[1] * p[2]) + x[0]) + p[3]))", simplifiedExpr.ToString());
      }
      {
        Expression<Expr.ParametricFunction> f = (p, x) => p[0] + p[1] * Math.Sqrt(p[2] * x[0] + Math.Sqrt(Math.Sqrt(Math.Sqrt(p[3] * x[1] * p[4] * x[1]) * p[5] * x[2] * p[6] * x[1] + p[7] * x[3] * p[8] * x[1])));
        var theta = new double[] { 5.42128, -1.12871, -0.230468, 0.531836, 0.693147, 0.531836, 0.693147, 3.14159, 1.77245 };
        Console.WriteLine(f.ToString());
        Console.WriteLine(string.Join(" ", theta.Select(pi => pi.ToString("e4"))));
        var simplifiedExpr = Expr.Simplify(f, theta, out var newP);
        var newSimplifiedStr = simplifiedExpr.ToString();
        Console.WriteLine(newSimplifiedStr);
        Console.WriteLine(string.Join(" ", newP.Select(pi => pi.ToString("e4"))));
        string oldSimplifiedStr;
        // simplify until no change (TODO: this shouldn't be necessary if foldParameters is implemented correctly)
        do {
          oldSimplifiedStr = newSimplifiedStr;
          simplifiedExpr = Expr.Simplify(simplifiedExpr, newP, out newP);
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
        var simplifiedExpr = Expr.Simplify(f, theta, out var newP);
        var newSimplifiedStr = simplifiedExpr.ToString();
        Console.WriteLine(newSimplifiedStr);
        Console.WriteLine(string.Join(" ", newP.Select(pi => pi.ToString("e4"))));
        string oldSimplifiedStr;
        // simplify until no change (TODO: this shouldn't be necessary if foldParameters is implemented correctly)
        do {
          oldSimplifiedStr = newSimplifiedStr;
          simplifiedExpr = Expr.Simplify(simplifiedExpr, newP, out newP);
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


    [DataTestMethod]
    [DataRow("x - x", "0")]
    [DataRow("1.0 - 2.0", "p[0]")]
    [DataRow("1.0f - 2.0f", "-1")]
    [DataRow("x / x", "1")]
    [DataRow("x - x", "0")]
    [DataRow("0f ** x", "0")]
    [DataRow("x + (-1f)", "(x[0] - 1)")]
    [DataRow("x - 1f", "(x[0] - 1)")]
    [DataRow("2.0 / x", "(p[0] / x[0])")]
    [DataRow("1.0 / 2.0", "p[0]")]
    [DataRow("abs(-x)", "Abs(x[0])")]
    [DataRow("pow(1f, x)", "1")]
    [DataRow("x*x*x", "Pow(x[0], 3)")]
    [DataRow("x*x*x*x", "Pow(x[0], 4)")]
    [DataRow("x*x/x", "x[0]")]
    [DataRow("x/(x*x)", "(1 / x[0])")]
    [DataRow("1/powabs(x1, x2)", "(p[0] / PowAbs(x[1], x[2]))")] // already minimal
    [DataRow("1f/powabs(x1, x2)", "(1 / PowAbs(x[1], x[2]))")]
    [DataRow("x1 / powabs(x1, x2)", "(x[1] / PowAbs(x[1], x[2]))")]
    [DataRow("powabs(x1, x2) / x1", "(PowAbs(x[1], x[2]) / x[1])")]
    [DataRow("1.0 - x - x - x", "((x[0] * -3) + p[0])")]
    [DataRow("1.0 * x + 2.0 * x", "(x[0] * p[0])")]
    [DataRow("1.0 * x - 2.0 * x", "(x[0] * p[0])")]
    [DataRow("(1 / x) + (x - (1 / x))", "((p[0] / x[0]) + x[0])")]
    [DataRow("(1f / x) + (x - (1f / x))", "x[0]")]
    [DataRow("1f / (x * 2.0 + x1 * 3.0 + 4.0) * (x2 * 5.0 + 6.0) * 7.0", "(((x[2] * p[0]) + p[1]) / (((x[1] * p[2]) + x[0]) + p[3]))")]
    [DataRow("1f / (2 - x)", "(1 / (p[0] - x[0]))")] // this expression is minimal and should not be transformed into something longer
    [DataRow("1f / ((1f / x) - x)", "(1 / ((1 / x[0]) - x[0]))")] // this is already minimal
    [DataRow("(2.0 - x) - x", "((x[0] * -2) + p[0])")]
    [DataRow("x - (2.0 - x)", "((x[0] * 2) + p[0])")]
    [DataRow("(2.0 + x) * x", "((x[0] + p[0]) * x[0])")] // already minimal
    [DataRow("x / (powabs (x, x))", "(x[0] / PowAbs(x[0], x[0]))")] // already minimal
    [DataRow("2.0 / (powabs (2.0, x))", "(p[0] / PowAbs(p[1], x[0]))")] // already minimal
    [DataRow("abs(2.0f)", "2")]
    [DataRow("abs(2.0)", "p[0]")]
    [DataRow("log(abs(2.0))", "p[0]")]
    [DataRow("sqrt(abs(2.0))", "p[0]")]
    [DataRow("powabs(2.0, 3.0)", "p[0]")]
    [DataRow("powabs(2.0, x)", "PowAbs(p[0], x[0])")] // here we cannot remove abs() because the base should not be negative for real-valued exponents
    [DataRow("2.0 + -x * 3.0", "((x[0] * p[0]) + p[1])")]
    [DataRow("(2.0 - x) * 3.0", "((x[0] * p[0]) + p[1])")]
    [DataRow("(2.0 - x) / 3.0", "((x[0] * p[0]) + p[1])")]
    [DataRow("x / Pow(x, 3.0)", "Pow(x[0], p[0])")] // with p[0] = -2
    [DataRow("x / Pow(x1, 3.0)", "(x[0] * Pow(x[1], p[0]))")] // with p[0] = -3
    [DataRow("powabs(1.0f / x, 3.0)", "PowAbs(x[0], p[0])")] // with p[0] = -3
    [DataRow("powabs(1.0f / x, 2.5)", "PowAbs(x[0], p[0])")] // with p[0] = -2.5, this is even allowed with real exponents
    [DataRow("pow(1.0f / x, 3.0f)", "Pow(x[0], -3)")] // this transformation is allowed for integer exponents
    [DataRow("1f / powabs(x, x)", "(1 / PowAbs(x[0], x[0]))")]
    [DataRow("(x * 2f) / x", "2")]
    [DataRow("3.0 / (2f*x)", "(p[0] / x[0])")]
    [DataRow("x / (2f*x)", "0.5")]
    [DataRow("x / (2*x)", "p[0]")]
    [DataRow("1f/x", "(1 / x[0])")]
    [DataRow("pow(x, -1f)", "(1 / x[0])")]
    [DataRow("powabs (x, -x) * 2.0", "(p[0] / PowAbs(x[0], x[0]))")]
    [DataRow("powabs(powabs(x, 2.0), 3.0)", "PowAbs(x[0], p[0])")]  // | |x|^p1 |^p2 == |x|^(p1*p2)
    [DataRow("powabs(pow(x, 2f), 1.5)", "PowAbs(x[0], p[0])")]
    [DataRow("powabs(pow(x, 3f), 1.5)", "PowAbs(x[0], p[0])")] // x^3 can be negative
    [DataRow("1f / powabs(2.0, 1f / x)", "PowAbs(p[0], (1 / x[0]))")]
    [DataRow("(2.0 / (x * 3.0 + 4.0))", "(p[0] / (x[0] + p[1]))")]
    [DataRow("x / (x * 2.0)", "p[0]")]
    [DataRow("2.0 / (x * 3.0)", "(p[0] / x[0])")]
    [DataRow("1f / (x * 3.0)", "(p[0] / x[0])")]
    [DataRow("2.0 - (3.0 / x)", "((p[0] / x[0]) + p[1])")]
    [DataRow("x - (2.0 / x)", "((p[0] / x[0]) + x[0])")]
    [DataRow("((x * 2) + (-1f / x))", "((x[0] * p[0]) - (1 / x[0]))")]
    [DataRow("x / 2f", "(x[0] * 0.5)")]
    [DataRow("(1f / (2.0 - x)) / (3.0 / (x * x))", "(p[0] / ((p[1] - x[0]) * Pow(x[0], -2)))")]
    [DataRow("(2.0 - x * x - x) * 3.0  ", "(((x[0] + Pow(x[0], 2)) * p[0]) + p[1])")]
    [DataRow("(2.0 - powabs (x, x) - x) * 3.0", "(((x[0] + PowAbs(x[0], x[0])) * p[0]) + p[1])")]
    [DataRow("(1f / x - x) * 2.0", "(((1 / x[0]) - x[0]) * p[0])")] // already minimal
    [DataRow("(x + pow(x,2) + pow(x,3)) * 2.0", "(((x[0] + Pow(x[0], p[0])) + Pow(x[0], p[1])) * p[2])")]
    [DataRow("(2.0 - 1f/x - x) * 2.0", "((((1 / x[0]) + x[0]) * p[0]) + p[1])")]
    [DataRow("2.0 * (3.0 - 1f / x - x)", "((((1 / x[0]) + x[0]) * p[0]) + p[1])")]
    [DataRow("(2.0 - 1f / x - x) / 3.0", "((((1 / x[0]) + x[0]) * p[0]) + p[1])")]
    [DataRow("(2.0 - 1f / x + x) * 3.0", "(((x[0] - (1 / x[0])) * p[0]) + p[1])")]
    [DataRow("((- x) + (x ** 2))", "(Pow(x[0], p[0]) - x[0])")]
    [DataRow("(x / (- x1)) + x2", "(x[2] - (x[0] / x[1]))")]
    [DataRow("x * (-x1) + x2", "(x[2] - (x[0] * x[1]))")]
    [DataRow("x - (2.0 / powabs (x, 1f / x) - x)", "((p[0] / PowAbs(x[0], (1 / x[0]))) + (x[0] * 2))")]
    [DataRow("x - x / (1f / x + x) + x ", "((x[0] * 2) - (x[0] / ((1 / x[0]) + x[0])))")]
    [DataRow("abs(2.0 - x)", "Abs(x[0] + p[0])")]
    [DataRow("powabs(2.0 - x, 3.0)", "PowAbs(x[0] + p[0], p[1])")]
    [DataRow("powabs(2.0 + x1 - x, 3.0)", "PowAbs((x[0] + x[1]) + p[0], p[1])")]
    [DataRow("x - (1f / (2.0 - (- x)))", "(x[0] - (1 / (x[0] + p[0])))")]
    [DataRow("(x ** -2f) * (x ** -3f)", "Pow(x[0], -5)")]
    [DataRow("(x ** 2f) / (x ** 3f)", "(1 / x[0])")]
    [DataRow("((- x) / x)", "-1")]
    [DataRow("x*(-x)", "(-Pow(x[0], 2)))")] // -> - x**2,  inconclusive
    [DataRow("((x ** -2f) * 3.0) / x", "(Pow(x[0], -3) * p[0]")] // -> x**-3 * p
    [DataRow("(1f - x) / x", "((1 / x[0]) - 1)")] // -> 1/x - 1
    [DataRow("(2.0 - (1f / x)) * x", "((x[0] * p[0]) - 1)")]
    [DataRow("(2.0 - x**2) / x", "((p[0] / x[0]) - x[0])")]// (p - (x ** 2)) / x  --> p/x - x
    [DataRow("x * (1f / (pow(abs (x), x)))", "(x[0] / (PowAbs(x[0], x[0])))")] //  --> x / pow(abs(x),x))
    [DataRow("x - 2f", "(x[0] + -2)")] // inconclusive, prefer addition over subtraction?
    [DataRow("((1f / x) + x) * x", "(1 + Pow(x[0], 2))")] //  -> 1 + x**2
    [DataRow("((1f / x) + x) / x", "(Pow(x[0], -2) + 1)")]//   -> x**-2
    [DataRow("abs(2 * x + x + 1.0 ) ** 1.5", "Pow(Abs(((x[0] * p[0]) + x[0]) + p[1]), p[2])")]
    [DataRow("((abs( ((x * 2.0) + (3.0 + x)) )) ** (4.0 + x))", "Pow((Abs(((x[0] * p[0]) + p[1]))), (x[0] + p[2]))")]

    public void SimplifyExpr(string exprStr, string expected) {
      var xParam = Expression.Parameter(typeof(double[]), "x");
      var pParam = Expression.Parameter(typeof(double[]), "p");
      var parser = new Parser.ExprParser(exprStr, new[] { "x", "x1", "x2", "x3" }, xParam, pParam);
      var expr = parser.Parse();
      var p = parser.ParameterValues;
      var simplifiedExpr = Expr.SimplifyRepeated(expr, p, out var newP, debugRules: true);
      Assert.AreEqual(expected, simplifiedExpr.Body.ToString());
    }

    [TestMethod]
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

    [TestMethod]
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

    [TestMethod]
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
      int N = 16;
      var rand = new Random(1234);
      var X = new double[N, 3];
      var colX = new double[3][];
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < 3; j++) {
          X[i, j] = i * 3 + j + 1
            * ((i * 3 + j) % 2 == 0 ? 1 : -1); // alternate sign

          if (i == 0) colX[j] = new double[N];
          colX[j][i] = X[i, j];
        }
      }
      var f1 = new double[N];
      var f2 = new double[N];
      var t = new double[5] { 1.0, -2.0, 3.0, -4.0, 5.0 };
      var J1 = new double[N, 5];
      var J2 = new double[N, 5];
      var jacX = new double[N, 3];

      // 1st option symbolic differentiation and compilation
      Expr.Broadcast(Expr.Gradient(expr, t.Length)).Compile()(t, X, f1, J1);

      // 2nd option: reverse autodiff interpreter
      var interpreter = new ExpressionInterpreter(expr, colX, N);
      interpreter.EvaluateWithJac(t, f2, jacX, J2);

      // 3rd option: numeric check
      var J3 = new double[N, 5];
      var eps = 1e-4;
      for (int j = 0; j < t.Length; j++) {
        t[j] -= eps;
        var fLow = new double[N];
        interpreter.Evaluate(t, fLow);
        t[j] += 2 * eps;
        var fHigh = new double[N];
        interpreter.Evaluate(t, fHigh);

        for (int i = 0; i < N; i++) {
          J3[i, j] = (fHigh[i] - fLow[i]) / (2 * eps);
        }
        t[j] -= eps;
      }

      for (int i = 0; i < N; i++) {
        if (!double.IsNaN(f1[i]) && !double.IsNaN(f2[i]))
          Assert.AreEqual(f1[i], f2[i], 1e-6);
        for (int j = 0; j < 5; j++) {
          if (!double.IsNaN(J1[i, j]) && !double.IsNaN(J2[i, j]) && !double.IsNaN(J3[i, j])) {
            Assert.AreEqual(J1[i, j], J2[i, j], 1e-6);
            Assert.AreEqual(J2[i, j], J3[i, j], Math.Max(1e-3, Math.Abs(J2[i, j] * 1e-3)));
          }
        }
      }
    }
  }
}


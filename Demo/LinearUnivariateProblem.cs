using HEAL.Expressions;
using System;
using System.Linq.Expressions;

namespace HEAL.NonlinearRegression.Demo {
  internal class LinearUnivariateProblem : INLSProblem {

    private static double[] pOpt = new double[] { 2.5, 2 };
    public LinearUnivariateProblem() {
      int m = 20;
      var d = 2;
      X = new double[m, d];
      y = new double[m];

      // generate data
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < d - 1; j++) {
          X[i, j] = i / (double)m * 20 - 10; //  rand.NextDouble() * 20 - 10;
        }
        X[i, d - 1] = 1.0;
      }

      Func(pOpt, X, y); // calculate target

      // and add noise
      var rand = new System.Random(1234);
      for (int i = 0; i < m; i++) y[i] += Util.RandNorm(rand, 0, stdDev: 10);
    }

    public double[,] X { get; private set; }

    public double[] y { get; private set; }

    public double[] ThetaStart => new double[] { .1, .1 };

    public Expression<Expr.ParametricFunction> ModelExpression => (p, x) => p[0] * x[0] + p[1] * x[1];

    public void Func(double[] theta, double[,] X, double[] f) {
      int m = X.GetLength(0);
      int d = X.GetLength(1);
      for (int i = 0; i < m; i++) {
        f[i] = 0;
        for (int j = 0; j < d; j++)
          f[i] += theta[j] * X[i, j];
      }
    }

    public void Jacobian(double[] theta, double[,] X, double[] f, double[,] jac) {
      Func(theta, X, f);
      // for linear models J(f(X)) = X
      // BEWARE: we only fill the first d columns of jac because jac might be larger than x
      int m = X.GetLength(0);
      int d = X.GetLength(1);
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < d; j++)
          jac[i, j] = X[i, j];
      }
    }
  }
}
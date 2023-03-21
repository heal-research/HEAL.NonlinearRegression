using System;

namespace HEAL.NonlinearRegression.Demo {
  internal class LinearProblem : INLSProblem {

    private static double[] pOpt = new double[] { 1, 2, 3, 4 };
    public LinearProblem() {
      int m = 20;
      var d = 4;
      X = new double[m, d];
      y = new double[m];
      var rand = new System.Random(1234);

      // generate data
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < d - 1; j++) {
          X[i, j] = rand.NextDouble() * 2 - 1; // u~(-1,1)
        }
        X[i, d - 1] = 1.0;
      }

      Func(pOpt, X, y); // calculate target

      // and add noise
      for (int i = 0; i < m; i++) y[i] += rand.NextDouble() * 0.2 - 0.1;
    }

    public double[,] X { get; private set; }

    public double[] y { get; private set; }

    public double[] ThetaStart => new double[] { .1, .1, .1, .1 };

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
      Array.Copy(X, jac, X.Length); // for linear models J(f(X)) = X
    }
  }
}
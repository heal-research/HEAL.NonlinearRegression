using System;

namespace HEAL.NonlinearRegression.Demo {
  internal class ExponentialProblem : INLSProblem {

    private static double[] pOpt = new double[] { 0.2, -3.0 };
    public ExponentialProblem() {

      int m = 20;
      X = new double[m, 1];
      y = new double[m];
      var rand = new System.Random(1234);

      // generate data
      for (int i = 0; i < m; i++) {
        X[i, 0] = i / (double)m;
      }

      Func(pOpt, X, y); // calculate target (without noise)
    }

    public double[,] X { get; private set; }

    public double[] y { get; private set; }

    public double[] ThetaStart => new double[] { 1.0, 1.0 };

    public void Func(double[] theta, double[,] X, double[] f) {
      int m = X.GetLength(0);
      int d = X.GetLength(1);
      for (int i = 0; i < m; i++) {
        f[i] = theta[0] * Math.Exp(X[i, 0] * theta[1]);
      }
    }

    public void Jacobian(double[] theta, double[,] X, double[] f, double[,] jac) {
      int m = X.GetLength(0);
      int d = X.GetLength(1);
      Func(theta, X, f);
      for (int i = 0; i < m; i++) {
        jac[i, 0] = Math.Exp(X[i, 0] * theta[1]);
        jac[i, 1] = X[i, 0] * f[i];
      }
    }
  }
}
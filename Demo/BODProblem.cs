using HEAL.Expressions;
using System;
using System.Linq.Expressions;

namespace HEAL.NonlinearRegression.Demo {
  // BOD example from Nonlinear Regression Analysis and Its Applications, Bates and Watts, 1988

  internal class BODProblem : INLSProblem {
    // A 1.4     (there are two BOD datasets)
    private double[] days = new double[] { 1, 2, 3, 4, 5, 7 };
    private double[] BOD = new double[] {
                             8.3,
                             10.3,
                             19.0,
                             16.0,
                             15.6,
                             19.8
                           };

    public double[,] X => Util.ToMatrix(days);

    public double[] y => BOD;

    public double[] ThetaStart => new double[] { 20, 0.24 };

    public Expression<Expr.ParametricFunction> ModelExpression => (double[] theta, double[] x) => theta[0] * (1 - Math.Exp(-theta[1] * x[0]));

    public void Func(double[] theta, double[,] X, double[] f) {
      var m = X.GetLength(0);
      for (int i = 0; i < m; i++) {
        f[i] = theta[0] * (1 - Math.Exp(-theta[1] * X[i, 0]));
      }
    }

    public void Jacobian(double[] theta, double[,] X, double[] f, double[,] jac) {
      Func(theta, X, f);
      var m = X.GetLength(0);
      for (int i = 0; i < m; i++) {
        jac[i, 0] = 1 - Math.Exp(-theta[1] * X[i, 0]);
        jac[i, 1] = theta[0] * X[i, 0] * Math.Exp(-theta[1] * X[i, 0]);
      }
    }
  }
}
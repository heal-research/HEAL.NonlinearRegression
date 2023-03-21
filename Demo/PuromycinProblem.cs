using System;

namespace HEAL.NonlinearRegression.Demo {

  // Puromycin example from Nonlinear Regression Analysis and Its Applications, Bates and Watts, 1988
  internal class PuromycinProblem : INLSProblem {
    // substrate concentration
    private double[] conc = new double[] {
                              0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56, 1.10, 1.10
                            };
    private double[] velocityTreated = new double[] {
                                         76, 47, 97, 107, 123, 139, 159, 152, 191, 201, 207, 200
                                       };
    public double[,] X => Util.ToMatrix(conc);

    public double[] y => velocityTreated;

    public double[] ThetaStart => new double[] { 205, 0.08 };  // Bates and Watts page 41
    // model: y = p1 x / (p2 + x)
    public void Func(double[] theta, double[,] X, double[] f) {
      var m = X.GetLength(0);
      for (int i = 0; i < m; i++) {
        f[i] = theta[0] * X[i, 0] / (theta[1] + X[i, 0]);
      }
    }

    public void Jacobian(double[] theta, double[,] X, double[] f, double[,] jac) {
      Func(theta, X, f);
      var m = X.GetLength(0);
      for (int i = 0; i < m; i++) {
        jac[i, 0] = X[i,0] / (theta[1] + X[i, 0]);
        jac[i, 1] = -theta[0] * X[i, 0] / Math.Pow(theta[1] + X[i, 0], 2);
      }
    }
  }
}
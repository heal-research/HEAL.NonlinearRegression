using System;

namespace HEAL.NonlinearRegression {
  // PCB example from Nonlinear Regression Analysis and Its Applications, Bates and Watts, 1988

  internal class PCBProblem : INLSProblem {
    private double[] age = new double[] {
                  1,
                  1,
                  1,
                  1,
                  2,
                  2,
                  2,
                  3,
                  3,
                  3,
                  4,
                  4,
                  4,
                  5,
                  6,
                  6,
                  6,
                  7,
                  7,
                  7,
                  8,
                  8,
                  8,
                  9,
                  11,
                  12,
                  12,
                  12
                };

    private double[] PCB = new double[] {
                  0.6,
                  1.6,
                  0.5,
                  1.2,
                  2.0,
                  1.3,
                  2.5,
                  2.2,
                  2.4,
                  1.2,
                  3.5,
                  4.1,
                  5.1,
                  5.7,
                  3.4,
                  9.7,
                  8.6,
                  4.0,
                  5.5,
                  10.5,
                  17.5,
                  13.4,
                  4.5,
                  30.4,
                  12.4,
                  13.4,
                  26.2,
                  7.4
                };
    public PCBProblem() {
      

      var m = PCB.Length;

      // model: y = b1 + b2 x
      y = new double[m];
      X = new double[m, 2];
      for (int i = 0; i < m; i++) {
        y[i] = Math.Log(PCB[i]);      // kn(PCB)
        X[i, 0] = 1.0;
        X[i, 1] = Math.Cbrt(age[i]);    // cbrt(age)
      }
    }


    public double[,] X { get; private set; }

    public double[] y { get; private set; }

    public double[] ThetaStart => new double[] { 1.0, 1.0 };

    public void Func(double[] theta, double[,] X, double[] f) {
      var m = X.GetLength(0);
      for (int i = 0; i < m; i++) {
        f[i] = theta[0] * X[i, 0] + theta[1] * X[i, 1];
      }
    }

    public void Jacobian(double[] theta, double[,] X, double[] f, double[,] jac) {
      Func(theta, X, f);
      var m = X.GetLength(0);
      for (int i = 0; i < m; i++) {
        jac[i, 0] = X[i, 0];
        jac[i, 1] = X[i, 1];
      }
    }
  }
}
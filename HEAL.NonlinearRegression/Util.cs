using System;
namespace HEAL.NonlinearRegression {
  internal static class Util {
    public static ResidualFunction CreateResidualFunction(Function func, double[,] x, double[] y) {
      return (p, f) => {
        func(p, x, f);
        for (int i = 0; i < f.Length; i++) {
          f[i] = f[i] - y[i];
        }
      };
    }

    public static ResidualJacobian CreateResidualJacobian(Jacobian jacobian, double[,] x, double[] y) {
      return (p, f, jac) => {
        jacobian(p, x, f, jac);
        for (int i = 0; i < f.Length; i++) {
          f[i] = f[i] - y[i];
        }
        // Jacobian can be passed through ( ∇(f(x,p) - y) == ∇f(x,p) )
      };
    }

    public static ResidualJacobian FixParameter(ResidualJacobian jacobian, int idx) {
      return (p, f, jac) => {
        jacobian(p, f, jac);
        for (int i = 0; i < f.Length; i++) {
          jac[i, idx] = 0.0; // derivative of fixed parameter is zero
        }
      };
    }

    public static alglib.ndimensional_fvec CreateAlgibResidualFunction(ResidualFunction func) {
      return (double[] p, double[] f, object o) => {
        func(p, f);
      };
    }
    public static alglib.ndimensional_jac CreateAlgibResidualJacobian(ResidualJacobian jac) {
      return (double[] p, double[] f, double[,] j, object o) => {
        jac(p, f, j);
      };
    }

    // takes function and reparameterizes it so that it produces paramValue for x0
    // f'(x) = f(x,p[1..k]) - f(x0,p[1..k]) + p[k+1]
    public static Function ReparameterizeFunc(Function func, double[] x0) {
      return (p, x, f) => {
        // calculate f(x0) first
        var _x0 = new double[1, x0.Length];
        Buffer.BlockCopy(x0, 0, _x0, 0, x0.Length * sizeof(double));
        func(p, _x0, f);
        var f_x0 = f[0];

        func(p, x, f); // func ignores the last parameter
        var m = x.GetLength(0);
        for (int i = 0; i < m; i++) {
          f[i] = f[i] - f_x0 + p[p.Length - 1];
        }
      };
    }

    public static Jacobian ReparameterizeJacobian(Jacobian jacobian, double[] x0) {
      return (p, x, f, jac) => {
        var m = jac.GetLength(0);
        var n = jac.GetLength(1) - 1; // the extended function has one extra parameter

        // calculate f(x0) first
        var _x0 = new double[1, x0.Length];
        Buffer.BlockCopy(x0, 0, _x0, 0, x0.Length * sizeof(double));
        jacobian(p, _x0, f, jac);
        var f_x0 = f[0];
        var j_x0 = new double[n];
        for (int j = 0; j < n; j++) {
          j_x0[j] = jac[0, j];
        }

        jacobian(p, x, f, jac);
        for (int i = 0; i < m; i++) {
          f[i] = f[i] - f_x0 + p[p.Length - 1];

          for (int j = 0; j < n; j++) {
            jac[i, j] -= j_x0[j];
          }
          jac[i, n] = 1; // derivative of extra parameter
        }
      };
    }

    // TODO: potentially better to adjust CalcTProfiles and Statistics to take Function and Jacobian parameters

    // creates a new Action to calculate the function output for a fixed dataset
    public static Action<double[], double[]> FuncForX(double[,] x, Function func) {
      return (p, f) => {
        func(p, x, f);
      };
    }

    // creates a new Action to calculate the Jacobian for a fixed dataset x
    public static Action<double[], double[], double[,]> JacobianForX(double[,] x, Jacobian jac) {
      return (p, f, J) => {
        jac(p, x, f, J);
      };
    }
  }
}

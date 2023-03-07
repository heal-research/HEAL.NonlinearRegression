using System;
using System.Linq;
namespace HEAL.NonlinearRegression {
  public static class Util {
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
    public static alglib.ndimensional_grad FixParameter(alglib.ndimensional_grad funcGrad, int idx, double fixedVal) {
      return (double[] p, ref double func, double[] grad, object obj) => {
        p[idx] = fixedVal;
        funcGrad(p, ref func, grad, obj);
        grad[idx] = 0.0; // derivative of fixed parameter is zero
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

    // takes function and re-parameterizes it so that f'(x0) == p[offsetParamIdx]
    // f'(x) = f(x,p[1..k]) - f(x0,p[1..k]) + p[offsetParamIdx]
    public static Function ReparameterizeFuncWithOffset(Function func, double[] x0, int offsetParamIdx) {
      return (p, x, f) => {
        // calculate f(x0) first
        var _x0 = new double[1, x0.Length];
        Buffer.BlockCopy(x0, 0, _x0, 0, x0.Length * sizeof(double));
        func(p, _x0, f);
        var f_x0 = f[0];

        func(p, x, f);
        var m = x.GetLength(0);
        for (int i = 0; i < m; i++) {
          f[i] = f[i] - f_x0 + p[offsetParamIdx];
        }
      };
    }

    // takes function and re-parameterizes it so that f'(x0) == p[scaleParamIdx]
    // f'(x) = f(x,p[1..k]) / f(x0,p[1..k]) * p[scaleParamIdx]
    public static Function ReparameterizeFuncWithScale(Function func, double[] x0, int scaleParamIdx) {
      return (p, x, f) => {
        // calculate f(x0) first
        var _x0 = new double[1, x0.Length];
        Buffer.BlockCopy(x0, 0, _x0, 0, x0.Length * sizeof(double));
        func(p, _x0, f);
        var f_x0 = f[0];

        func(p, x, f);
        var m = x.GetLength(0);
        for (int i = 0; i < m; i++) {
          f[i] = f[i] / f_x0 * p[scaleParamIdx];
        }
      };
    }

    public static Jacobian ReparameterizeJacobianWithOffset(Jacobian jacobian, double[] x0, int offsetParamIdx) {
      return (p, x, f, jac) => {
        var m = jac.GetLength(0);
        var n = jac.GetLength(1) - 1;

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
          f[i] = f[i] - f_x0 + p[offsetParamIdx];

          for (int j = 0; j < n; j++) {
            jac[i, j] -= j_x0[j];
          }
          jac[i, offsetParamIdx] = 1;
        }
      };
    }

    public static Jacobian ReparameterizeJacobianWithScale(Jacobian jacobian, double[] x0, int scaleParamIdx) {
      return (p, x, f, jac) => {
        var m = jac.GetLength(0);
        var n = jac.GetLength(1) - 1;

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
          f[i] = f[i] / f_x0 * p[scaleParamIdx];

          for (int j = 0; j < n; j++) {
            jac[i, j] -= j_x0[j];
          }
          jac[i, scaleParamIdx] = f[i] / f_x0;
        }
      };
    }


    internal static alglib.ndimensional_grad CreateGaussianNegLogLikelihood(Jacobian modelJac, double[] y, double[,] X, double sErr) {
      return (double[] p, ref double f, double[] grad, object obj) => {
        var m = y.Length;
        var n = p.Length;
        var yPred = new double[m];
        var yJac = new double[m, n];
        modelJac(p, X, yPred, yJac);

        f = 0.0;
        Array.Clear(grad, 0, n);
        for (int i = 0; i < m; i++) {
          var res = y[i] - yPred[i];
          f += 0.5 * res * res / (sErr * sErr);
          for (int j = 0; j < n; j++) {
            grad[j] += -res * yJac[i, j] / (sErr * sErr);
          }
        }
      };
    }

    internal static alglib.ndimensional_grad CreateBernoulliNegLogLikelihood(Jacobian modelJac, double[] y, double[,] X) {
      return (double[] p, ref double f, double[] grad, object obj) => {
        var m = y.Length;
        var n = p.Length;
        var yPred = new double[m];
        var yJac = new double[m, n];
        modelJac(p, X, yPred, yJac);

        f = 0.0;
        Array.Clear(grad, 0, n);
        for (int i = 0; i < m; i++) {
          if (y[i] != 0.0 && y[i] != 1.0) throw new ArgumentException("target variable must be binary (0/1) for Bernoulli likelihood");
          var prob = 1.0 / (1 + Math.Exp(-yPred[i]));
          f += -y[i] * Math.Log(prob) - (1 - y[i]) * Math.Log(1 - prob);

          for (int j = 0; j < n; j++) {
            var dProb = Math.Exp(yPred[i]) * yJac[i, j] / Math.Pow(Math.Exp(yPred[i]) + 1, 2);
            grad[j] += -y[i] * dProb / prob - (1 - y[i]) * dProb / (1 - prob);
          }
        }
      };
    }

    public static double Variance(double[] x) {
      var xm = x.Average();
      var SSR = 0.0;
      for (int i = 0; i < x.Length; i++) {
        var r = x[i] - xm;
        SSR += r * r;
      }
      return SSR / x.Length;
    }
  }
}

using System;

namespace HEAL.NonlinearRegression {
  public static class NonlinearRegression {
    public class Report {
      public int NumJacEvals { get; internal set; }
      public int NumFuncEvals { get; internal set; }
      public int Iterations { get; internal set; }
      public bool Success { get; internal set; }

      public Statistics? Statistics { get; internal set; }

      public override string ToString() {
        return $"Successful: {Success}, NumIters: {Iterations}, NumFuncEvals: {NumFuncEvals}, NumJacEvals: {NumJacEvals}";
      }
    }

    /// <summary>
    /// Least-squares fitting for func with Jacobian to target y using initial values p.
    /// Uses Levenberg-Marquardt algorithm.
    /// </summary>
    /// <param name="p">Initial values and optimized parameters on exit. Initial parameters are overwritten.</param>
    /// <param name="func">The model function.</param>
    /// <param name="jacobian">The Jacobian of func. The Action calculates the function values and the Jacobian</param>
    /// <param name="y">Target values</param>
    /// <param name="report">Report with fitting results and statistics</param>
    /// <param name="maxIterations"></param>
    /// <param name="scale">Optional parameter to set parameter scale. Useful if parameters are given on very different measurement scales.</param>
    /// <param name="stepMax">Optional parameter to limit the step size in Levenberg-Marquardt. Can be useful on quickly changing functions (e.g. exponentials).</param>
    /// <param name="callback">A callback which is called on each iteration. Return true to stop the algorithm.</param>
    /// <exception cref="InvalidProgramException"></exception>
    public static void FitLeastSquares(double[] p, Function func, Jacobian jacobian, double[,] x, double[] y,
        out Report report,
        int maxIterations = 0, double[]? scale = null, double stepMax = 0.0, Func<double[], double, bool>? callback = null) {
      int m = y.Length;
      int n = p.Length;
      alglib.minlmcreatevj(m, p, out var state);
      alglib.minlmsetcond(state, epsx: 0.0, maxIterations);
      if (scale != null) alglib.minlmsetscale(state, scale);
      if (stepMax > 0.0) alglib.minlmsetstpmax(state, stepMax);

      void residual(double[] p, double[] fi, object o) {
        func(p, x, fi);
        for (int i = 0; i < m; i++) {
          fi[i] = fi[i] - y[i]; // this order to simplify calculation of Jacobian for residuals (which is a pass-through)
        }
      }

      void residualJac(double[] p, double[] fi, double[,] jac, object o) {
        jacobian(p, x, fi, jac);
        for (int i = 0; i < m; i++) {
          fi[i] = fi[i] - y[i];
        }
      }

      void _rep(double[] x, double f, object o) {
        if (callback != null && callback(x, f)) {
          alglib.minlmrequesttermination(state);
        }
      }
      // alglib.minlmoptguardgradient(state, 1e-6);
      alglib.minlmoptimize(state, residual, residualJac, _rep, obj: null);
      alglib.minlmresults(state, out var pOpt, out var rep);
      // alglib.minlmoptguardresults(state, out var optGuardReport);
      // if (optGuardReport.badgradsuspected) throw new InvalidProgramException();

      if (rep.terminationtype >= 0) {
        Array.Copy(pOpt, p, p.Length);
        var yPred = new double[m];
        func(pOpt, x, yPred);
        var ssr = 0.0;
        for (int i = 0; i < m; i++) {
          var res = y[i] - yPred[i];
          ssr += res * res;
        }

        var statistics = new Statistics(m, n, ssr, yPred, pOpt, JacobianForX(x, jacobian));

        // t-profiles are problematic to calculate when the noise level is too low
        if (statistics.s >= 1e-8) {
          statistics.CalcTProfiles(y, FuncForX(x, func), JacobianForX(x, jacobian));
        }

        report = new Report() {
          Success = true,
          Iterations = rep.iterationscount,
          NumFuncEvals = rep.nfunc,
          NumJacEvals = rep.njac,
          Statistics = statistics
        };
      } else {
        report = new Report() {
          Success = false,
          Iterations = rep.iterationscount,
          NumFuncEvals = rep.nfunc,
          NumJacEvals = rep.njac
        };
      }
    }

    // TODO: potentially better to adjust CalcTProfiles and Statistics to take Function and Jacobian parameters

    // creates a new Action to calculate the function output for a fixed dataset
    private static Action<double[], double[]> FuncForX(double[,] x, Function func) {
      return (p, f) => {
        func(p, x, f);
      };
    }

    // creates a new Action to calculate the Jacobian for a fixed dataset x
    private static Action<double[], double[], double[,]> JacobianForX(double[,] x, Jacobian jac) {
      return (p, f, J) => {
        jac(p, x, f, J);
      };
    }
  }
}

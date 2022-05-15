using System;

namespace HEAL.NonlinearRegression {
  public static partial class NonlinearRegression {
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

    // least squares fitting of func with Jacobian to target y using initial values p.
    // uses Levenberg-Marquardt algorithm.
    // p is updated with optimized parameters if successful.
    // report parameter contains fitting results and statistics

    /// <summary>
    /// Least-squares fitting for func with Jacobian to target y using initial values p.
    /// Uses Levenberg-Marquardt algorithm.
    /// </summary>
    /// <param name="p">Initial values and optimized parameters on exit. Initial parameters are overwritten.</param>
    /// <param name="func"></param>
    /// <param name="jacobian">The jacobian of the function. The Action calculates the function values and the Jacobian</param>
    /// <param name="y">Target values</param>
    /// <param name="report">Report with fitting results and statistics</param>
    /// <param name="maxIterations"></param>
    /// <param name="scale">Optional parameter to set parameter scale. Useful if parameters are given on very different measurement scales.</param>
    /// <param name="stepMax">Optional parameter to limit the step size in Levenberg-Marquardt. Can be useful on quickly changing functions (e.g. exponentials).</param>
    /// <param name="callback">A callback which is called on each iteration. Return true to stop the algorithm.</param>
    /// <exception cref="InvalidProgramException"></exception>
    public static void FitLeastSquares(double[] p, Action<double[], double[]> func, Action<double[], double[], double[,]> jacobian, double[] y,
        out Report report,
        int maxIterations = 0, double[]? scale = null, double stepMax = 0.0, Func<double[], double, bool>? callback = null) {
      int m = y.Length;
      int n = p.Length;
      alglib.minlmcreatevj(m, p, out var state);
      alglib.minlmsetcond(state, epsx: 0.0, maxIterations);
      if (scale != null) alglib.minlmsetscale(state, scale);
      if (stepMax > 0.0) alglib.minlmsetstpmax(state, stepMax);

      void residual(double[] x, double[] fi, object o) {
        func(x, fi);
        for (int i = 0; i < m; i++) {
          fi[i] = fi[i] - y[i]; // this order to simplify calculation of Jacobian for residuals (which is a pass-through)
        }
      }

      void residualJac(double[] x, double[] fi, double[,] jac, object o) {
        jacobian(x, fi, jac);
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
        func(pOpt, yPred);
        var ssr = 0.0;
        for (int i = 0; i < m; i++) {
          var res = y[i] - yPred[i];
          ssr += res * res;
        }

        var statistics = new Statistics() {
          m = m,
          n = n,
          SSR = ssr,
          yPred = yPred,
          paramEst = pOpt,
        };

        statistics.CalcParameterStatistics(jacobian);

        // t-profiles are problematic to calculate when the noise level is too low
        if (statistics.s >= 1e-8) {
          statistics.CalcTProfiles(y, func, jacobian);
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
  }
}

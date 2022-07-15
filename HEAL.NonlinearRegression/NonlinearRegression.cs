using System;
using System.IO;
using System.Linq;

namespace HEAL.NonlinearRegression {
  public class NonlinearRegression {
    public class OptimizationReport {
      public int NumJacEvals { get; internal set; }
      public int NumFuncEvals { get; internal set; }
      public int Iterations { get; internal set; }
      public bool Success { get; internal set; }
      public override string ToString() {
        return $"Successful: {Success}, NumIters: {Iterations}, NumFuncEvals: {NumFuncEvals}, NumJacEvals: {NumJacEvals}";
      }
    }

    private Function func;
    private Jacobian jacobian;

    // results
    private double[] paramEst;

    public double[]? ParamEst { get { return paramEst?.Clone() as double[]; } }

    public OptimizationReport OptReport { get; private set; }
    public LeastSquaresStatistics Statistics { get; private set; }

    public NonlinearRegression() { }

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
    public void Fit(double[] p, Function func, Jacobian jacobian, double[,] x, double[] y,
        int maxIterations = 0, double[]? scale = null, double stepMax = 0.0, Func<double[], double, bool>? callback = null) {

      this.func = func;
      this.jacobian = jacobian;
      int m = y.Length;
      int n = p.Length;
      alglib.minlmcreatevj(m, p, out var state);
      alglib.minlmsetcond(state, epsx: 0.0, maxIterations);
      if (scale != null) alglib.minlmsetscale(state, scale);
      if (stepMax > 0.0) alglib.minlmsetstpmax(state, stepMax);

      void residualFunc(double[] p, double[] fi, object o) {
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
      alglib.minlmoptimize(state, residualFunc, residualJac, _rep, obj: null);
      alglib.minlmresults(state, out paramEst, out var rep);
      // alglib.minlmoptguardresults(state, out var optGuardReport);
      // if (optGuardReport.badgradsuspected) throw new InvalidProgramException();

      if (rep.terminationtype >= 0) {
        Array.Copy(paramEst, p, p.Length);
        Statistics = new LeastSquaresStatistics(m, n, state.f, state.fi, paramEst, JacobianForX(x, jacobian));

        OptReport = new OptimizationReport() {
          Success = true,
          Iterations = rep.iterationscount,
          NumFuncEvals = rep.nfunc,
          NumJacEvals = rep.njac,
        };
      } else {
        // error
        paramEst = null;
        OptReport = new OptimizationReport() {
          Success = false,
          Iterations = rep.iterationscount,
          NumFuncEvals = rep.nfunc,
          NumJacEvals = rep.njac
        };
      }
    }

    public double[] Predict(double[,] x) {
      if (paramEst == null) throw new InvalidOperationException("Call fit first.");
      var m = x.GetLength(0);
      var y = new double[m];
      func(paramEst, x, y);
      return y;
    }

    public double[,] PredictWithIntervals(double[,] x, IntervalEnum intervalType) {
      switch (intervalType) {
        case IntervalEnum.None: {
            var yPred = Predict(x);
            var y = new double[yPred.Length, 1];
            Buffer.BlockCopy(yPred, 0, y, 0, yPred.Length * sizeof(double));
            return y;
          }
        case IntervalEnum.LinearApproximation: {
            throw new NotImplementedException();
            break;
          }
        case IntervalEnum.TProfile: {
            throw new NotImplementedException();
            break;
          }
      }
      throw new InvalidProgramException();
    }

    public void WriteStatistics() {
      WriteStatistics(Console.Out);
    }

    private void WriteStatistics(TextWriter writer) {
      var p = ParamEst;
      var se = Statistics.paramStdError;
      Statistics.GetParameterIntervals(0.05, out var seLow, out var seHigh);
      writer.WriteLine($"SSR {Statistics.SSR:e4} s {Statistics.s:e4}");
      writer.WriteLine($"{"Para"} {"Estimate",14}  {"Std. error",14} {"Lower",14} {"Upper",14} Correlation matrix");
      for (int i = 0; i < Statistics.n; i++) {
        var j = Enumerable.Range(0, i + 1);
        writer.WriteLine($"{i,5} {p[i],14:e4} {se[i],14:e4} {seLow[i],14:e4} {seHigh[i],14:e4} {string.Join(" ", j.Select(ji => Statistics.correlation[i, ji].ToString("f2")))}");
      }
      writer.WriteLine();

      writer.WriteLine($"{"yPred",14} {"low",14}  {"high",14}");
      Statistics.GetPredictionIntervals(0.05, out var predLow, out var predHigh, includeNoise: false);
      for (int i = 0; i < Statistics.m; i++) {
        writer.WriteLine($"{Statistics.yPred[i],14:e4} {predLow[i],14:e4} {predHigh[i],14:e4}");
      }
    }

    #region helper

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
    #endregion
  }
}

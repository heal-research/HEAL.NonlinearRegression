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

    internal Function func;
    internal Jacobian jacobian;
    internal double[,] x;
    internal double[] y;

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
      this.x = (double[,])x.Clone();
      this.y = (double[])y.Clone();

      int m = y.Length;
      int n = p.Length;
      alglib.minlmcreatevj(m, p, out var state);
      alglib.minlmsetcond(state, epsx: 0.0, maxIterations);
      if (scale != null) alglib.minlmsetscale(state, scale);
      if (stepMax > 0.0) alglib.minlmsetstpmax(state, stepMax);

      var alglibResFunc = Util.CreateAlgibResidualFunction(Util.CreateResidualFunction(func, this.x, this.y));
      var alglibResJac = Util.CreateAlgibResidualJacobian(Util.CreateResidualJacobian(jacobian, this.x, this.y));

      void _rep(double[] x, double f, object o) {
        if (callback != null && callback(x, f)) {
          alglib.minlmrequesttermination(state);
        }
      }
      //alglib.minlmoptguardgradient(state, 1e-6);
      alglib.minlmoptimize(state, alglibResFunc, alglibResJac, _rep, obj: null);
      alglib.minlmresults(state, out paramEst, out var rep);
      //alglib.minlmoptguardresults(state, out var optGuardReport);
      //if (optGuardReport.badgradsuspected) throw new InvalidProgramException();

      if (rep.terminationtype >= 0) {
        Array.Copy(paramEst, p, p.Length);
        var yPred = new double[m];
        func(paramEst, x, yPred);
        // recalculate SSR because alglib does not if the estimation is already optimal
        var SSR = 0.0;
        for (int i = 0; i < y.Length; i++) {
          var r = y[i] - yPred[i];
          SSR += r * r;
        }
        Statistics = new LeastSquaresStatistics(m, n, SSR, yPred, paramEst, jacobian, x);

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

    public double[,] PredictWithIntervals(double[,] x, IntervalEnum intervalType, double alpha = 0.05, bool includeNoise = false) {
      var m = x.GetLength(0);
      switch (intervalType) {
        case IntervalEnum.None: {
            var yPred = Predict(x);
            var y = new double[m, 1];
            Buffer.BlockCopy(yPred, 0, y, 0, yPred.Length * sizeof(double));
            return y;
          }
        case IntervalEnum.LinearApproximation: {
            var yPred = Predict(x);
            var y = new double[m, 3];
            Statistics.GetPredictionIntervals(alpha, out var low, out var high, includeNoise);
            for (int i = 0; i < m; i++) {
              y[i, 0] = yPred[i];
              y[i, 1] = low[i];
              y[i, 2] = high[i];
            }
            return y;
          }
        case IntervalEnum.TProfile: {
            if (includeNoise) throw new NotImplementedException("Prediction intervals based on tProfiles with noise not yet supported.");
            var yPred = Predict(x);
            var y = new double[m, 3];
            TProfile.GetPredictionIntervals(x, this, out var low, out var high, alpha, includeNoise);
            for (int i = 0; i < m; i++) {
              y[i, 0] = yPred[i];
              y[i, 1] = low[i];
              y[i, 2] = high[i];
            }
            return y;
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

    }

    #region helper


    #endregion
  }
}

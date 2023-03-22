﻿using System;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using HEAL.Expressions;

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

    internal Function? modelFunc;
    internal Jacobian? modelJacobian;
    internal Expression<Expr.ParametricFunction>? modelExpr;
    public LikelihoodEnum? LikelihoodType { get; private set; }
    internal double[,]? x;
    internal double[]? y;

    // results
    private double[]? paramEst;
    public alglib.ndimensional_grad? NegLogLikelihoodFunc { get; private set; } // mainly for internal use

    public Hessian? FisherInformation { get; private set; }

    public double[]? ParamEst { get { return paramEst?.Clone() as double[]; } }

    public OptimizationReport? OptReport { get; private set; }
    public LaplaceApproximation? Statistics { get; private set; }


    public double NegLogLikelihood {
      // negative log likelihood of the model for the estimated parameters
      get {
        double f = 0.0;
        var grad = new double[paramEst.Length];
        NegLogLikelihoodFunc(ParamEst, ref f, grad, obj: null);
        return f;
      }
    }

    public double Deviance => 2.0 * NegLogLikelihood;
    public double AIC => ModelSelection.AIC(-NegLogLikelihood, Statistics.n);
    public double AICc => ModelSelection.AICc(-NegLogLikelihood, Statistics.n, Statistics.m);
    public double BIC => ModelSelection.BIC(-NegLogLikelihood, Statistics.n, Statistics.m);

    public NonlinearRegression() { }

    /// <summary>
    /// Least-squares fitting for func with Jacobian to target y using initial values p.
    /// Uses Levenberg-Marquardt algorithm.
    /// </summary>
    /// <param name="p">Initial values and optimized parameters on exit. Initial parameters are overwritten.</param>
    /// <param name="expr">The expression (p, x) => to fit. Where p is the parameter vector to be optimized.</param> 
    /// <param name="likelihood">The likelihood type.</param> 
    /// <param name="y">Target values</param>
    /// <param name="report">Report with fitting results and statistics</param>
    /// <param name="maxIterations"></param>
    /// <param name="scale">Optional parameter to set parameter scale. Useful if parameters are given on very different measurement scales.</param>
    /// <param name="stepMax">Optional parameter to limit the step size in Levenberg-Marquardt. Can be useful on quickly changing functions (e.g. exponentials).</param>
    /// <param name="callback">A callback which is called on each iteration. Return true to stop the algorithm.</param>
    /// <exception cref="InvalidProgramException"></exception>
    public void Fit(double[] p, Expression<Expr.ParametricFunction> expr, LikelihoodEnum likelihood, double[,] x, double[] y,
      int maxIterations = 0, double[]? scale = null, double stepMax = 0.0,
      Func<double[], double, bool>? callback = null) {

      this.modelExpr = expr;
      this.LikelihoodType = likelihood;

      var _func = Expr.Broadcast(expr).Compile();
      var _jac = Expr.Jacobian(expr, p.Length).Compile();
      this.modelFunc = (double[] p, double[,] X, double[] f) => _func(p, X, f); // wrapper only necessary because return values are incompatible;
      this.modelJacobian = (double[] p, double[,] X, double[] f, double[,] jac) => _jac(p, X, f, jac);

      this.NegLogLikelihoodFunc = null;
      Hessian fisherInformation = null;

      // TODO extract likelihoods in different base types?
      if (likelihood == LikelihoodEnum.Gaussian) {
        // TODO should we use specified noise error here?
        this.NegLogLikelihoodFunc = Util.CreateGaussianNegLogLikelihood(modelJacobian, y, x, sErr: 1.0);
        this.FisherInformation = Util.CreateGaussianNegLogLikelihoodHessian(modelJacobian, y, sErr: 1.0);
      } else if (likelihood == LikelihoodEnum.Bernoulli) {
        this.NegLogLikelihoodFunc = Util.CreateBernoulliNegLogLikelihood(modelJacobian, y, x);
        this.FisherInformation = Util.CreateBernoulliNegLogLikelihoodHessian(modelJacobian, y);
      }

      Fit(p, fisherInformation, x, y, maxIterations, scale, stepMax, callback);
    }

    private void Fit(double[] p, Hessian fisherInformation, double[,] x, double[] y,
        int maxIterations = 0, double[]? scale = null, double stepMax = 0.0, Func<double[], double, bool>? callback = null) {

      this.x = (double[,])x.Clone();
      this.y = (double[])y.Clone();

      int m = y.Length;
      int n = p.Length;

      #region Conjugate Gradient
      alglib.mincgcreate(p, out var state);
      alglib.mincgsetcond(state, 0.0, 0.0, 0.0, maxIterations);
      if (scale != null) alglib.mincgsetprecdiag(state, scale);
      if (stepMax > 0.0) alglib.mincgsetstpmax(state, stepMax);

      void _rep(double[] x, double f, object o) {
        if (callback != null && callback(x, f)) {
          alglib.mincgrequesttermination(state);
        }
      }
      alglib.mincgoptimize(state, NegLogLikelihoodFunc, _rep, obj: null);
      alglib.mincgresults(state, out paramEst, out var rep);
      #endregion

      if (rep.terminationtype >= 0) {
        Array.Copy(paramEst, p, p.Length);
        // TODO: modelFunc is only required for SSR calculation but this is probably not necessary for LaplaceApproximation
        // evaluate ypred and SSR
        var yPred = new double[m];
        var SSR = 0.0;
        modelFunc(paramEst, x, yPred);
        for (int i = 0; i < yPred.Length; i++) {
          var r = y[i] - yPred[i];
          SSR += r * r;
        }
        Statistics = new LaplaceApproximation(m, n, SSR, yPred, paramEst, fisherInformation, x);

        OptReport = new OptimizationReport() {
          Success = true,
          Iterations = rep.iterationscount,
          NumFuncEvals = rep.nfev
        };
      } else {
        // error
        paramEst = null;
        OptReport = new OptimizationReport() {
          Success = false,
          Iterations = rep.iterationscount,
          NumFuncEvals = rep.nfev
        };
      }
    }


    /// <summary>
    /// Use an existing (fitted) model to initialize NLR without fitting.
    /// </summary>
    /// <param name="p"></param>
    /// <param name="parametricExpr"></param>
    /// <param name="trainX"></param>
    /// <param name="trainY"></param>
    public void SetModel(double[] p, Expression<Expr.ParametricFunction> expr, LikelihoodEnum likelihood, double[,] x, double[] y) {
      var m = y.Length;
      int n = p.Length;


      this.modelExpr = expr;
      var _func = Expr.Broadcast(expr).Compile();
      var _jac = Expr.Jacobian(expr, p.Length).Compile();
      this.modelFunc = (double[] p, double[,] X, double[] f) => _func(p, X, f);
      this.modelJacobian = (double[] p, double[,] X, double[] f, double[,] jac) => _jac(p, X, f, jac);

      this.paramEst = (double[])p.Clone();
      this.x = (double[,])x.Clone();
      this.y = (double[])y.Clone();

      // evaluate ypred and SSR
      var yPred = new double[m];
      var SSR = 0.0;
      modelFunc(paramEst, x, yPred);
      for (int i = 0; i < yPred.Length; i++) {
        var r = y[i] - yPred[i];
        SSR += r * r;
      }

      if (likelihood == LikelihoodEnum.Gaussian) {
        // TODO: should we use the specified noise sigma here?
        var s = Math.Sqrt(SSR / (m - n));
        this.NegLogLikelihoodFunc = Util.CreateGaussianNegLogLikelihood(modelJacobian, y, x, s);
        this.FisherInformation = Util.CreateGaussianNegLogLikelihoodHessian(modelJacobian, y, s);
        Statistics = new LaplaceApproximation(m, n, SSR, yPred, paramEst, FisherInformation, x);
      } else if (likelihood == LikelihoodEnum.Bernoulli) {
        this.NegLogLikelihoodFunc = Util.CreateBernoulliNegLogLikelihood(modelJacobian, y, x);
        this.FisherInformation = Util.CreateBernoulliNegLogLikelihoodHessian(modelJacobian, y);
        Statistics = new LaplaceApproximation(m, n, SSR, yPred, paramEst, FisherInformation, x);
      }
    }

    public double[] Predict(double[,] x) {
      if (paramEst == null) throw new InvalidOperationException("Call fit first.");
      var m = x.GetLength(0);
      var y = new double[m];
      modelFunc(paramEst, x, y);
      return y;
    }

    public double[,] PredictWithIntervals(double[,] x, IntervalEnum intervalType, double alpha = 0.05) {
      var m = x.GetLength(0);
      switch (intervalType) {
        case IntervalEnum.None: {
            var yPred = Predict(x);
            var y = new double[m, 1];
            Buffer.BlockCopy(yPred, 0, y, 0, yPred.Length * sizeof(double));
            return y;
          }
        case IntervalEnum.LaplaceApproximation: {
            var yPred = Predict(x);
            var y = new double[m, 4];
            Statistics.GetPredictionIntervals(modelJacobian, x, alpha, out var resStdErr, out var low, out var high);
            for (int i = 0; i < m; i++) {
              y[i, 0] = yPred[i];
              y[i, 1] = resStdErr[i];
              y[i, 2] = low[i];
              y[i, 3] = high[i];
            }
            return y;
          }
        case IntervalEnum.TProfile: {
            var yPred = Predict(x);
            var y = new double[m, 3];
            TProfile.GetPredictionIntervals(x, this, out var low, out var high, alpha);
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
      var noiseSigma = Statistics.s;
      var mdl = MinimumDescriptionLength.MDL(modelExpr, paramEst, -NegLogLikelihood, y, noiseSigma, x, approxHessian: true);
      var aicc = ModelSelection.AICc(-NegLogLikelihood, Statistics.n, Statistics.m);
      var bic = ModelSelection.BIC(-NegLogLikelihood, Statistics.n, Statistics.m);
      writer.WriteLine($"SSR: {Statistics.SSR:e4} s: {Statistics.s:e4} AICc: {aicc:f1} BIC: {bic:f1} MDL: {mdl:f1}");
      var p = ParamEst;
      var se = Statistics.paramStdError;
      if (se != null) {
        Statistics.GetParameterIntervals(0.05, out var seLow, out var seHigh);
        writer.WriteLine($"{"Para"} {"Estimate",14}  {"Std. error",14} {"z Score",11} {"Lower",14} {"Upper",14} Correlation matrix");
        for (int i = 0; i < Statistics.n; i++) {
          var j = Enumerable.Range(0, i + 1);
          writer.WriteLine($"{i,5} {p[i],14:e4} {se[i],14:e4} {p[i] / se[i],11:e2} {seLow[i],14:e4} {seHigh[i],14:e4} {string.Join(" ", j.Select(ji => Statistics.correlation[i, ji].ToString("f2")))}");
        }
        writer.WriteLine();
      }
    }

    #region helper


    #endregion
  }
}

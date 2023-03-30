using System;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using HEAL.Expressions;
using HEAL.Expressions.Parser;
using HEAL.NonlinearRegression.Likelihoods;

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

    // TODO: these are duplicated in the likelihood 
    internal Expr.ParametricVectorFunction? modelFunc; // for prediction
    internal Expr.ParametricJacobianFunction? modelJacobian; // for Laplace approximation

    internal Expression<Expr.ParametricFunction>? modelExpr;
    // public LikelihoodEnum? LikelihoodType { get; private set; }
    internal double[,]? x;
    internal double[]? y;

    // results
    private double[]? paramEst;

    public double[]? ParamEst { get { return paramEst?.Clone() as double[]; } }

    public OptimizationReport? OptReport { get; private set; }
    public LaplaceApproximation? Statistics { get; private set; }


    // negative log likelihood of the model for the estimated parameters
    public double NegLogLikelihood => Likelihood.NegLogLikelihood(ParamEst);

    // deviance is 2* log likelihood for gaussian case
    // deviance is 2 * (loglike(model) - loglike(optimalModel)) for general likelihoods where optimalModel has one parameter for each output and produces a perfect fit
    // https://en.wikipedia.org/wiki/Deviance_(statistics)

    // for MLE and training data
    public double Deviance => 2.0 * NegLogLikelihood; // for Gaussian: Deviance = SSR /sErr^2

    public LikelihoodBase Likelihood { get; private set; }
    public double Dispersion { get; set; } // for Gaussian: Dispersion = sErr (estimated as Math.Sqrt(SSR / (m-n))); for Bernoulli: Dispersion = 1;
    public double AIC => ModelSelection.AIC(-NegLogLikelihood, Likelihood.NumberOfParameters);
    public double AICc => ModelSelection.AICc(-NegLogLikelihood, Likelihood.NumberOfParameters, Likelihood.NumberOfObservations);
    public double BIC => ModelSelection.BIC(-NegLogLikelihood, Likelihood.NumberOfParameters, Likelihood.NumberOfObservations);

    public NonlinearRegression() { }

    /// <summary>
    /// Least-squares fitting for func with Jacobian to target y using initial values p.
    /// Uses Levenberg-Marquardt algorithm.
    /// </summary>
    /// <param name="modelExpr">The expression to fit including initial parameter values (e.g. 0.1 * x^2f / (1 + 0.2 * x) )</param> 
    /// <param name="variableNames">The variable names occuring in the expression</param> 
    /// <param name="likelihood">The likelihood type.</param> 
    /// <param name="noiseSigma">The noise sigma for the Gaussian likelihood (if known)</param>
    /// <param name="x">Matrix of input values</param>
    /// <param name="y">Vector of target values</param>
    /// <param name="report">Report with fitting results and statistics</param>
    /// <param name="maxIterations"></param>
    /// <param name="scale">Optional parameter to set parameter scale. Useful if parameters are given on very different measurement scales.</param>
    /// <param name="stepMax">Optional parameter to limit the step size in Levenberg-Marquardt. Can be useful on quickly changing functions (e.g. exponentials).</param>
    /// <param name="callback">A callback which is called on each iteration. Return true to stop the algorithm.</param>
    /// <exception cref="InvalidProgramException"></exception>
    public void Fit(string modelExpr, string[] variableNames, LikelihoodEnum likelihood, double[,] x, double[] y, double? noiseSigma = null,
      int maxIterations = 0, double[]? scale = null, double stepMax = 0.0,
      Func<double[], double, bool>? callback = null) {

      // parse expression and extract parameter values
      var variablesParameter = Expression.Parameter(typeof(double[]), "x");
      var parametersParameter = Expression.Parameter(typeof(double[]), "p");
      var parser = new ExprParser(modelExpr, variableNames, variablesParameter, parametersParameter);
      var parametricExpr = parser.Parse();
      var p = parser.ParameterValues;

      Fit(p, parametricExpr, likelihood, x, y, noiseSigma, maxIterations, scale, stepMax, callback);
    }

    /// <summary>
    /// Least-squares fitting for func with Jacobian to target y using initial values p.
    /// Uses Levenberg-Marquardt algorithm.
    /// </summary>
    /// <param name="p">Initial values and optimized parameters on exit. Initial parameters are overwritten.</param>
    /// <param name="expr">The expression (p, x) => to fit. Where p is the parameter vector to be optimized.</param> 
    /// <param name="likelihood">The likelihood type.</param> 
    /// <param name="noiseSigma">The noise sigma for the Gaussian likelihood (if known)</param>
    /// <param name="x">Matrix of input values</param>
    /// <param name="y">Vector of target values</param>
    /// <param name="report">Report with fitting results and statistics</param>
    /// <param name="maxIterations"></param>
    /// <param name="scale">Optional parameter to set parameter scale. Useful if parameters are given on very different measurement scales.</param>
    /// <param name="stepMax">Optional parameter to limit the step size in Levenberg-Marquardt. Can be useful on quickly changing functions (e.g. exponentials).</param>
    /// <param name="callback">A callback which is called on each iteration. Return true to stop the algorithm.</param>
    /// <exception cref="InvalidProgramException"></exception>
    public void Fit(double[] p, Expression<Expr.ParametricFunction> expr, LikelihoodEnum likelihood, double[,] x, double[] y, double? noiseSigma = null,
    int maxIterations = 0, double[]? scale = null, double stepMax = 0.0,
    Func<double[], double, bool>? callback = null) {

      this.modelExpr = expr;

      var _func = Expr.Broadcast(expr).Compile();
      var _jac = Expr.Jacobian(expr, p.Length).Compile();
      this.modelFunc = (double[] p, double[,] X, double[] f) => _func(p, X, f); // wrapper only necessary because return values are incompatible;
      this.modelJacobian = (double[] p, double[,] X, double[] f, double[,] jac) => _jac(p, X, f, jac);

      if (likelihood == LikelihoodEnum.Gaussian) {
        this.Likelihood = new SimpleGaussianLikelihood(x, y, expr, noiseSigma ?? 1.0);
        this.Dispersion = noiseSigma ?? 1.0;
      } else if (likelihood == LikelihoodEnum.Bernoulli) {
        this.Likelihood = new BernoulliLikelihood(x, y, expr, p.Length);
        this.Dispersion = 1.0; // assumed to be 1 for Bernoulli
      }

      Fit(p, x, y, maxIterations, scale, stepMax, callback);

      // if successful
      if (paramEst != null) {
        if (likelihood == LikelihoodEnum.Gaussian && noiseSigma == null) {
          // update dispersion with estimated value after fitting (if noiseSigma was not specified)
          this.Dispersion = Math.Sqrt(Deviance / (y.Length - p.Length)); // s = Math.Sqrt(SSR / (m - n))
          this.Likelihood = new SimpleGaussianLikelihood(x, y, expr, this.Dispersion); // update because we may have an estimate for noise sigma now
        }
        Statistics = new LaplaceApproximation(y.Length, paramEst.Length, paramEst, Likelihood);
      }
    }

    private void Fit(double[] p, double[,] x, double[] y,
        int maxIterations = 0, double[]? scale = null, double stepMax = 0.0, Func<double[], double, bool>? callback = null) {

      this.x = (double[,])x.Clone();
      this.y = (double[])y.Clone();

      int m = y.Length;
      int n = p.Length;

      #region Conjugate Gradient
      alglib.mincgcreate(p, out var state);
      alglib.mincgsetcond(state, 0.0, 0.0, 0.0, maxIterations);
      if (scale != null) {
        alglib.mincgsetscale(state, scale);
        alglib.mincgsetprecdiag(state, scale);
      }
      if (stepMax > 0.0) alglib.mincgsetstpmax(state, stepMax);

      // reporting function for alglib cgoptimize (to allow early stopping)
      void _rep(double[] x, double f, object o) {
        if (callback != null && callback(x, f)) {
          alglib.mincgrequesttermination(state);
        }
      }

      // objective function for alglib cgoptimize
      void objFunc(double[] p, ref double f, double[] grad, object obj) {
        Likelihood.NegLogLikelihoodGradient(p, out f, grad);
      }

      alglib.mincgoptimize(state, objFunc, _rep, obj: null);
      alglib.mincgresults(state, out paramEst, out var rep);
      #endregion

      if (rep.terminationtype >= 0) {
        Array.Copy(paramEst, p, p.Length);

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
    public void SetModel(double[] p, Expression<Expr.ParametricFunction> expr, LikelihoodEnum likelihood, double? noiseSigma, double[,] x, double[] y) {
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

      if (likelihood == LikelihoodEnum.Gaussian) {
        // evaluate ypred and SSR
        var yPred = Predict(x);
        var SSR = 0.0;
        for (int i = 0; i < yPred.Length; i++) {
          var r = y[i] - yPred[i];
          SSR += r * r;
        }

        this.Dispersion = noiseSigma ?? Math.Sqrt(SSR / (m - n));
        this.Likelihood = new SimpleGaussianLikelihood(x, y, modelExpr, this.Dispersion);
      } else if (likelihood == LikelihoodEnum.Bernoulli) {
        this.Dispersion = 1.0;
        this.Likelihood = new BernoulliLikelihood(x, y, modelExpr, p.Length);
      }
      Statistics = new LaplaceApproximation(m, n, paramEst, Likelihood);
    }

    public double[] Predict(double[,] x) {
      if (paramEst == null) throw new InvalidOperationException("Call Fit or SetModel first.");
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
      var mdl = ModelSelection.MDL(modelExpr, paramEst, -NegLogLikelihood, Statistics.diagH);
      if (Likelihood is SimpleGaussianLikelihood) {
        writer.WriteLine($"SSR: {Deviance * Dispersion * Dispersion:e4}  s: {Dispersion:e4} AICc: {AICc:f1} BIC: {BIC:f1} MDL: {mdl:f1}");
      } else if (Likelihood is BernoulliLikelihood) {
        writer.WriteLine($"Deviance: {Deviance:e4}  Dispersion: {Dispersion:e4} AICc: {AICc:f1} BIC: {BIC:f1} MDL: {mdl:f1}");
      }
      var p = ParamEst;
      var se = Statistics.ParamStdError;
      if (se != null) {
        Statistics.GetParameterIntervals(0.05, out var seLow, out var seHigh);
        writer.WriteLine($"{"Para"} {"Estimate",14}  {"Std. error",14} {"z Score",11} {"Lower",14} {"Upper",14} Correlation matrix");
        for (int i = 0; i < p.Length; i++) {
          var j = Enumerable.Range(0, i + 1);
          writer.WriteLine($"{i,5} {p[i],14:e4} {se[i],14:e4} {p[i] / se[i],11:e2} {seLow[i],14:e4} {seHigh[i],14:e4} {string.Join(" ", j.Select(ji => Statistics.Correlation[i, ji].ToString("f2")))}");
        }
        writer.WriteLine();
      }
    }
  }
}

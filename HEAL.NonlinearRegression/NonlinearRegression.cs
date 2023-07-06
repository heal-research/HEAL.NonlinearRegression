using System;
using System.IO;
using System.Linq;
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

    // results
    private double[] paramEst;

    public double[] ParamEst { get { return paramEst?.Clone() as double[]; } }

    public OptimizationReport OptReport { get; private set; }
    public ApproximateLikelihood LaplaceApproximation { get; private set; }


    // negative log likelihood of the model for the estimated parameters
    public double NegLogLikelihood => Likelihood.NegLogLikelihood(paramEst);

    // deviance is 2 * (loglike(model) - loglike(optimalModel)) for general likelihoods where optimalModel has one parameter for each output and produces a perfect fit
    // https://en.wikipedia.org/wiki/Deviance_(statistics)

    // for MLE and training data
    public double Deviance => 2.0 * NegLogLikelihood - 2.0 * Likelihood.BestNegLogLikelihood(ParamEst); // for Gaussian: Deviance = SSR /sErr^2

    public LikelihoodBase Likelihood { get; private set; }
    public double AIC => ModelSelection.AIC(-NegLogLikelihood, Likelihood.NumberOfParameters);
    public double AICc => ModelSelection.AICc(-NegLogLikelihood, Likelihood.NumberOfParameters, Likelihood.NumberOfObservations);
    public double BIC => ModelSelection.BIC(-NegLogLikelihood, Likelihood.NumberOfParameters, Likelihood.NumberOfObservations);

    /// <summary>
    /// Least-squares fitting for func with Jacobian to target y using initial values p.
    /// Uses Levenberg-Marquardt algorithm.
    /// </summary>
    /// <param name="p">Initial values and optimized parameters on exit. Initial parameters are overwritten.</param>
    /// <param name="likelihood">The likelihood type.</param> 
    /// <param name="maxIterations"></param>
    /// <param name="scale">Optional parameter to set parameter scale. Useful if parameters are given on very different measurement scales.</param>
    /// <param name="stepMax">Optional parameter to limit the step size in the conjugate gradients solver. Can be useful on quickly changing functions (e.g. exponentials).</param>
    /// <param name="callback">A callback which is called on each iteration. Return true to stop the algorithm.</param>
    /// <exception cref="InvalidProgramException"></exception>
    public void Fit(double[] p, LikelihoodBase likelihood, int maxIterations = 0, double[] scale = null, double[] diagHess = null, double stepMax = 0.0, double epsF = 0.0,
                    Func<double[], double, bool> callback = null) {
      this.Likelihood = likelihood;

      Fit(p, maxIterations, scale, diagHess, stepMax, epsF, callback);

      // if successful
      if (paramEst != null && !paramEst.Any(double.IsNaN)) {
        // update sErr with estimated value after fitting (if sErr is at the default value for Gaussian likelihood)
        if (likelihood is SimpleGaussianLikelihood gaussLik && gaussLik.SigmaError == 1.0) {
          // TODO: should not set SigmaError when the user specified noiseSigma=1.0 on CL
          gaussLik.SigmaError = Math.Sqrt(Deviance / (likelihood.Y.Length - p.Length)); // s = Math.Sqrt(SSR / (m - n))
        }
        LaplaceApproximation = likelihood.LaplaceApproximation(paramEst);
      }
    }

    private void Fit(double[] p, int maxIterations = 0, double[] scale = null, double[] diagHess = null, double stepMax = 0.0, double epsF = 0.0, Func<double[], double, bool> callback = null) {
      int n = p.Length;
      if (n == 0) return;


      #region L-BFGS
      alglib.minlbfgscreate(Math.Min(30, p.Length), p, out var state); // TODO: check parameters
      alglib.minlbfgssetcond(state,0.0, 1e-6, 1e-3, maxIterations);
      //alglib.minlbfgsoptguardgradient(state, 1e-3);
      // alglib.minlbfgsoptguardsmoothness(state);
      if (scale != null) {
        alglib.minlbfgssetscale(state, scale);
      }
      if (diagHess != null) {
        alglib.minlbfgssetprecdiag(state, diagHess);
      }
      if (stepMax > 0.0) alglib.minlbfgssetstpmax(state, stepMax);

      // reporting function for alglib cgoptimize (to allow early stopping)
      void _rep(double[] x, double f, object o) {
        if (callback != null && callback(x, f)) {
          alglib.minlbfgsrequesttermination(state);
        }
      }

      // objective function for alglib cgoptimize
      void objFunc(double[] x, ref double f, double[] grad, object obj) {
        Likelihood.NegLogLikelihoodGradient(x, out f, grad);
        // var fisher = Likelihood.FisherInformation(x);
        // 
        // if (!fisher.OfType<double>().Any(double.IsNaN) && alglib.spdmatrixcholesky(ref fisher, fisher.GetLength(0), true)) {
        //   alglib.minlbfgssetcholeskypreconditioner(state, fisher, true);
        // }
        if (double.IsNaN(f) || double.IsInfinity(f)) {
          f = 1e300;
          Array.Clear(grad, 0, grad.Length);
        }
      }

      alglib.minlbfgsoptimize(state, objFunc, _rep, obj: null);
      alglib.minlbfgsresults(state, out paramEst, out var rep);
      // alglib.minlbfgsoptguardresults(state, out var optGuardRes);
      // if(optGuardRes.badgradsuspected) {
      //   throw new InvalidProgramException();
      // }
      #endregion


      #region CG
      /*
      alglib.mincgcreate(p, out var state);
      alglib.mincgsetcond(state, 0.0, epsF, 0.0, maxIterations);
      // alglib.mincgoptguardgradient(state, 1e-6);
      // alglib.mincgoptguardsmoothness(state, 1);
      if (scale != null) {
        alglib.mincgsetscale(state, scale);
      }
      if (diagHess != null) {
        alglib.mincgsetprecdiag(state, diagHess);
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
      // alglib.mincgoptguardresults(state, out var optGuardRes);
      */
      #endregion

      // if successful
      if (rep.terminationtype >= 0) {
        Array.Copy(paramEst, p, p.Length);

        OptReport = new OptimizationReport() {
          Success = true,
          Iterations = rep.iterationscount,
          NumFuncEvals = rep.nfev,
          NumJacEvals = rep.nfev
        };
      } else {
        // error
        paramEst = null;
        OptReport = new OptimizationReport() {
          Success = false,
          Iterations = rep.iterationscount,
          NumFuncEvals = rep.nfev,
          NumJacEvals = rep.nfev
        };
      }
    }


    /// <summary>
    /// Use an existing (fitted) model to initialize NLR without fitting.
    /// </summary>
    /// <param name="p">Parameter values</param>
    /// <param name="likelihood">The likelihood function containing the model</param>
    public void SetModel(double[] p, LikelihoodBase likelihood) {
      this.Likelihood = likelihood;

      this.paramEst = (double[])p.Clone();

      LaplaceApproximation = Likelihood.LaplaceApproximation(paramEst);
    }

    /// <summary>
    /// Calculate vector of predictions for input matrix x.
    /// </summary>
    /// <param name="x">Input matrix</param>
    /// <returns>Vector of predictions</returns>
    /// <exception cref="InvalidOperationException">When Fit() or SetModel() has not been called first.</exception>
    public double[] Predict(double[,] x) {
      if (paramEst == null) throw new InvalidOperationException("Call Fit or SetModel first.");
      var interpreter = new ExpressionInterpreter(Likelihood.ModelExpr, Util.ToColumns(x));
      return interpreter.Evaluate(paramEst);
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
            LaplaceApproximation.GetPredictionIntervals(ParamEst, x, alpha, out var resStdErr, out var low, out var high);
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
      var dl = ModelSelection.DL(paramEst, Likelihood);
      var dl2 = ModelSelection.DLLattice(paramEst, Likelihood);
      if (Likelihood is SimpleGaussianLikelihood gaussLik) {
        var ssr = Util.SSR(Predict(Likelihood.X), Likelihood.Y);
        var rmse = Math.Sqrt(ssr / (Likelihood.NumberOfObservations - paramEst.Length));
        var s = gaussLik.SigmaError;
        writer.WriteLine($"SSR: {ssr:e4}  s: {s:e4} RMSE: {rmse:e4} AICc: {AICc:f1} BIC: {BIC:f1} DL: {dl:f1}  DL (lattice): {dl2:f1}");
      } else {
        writer.WriteLine($"Deviance: {Deviance:e4}  AICc: {AICc:f1} BIC: {BIC:f1} DL: {dl:f1}  DL (lattice): {dl2:f1}");
      }
      LaplaceApproximation?.WriteStatistics(writer);
    }
  }
}

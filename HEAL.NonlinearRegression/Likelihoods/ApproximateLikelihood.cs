using HEAL.Expressions;
using System;
using System.Linq;
using System.Linq.Expressions;
namespace HEAL.NonlinearRegression {

  // Laplace approximation of a likelihood function around its MLE
  public class ApproximateLikelihood : LikelihoodBase {
    private readonly double minNegLogLik;
    private readonly double[] pOpt;
    private readonly double[,] hessian;

    internal ApproximateLikelihood(ApproximateLikelihood original) : base(original) {
      this.minNegLogLik = original.minNegLogLik;
      this.pOpt = original.pOpt;
      this.hessian = original.hessian;
    }
    
    // TODO: x and y are not necessary. -> remove
    public ApproximateLikelihood(double[,] x, double[] y, Expression<Expr.ParametricFunction> modelExpr, double minNegLogLik, double[] pOpt, double[,] hessian)
      : base(modelExpr, x, y, numLikelihoodParams: 0) {
      this.minNegLogLik = minNegLogLik;
      this.pOpt = pOpt;
      // gradient is assumed to be zero in the MLE
      this.hessian = hessian;
    }

    public override double[,] FisherInformation(double[] p) {
      return hessian;
    }

    // for the calculation of deviance
    public override double BestNegLogLikelihood(double[] p) {
      return minNegLogLik;
    }

    public override double NegLogLikelihood(double[] p) {
      NegLogLikelihoodGradient(p, out var nll, nll_grad: null);
      return nll;
    }

    public override void NegLogLikelihoodGradient(double[] p, out double nll, double[] nll_grad) {
      var n = p.Length;
      nll = minNegLogLik;

      if (nll_grad != null) Array.Clear(nll_grad, 0, n);

      for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {
          nll += 0.5 * (p[i] - pOpt[i]) * hessian[i, j] * (p[j] - pOpt[j]);
          nll_grad[i] += +hessian[i, j] * (p[j] - pOpt[j]); // https://stats.stackexchange.com/questions/90134/gradient-of-multivariate-gaussian-log-likelihood
        }
      }
    }

    // only valid for maximum likelihood estimte
    // these values can be called in the CTOR
    public void CalcParameterStatistics(double[] pOpt, out double[] paramStdError, out double[,] invH, out double[,] correlation) {
      var n = pOpt.Length;

      var U = (double[,])hessian.Clone();

      // copy diagonal of Fisher information (for preconditioning in CG)
      var diagH = new double[n];
      for (int i = 0; i < n; i++) diagH[i] = U[i, i];

      try {
        if (alglib.spdmatrixcholesky(ref U, n, isupper: true) == false) {
          throw new InvalidOperationException("Cannot decompose Hessian (not SDP?)");
        }
        alglib.spdmatrixcholeskyinverse(ref U, n, isupper: true, out var info, out var rep, null); // calculates (U^T U) ^-1 = H^-1
        if (info < 0) {
          throw new InvalidOperationException("Cannot invert Hessian");
        }
      } catch (alglib.alglibexception) {
        Console.Error.WriteLine("LaplaceApproximation: Cannot decompose or invert Hessian");
        throw;
      }

      invH = U; U = null; // rename 

      // fill up rest of invH because alglib only works on the upper triangle (prevents problems below)
      for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
          invH[j, i] = invH[i, j];
        }
      }

      // invH is the covariance matrix
      // se is diagonal of covariance matrix
      paramStdError = new double[n];
      correlation = new double[n, n];
      for (int i = 0; i < n; i++) {
        paramStdError[i] = Math.Sqrt(invH[i, i]);
      }

      for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
          correlation[i, j] = invH[j, i] / (paramStdError[i] * paramStdError[j]);
          if (i != j) correlation[j, i] = correlation[i, j];
        }
      }
    }

    // only valid for maximum likelihood estimate
    public void GetParameterIntervals(double[] pOpt, double alpha, out double[] low, out double[] high) {
      var n = pOpt.Length;

      low = new double[n];
      high = new double[n];

      CalcParameterStatistics(pOpt, out var paramStdError, out _, out _);

      var t = -alglib.invstudenttdistribution(NumberOfObservations - n, alpha / 2.0);

      for (int i = 0; i < n; i++) {
        low[i] = pOpt[i] - paramStdError[i] * t;
        high[i] = pOpt[i] + paramStdError[i] * t;
      }
    }



    // only valid for maximum likelihood estimate
    public void GetPredictionIntervals(double[] pOpt, double[,] x, double alpha, out double[] resStdError, out double[] low, out double[] high) {
      var n = pOpt.Length;
      int numRows = x.GetLength(0);
      low = new double[numRows];
      high = new double[numRows];
      resStdError = new double[numRows];

      double[,] yJac = new double[numRows, n];

      CalcParameterStatistics(pOpt, out _, out var invH, out _);

      // cannot use the interpreter of the likelihood because we now need to evaluate the model for a new dataset x
      var interpreter = new ExpressionInterpreter(ModelExpr, Util.ToColumns(x), numRows);
      var yPred = interpreter.EvaluateWithJac(pOpt, null, yJac);

      for (int i = 0; i < numRows; i++) {
        resStdError[i] = 0.0;
        var row = new double[n];
        for (int j = 0; j < n; j++) {
          for (int k = 0; k < n; k++) {
            row[j] += invH[j, k] * yJac[i, k];
          }
        }
        for (int j = 0; j < n; j++) {
          resStdError[i] += yJac[i, j] * row[j];
        }
        resStdError[i] = Math.Sqrt(resStdError[i]);
      }

      // point-wise interval
      // https://en.wikipedia.org/wiki/Confidence_and_prediction_bands
      var t = -alglib.invstudenttdistribution(NumberOfObservations - n, alpha / 2);

      for (int i = 0; i < numRows; i++) {
        low[i] = yPred[i] - resStdError[i] * t;
        high[i] = yPred[i] + resStdError[i] * t;
      }
    }

    public override LikelihoodBase Clone() {
      return new ApproximateLikelihood(this);
    }

    public void WriteStatistics(System.IO.TextWriter writer) {
      CalcParameterStatistics(pOpt, out var se, out _, out var correlation);
      GetParameterIntervals(pOpt, 0.05, out var seLow, out var seHigh);
      writer.WriteLine($"{"Para"} {"Estimate",14}  {"Std. error",14} {"z Score",11} {"Lower",14} {"Upper",14} Correlation matrix");
      for (int i = 0; i < pOpt.Length; i++) {
        var j = Enumerable.Range(0, i + 1);
        writer.WriteLine($"{i,5} {pOpt[i],14:e4} {se[i],14:e4} {pOpt[i] / se[i],11:e2} {seLow[i],14:e4} {seHigh[i],14:e4} {string.Join(" ", j.Select(ji => correlation[i, ji].ToString("f2")))}");
      }
      writer.WriteLine();
    }
  }
}

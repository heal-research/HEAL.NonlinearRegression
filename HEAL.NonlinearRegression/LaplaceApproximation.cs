using HEAL.Expressions;
using System;
using System.Linq;

namespace HEAL.NonlinearRegression {
  public class LaplaceApproximation {
    public double[] ParamEst { get; internal set; } // estimated values for parameters θ
    public double[] ParamStdError { get; internal set; } // standard error for parameters (se(θ) in Bates and Watts)
    public double[,] Correlation { get; internal set; }// correlation matrix for parameters


    private double[,] invH; // covariance matrix for training set (required for prediction intervals)
    public double[] diagH; // used for preconditioning of CG in profile likelihood (TODO: public visibility problematic)
    private int m; // number of observations

    public LaplaceApproximation(double[] paramEst, LikelihoodBase likelihood) {
      this.ParamEst = (double[])paramEst.Clone();
      this.m = likelihood.Y.Length;
      try {
        CalcParameterStatistics(likelihood);
      } catch (Exception) {
        Console.Error.WriteLine($"Problem while calculating statistics. Prediction intervals will not work.");
      }
    }

    private void CalcParameterStatistics(LikelihoodBase likelihood) {
      var pOpt = ParamEst;
      var n = pOpt.Length;

      var U = likelihood.FisherInformation(pOpt); // Hessian is symmetric positive definite in pOpt
      
      // copy diagonal of Fisher information (for preconditioning in CG)
      diagH = new double[n];
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
      var se = new double[n];
      var C = new double[n, n];
      for (int i = 0; i < n; i++) {
        se[i] = Math.Sqrt(invH[i, i]);
      }

      for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
          C[i, j] = invH[j, i] / se[i] / se[j];
          if (i != j) C[j, i] = C[i, j];
        }
      }

      Correlation = C;
      ParamStdError = se.ToArray();
    }

    public void GetParameterIntervals(double alpha, out double[] low, out double[] high) {
      var n = ParamEst.Length;

      low = new double[n];
      high = new double[n];

      var t = -alglib.invstudenttdistribution(m - n, alpha / 2.0);

      for (int i = 0; i < n; i++) {
        low[i] = ParamEst[i] - ParamStdError[i] * t;
        high[i] = ParamEst[i] + ParamStdError[i] * t;
      }
    }



    public void GetPredictionIntervals(Expr.ParametricJacobianFunction modelJacobian, double[,] x, double alpha, out double[] resStdError, out double[] low, out double[] high) {
      var n = ParamEst.Length;
      int numRows = x.GetLength(0);
      low = new double[numRows];
      high = new double[numRows];
      resStdError = new double[numRows];

      var yPred = new double[numRows];
      var J = new double[numRows, n];
      modelJacobian(ParamEst, x, yPred, J);

      for (int i = 0; i < numRows; i++) {
        resStdError[i] = 0.0;
        var row = new double[n];
        for (int j = 0; j < n; j++) {
          for (int k = 0; k < n; k++) {
            row[j] += invH[j, k] * J[i, k];
          }
        }
        for (int j = 0; j < n; j++) {
          resStdError[i] += J[i, j] * row[j];
        }
        resStdError[i] = Math.Sqrt(resStdError[i]);
      }

      // point-wise interval
      // https://en.wikipedia.org/wiki/Confidence_and_prediction_bands
      var t = -alglib.invstudenttdistribution(this.m - n, alpha / 2);

      for (int i = 0; i < numRows; i++) {
        low[i] = yPred[i] - resStdError[i] * t;
        high[i] = yPred[i] + resStdError[i] * t;
      }
    }
  }
}
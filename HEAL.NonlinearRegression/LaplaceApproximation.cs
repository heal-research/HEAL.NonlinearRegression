using System;
using System.Linq;

namespace HEAL.NonlinearRegression {
  public class LaplaceApproximation {
    public int m { get; internal set; } // number of observations
    public int n { get; internal set; } // number of parameters
    public double[] yPred { get; internal set; }
    public double SSR { get; internal set; } // sum of squared residuals, S(θ) in Bates and Watts
    public double s => Math.Sqrt(SSR / (m - n)); // s²: residual mean square or variance estimate based on m-n degrees of freedom 
    public double[] paramEst { get; internal set; } // estimated values for parameters θ
    public double[] paramStdError { get; internal set; } // standard error for parameters (se(θ) in Bates and Watts)
    public double[,] correlation { get; internal set; }// correlation matrix for parameters


    private double[,] invH;

    public LaplaceApproximation(int m, int n, double SSR, double[] yPred, double[] paramEst, Hessian negLogLikeHessian, double[,] x) {
      this.m = m;
      this.n = n;
      this.SSR = SSR;
      this.yPred = yPred;
      this.paramEst = (double[])paramEst.Clone();
      try {
        CalcParameterStatistics(negLogLikeHessian, x);
      } catch (Exception e) {
        System.Console.Error.WriteLine($"Problem while calculating statistics. Prediction intervals will not work.");
      }
    }

    private void CalcParameterStatistics(Hessian negLogLikeHessian, double[,] x) {
      int m = x.GetLength(0);
      var pOpt = paramEst;

      // var yPred = new double[m];
      var U = new double[n, n];
      negLogLikeHessian(pOpt, x, U); // Hessian is symmetric positive definite in pOpt
      try {
        alglib.spdmatrixcholesky(ref U, n, isupper: true);
        alglib.spdmatrixcholeskyinverse(ref U, n, isupper: true, out var info, out var rep, null); // calculates (U^T U) ^-1 = H^-1
        invH = U; U = null; // rename 
        // fill up rest of invH because alglib only works on the upper triangle (prevents problems below)
        for (int i = 0;i<n-1;i++) {
          for(int j=i+1;j<n;j++) {
            invH[j, i] = invH[i, j];
          }
        }


        // if (info < 0) {
        //   System.Console.Error.WriteLine("Jacobian is not of full rank or contains NaN values");
        //   throw new InvalidOperationException("Cannot invert R");
        // }
      } catch (alglib.alglibexception) {
        System.Console.Error.WriteLine("LaplaceApproximation: Cannot invert Hessian of likelihood");
        throw;
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

      correlation = C;
      paramStdError = se.Select(sei => sei * s).ToArray(); // not sure why multiplication with s is needed here (should be contained within invH already)
    }

    public void GetParameterIntervals(double alpha, out double[] low, out double[] high) {
      low = new double[n];
      high = new double[n];

      // for approximate confidence interval of each parameter
      var t = alglib.invstudenttdistribution(m - n, 1 - alpha / 2.0);

      for (int i = 0; i < n; i++) {
        low[i] = paramEst[i] - paramStdError[i] * t;
        high[i] = paramEst[i] + paramStdError[i] * t;
      }
    }



    public void GetPredictionIntervals(Jacobian jacobian, double[,] x, double alpha, out double[] resStdError, out double[] low, out double[] high, bool includeNoise = false) {
      int numRows = x.GetLength(0);
      low = new double[numRows];
      high = new double[numRows];
      resStdError = new double[numRows];

      var yPred = new double[numRows];
      var J = new double[numRows, n];
      jacobian(paramEst, x, yPred, J); // jacobian for the model

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
        resStdError[i] = s* Math.Sqrt(resStdError[i]);
      }
      // 
      // // https://en.wikipedia.org/wiki/Confidence_and_prediction_bands
      var f = alglib.invfdistribution(n, this.m - n, alpha);
      var t = alglib.invstudenttdistribution(this.m - n, 1 - alpha / 2);

      var noiseStdDev = includeNoise ? s : 0.0;

      // Console.WriteLine($"noiseStdDev: {noiseStdDev} f: {f}, Math.Sqrt(n * f) {Math.Sqrt(n * f)} t: {t}");

      //   // point-wise interval
      for (int i = 0; i < numRows; i++) {
        low[i] = yPred[i] - (resStdError[i] + noiseStdDev) * t;
        high[i] = yPred[i] + (resStdError[i] + noiseStdDev) * t;
      }
      // old code to calculate pointwise and simultaneous intervals
      // if (m == 1) {
      //   // point-wise interval
      //   low[0] = yPred[0] - (resStdError[0] + noiseStdDev) * t;
      //   high[0] = yPred[0] + (resStdError[0] + noiseStdDev) * t;
      // } else {
      //   // simultaneous interval (band)
      //   for (int i = 0; i < m; i++) {
      //     low[i] = yPred[i] - (resStdError[i] + noiseStdDev)* Math.Sqrt(n * f); // not sure if t or sqrt(n*f) should be used for the noise part
      //     high[i] = yPred[i] + (resStdError[i] + noiseStdDev) * Math.Sqrt(n * f);
      //   }
      // }
    }


  }
}
using System;
using System.Runtime.ExceptionServices;

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

      var yPred = new double[m];
      var U = new double[n, n];
      negLogLikeHessian(pOpt, x, yPred, U); // Hessian is symmetric positive definite in pOpt
      double[,] invH;
      try {
        // clear lower part of Hessian (required by alglib.cholesky)
        for (int i = 0; i < n; i++) {
          for (int j = i + 1; j < n; j++) {
            U[j, i] = 0.0;
          }
        }
        alglib.spdmatrixcholesky(ref U, n, isupper: true); // probably we need to clear the lower part of H
        alglib.spdmatrixcholeskyinverse(ref U, n, isupper: true, out var info, out var rep, null);
        invH = U; U = null; // rename 

        // invH is the covariance matrix

        // if (info < 0) {
        //   System.Console.Error.WriteLine("Jacobian is not of full rank or contains NaN values");
        //   throw new InvalidOperationException("Cannot invert R");
        // }
      } catch (alglib.alglibexception) {
        System.Console.Error.WriteLine("LaplaceApproximation: Cannot invert Hessian of likelihood");
        throw;
      }

      // extract U^-1 into diag(|u1|,|u2|, ...|up|) L where L has unit length rows
      var se = new double[n];
      var C = new double[n, n];
      for (int i = 0; i < n; i++) {
        se[i] = Math.Sqrt(invH[i, i]);
      }

      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          C[i, j] = invH[j, i] / se[i] / se[j];
          if (i != j) C[j, i] = C[i, j];
        }
      }

      correlation = C;
      paramStdError = se;
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

      // TODO (similar to above)

      // var yPred = new double[numRows];
      // var J = new double[numRows, n];
      // jacobian(paramEst, x, yPred, J); // jacobian for x
      // 
      // // 1-alpha approximate inference interval for the expected response
      // // (1.37), page 23
      // var J_invR = new double[numRows, n];
      // alglib.rmatrixgemm(numRows, n, n, 1.0, J, 0, 0, 0, invR, 0, 0, 0, 0.0, ref J_invR, 0, 0); // Use invR from trainX because it captures the correlation
      //                                                                                     // and uncertainty of parameters based on the training set.
      // 
      // // s * || J_invR ||
      // for (int i = 0; i < numRows; i++) {
      //   resStdError[i] = 0.0;
      //   for (int j = 0; j < n; j++) {
      //     resStdError[i] += J_invR[i, j] * J_invR[i, j];
      //   }
      //   resStdError[i] = Math.Sqrt(resStdError[i]); // length of row vector in JR
      //   resStdError[i] *= s; // standard error for residuals  (determined on training set)
      // }
      // 
      // // https://en.wikipedia.org/wiki/Confidence_and_prediction_bands
      // var f = alglib.invfdistribution(n, this.m - n, alpha);
      // var t = alglib.invstudenttdistribution(this.m - n, 1 - alpha / 2);
      // 
      // var noiseStdDev = includeNoise ? s : 0.0;
      // 
      // // Console.WriteLine($"noiseStdDev: {noiseStdDev} f: {f}, Math.Sqrt(n * f) {Math.Sqrt(n * f)} t: {t}");
      // 
      // //   // point-wise interval
      // for (int i = 0; i < numRows; i++) {
      //   low[i] = yPred[i] - (resStdError[i] + noiseStdDev) * t;
      //   high[i] = yPred[i] + (resStdError[i] + noiseStdDev) * t;
      // }
      // // old code to calculate pointwise and simultaneous intervals
      // // if (m == 1) {
      // //   // point-wise interval
      // //   low[0] = yPred[0] - (resStdError[0] + noiseStdDev) * t;
      // //   high[0] = yPred[0] + (resStdError[0] + noiseStdDev) * t;
      // // } else {
      // //   // simultaneous interval (band)
      // //   for (int i = 0; i < m; i++) {
      // //     low[i] = yPred[i] - (resStdError[i] + noiseStdDev)* Math.Sqrt(n * f); // not sure if t or sqrt(n*f) should be used for the noise part
      // //     high[i] = yPred[i] + (resStdError[i] + noiseStdDev) * Math.Sqrt(n * f);
      // //   }
      // // }
    }


  }
}
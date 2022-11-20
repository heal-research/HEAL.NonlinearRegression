﻿using System;

namespace HEAL.NonlinearRegression {
  public class LeastSquaresStatistics {
    public int m { get; internal set; } // number of observations
    public int n { get; internal set; } // number of parameters
    public double[] yPred { get; internal set; }
    public double SSR { get; internal set; } // sum of squared residuals, S(θ) in Bates and Watts
    public double s => Math.Sqrt(SSR / (m - n)); // s²: residual mean square or variance estimate based on m-n degrees of freedom 
    public double[] paramEst { get; internal set; } // estimated values for parameters θ
    public double[] paramStdError { get; internal set; } // standard error for parameters (se(θ) in Bates and Watts)
    public double[,] correlation { get; internal set; }// correlation matrix for parameters

    public double LogLikelihood {
      get {
        var v = SSR / m;
        return -m / 2.0 * Math.Log(2 * Math.PI) - m / 2.0 * Math.Log(v) - SSR / (2.0 * v);
      }
    }

    public double AIC => 2 * (n + 1) - 2 * LogLikelihood; // also count noise sigma as a parameter
    public double AICc => AIC + 2 * (n + 1) * (n + 2) / (m - (n + 1) - 1); // sigma is counted as a parameter

    public double BIC => (n + 1) * Math.Log(m) - 2 * LogLikelihood;

    private double[,] invR;

    public LeastSquaresStatistics(int m, int n, double SSR, double[] yPred, double[] paramEst, Jacobian jacobian, double[,] x) {
      this.m = m;
      this.n = n;
      this.SSR = SSR;
      this.yPred = yPred;
      this.paramEst = (double[])paramEst.Clone();
      CalcParameterStatistics(jacobian, x);
    }

    // TODO
    // - Studentized residuals
    // - t-profile confidence intervals for parameters
    // - output for gnuplot

    // Douglas Bates and Donald Watts, Nonlinear Regression and Its Applications, John Wiley and Sons, 1988
    // Appendix A3.2
    // Linear approximation for parameter and inference intervals.
    // Exact for linear models. Good approximation for nonlinear models when parameters are close to linear.
    // Check t profiles and pairwise profile plots for deviation from linearity.
    private void CalcParameterStatistics(Jacobian jacobian, double[,] x) {
      int m = x.GetLength(0);
      var pOpt = paramEst;

      var yPred = new double[m];
      var J = new double[m, n];
      jacobian(pOpt, x, yPred, J);
      // clone J for the QR decomposition
      var QR = (double[,])J.Clone();
      try {
        alglib.rmatrixqr(ref QR, m, n, out _);
        alglib.rmatrixqrunpackr(QR, n, n, out invR); // get R which is inverted in-place in the next statement

        // inverse of R
        alglib.rmatrixtrinverse(ref invR, isupper: true, out var info, out var invReport);
        if (info < 0) {
          System.Console.WriteLine("Jacobian is not of full rank or contains NaN values");
          throw new InvalidOperationException("Cannot invert R");
        }
      } catch (alglib.alglibexception) {
        System.Console.WriteLine("Jacobian is not of full rank or contains NaN values");
        throw;
      }

      // extract R^-1 into diag(|r1|,|r2|, ...|rp|) L where L has unit length rows
      var L = new double[n, n];
      var se = new double[n];
      for (int i = 0; i < n; i++) {
        se[i] = 0;
        for (int j = i; j < n; j++) {
          se[i] += invR[i, j] * invR[i, j];
        }
        se[i] = Math.Sqrt(se[i]); // length of row

        // divide each row by its length to produce L
        for (int j = i; j < n; j++) {
          L[i, j] = invR[i, j] / se[i];
        }
      }

      // multiply each row length by s to give parameter standard errors
      for (int i = 0; i < n; i++)
        se[i] *= s;

      // form correlation matrix LL^T
      var C = new double[n, n];
      alglib.rmatrixgemm(n, n, n, alpha: 1.0, L, 0, 0, optypea: 0, L, 0, 0, optypeb: 1, 0.0, ref C, 0, 0);
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
      int m = x.GetLength(0);
      low = new double[m];
      high = new double[m];
      resStdError = new double[m];

      var yPred = new double[m];
      var J = new double[m, n];
      jacobian(paramEst, x, yPred, J); // jacobian for x

      // 1-alpha approximate inference interval for the expected response
      // (1.37), page 23
      var J_invR = new double[m, n];
      alglib.rmatrixgemm(m, n, n, 1.0, J, 0, 0, 0, invR, 0, 0, 0, 0.0, ref J_invR, 0, 0); // Use invR from trainX because it captures the correlation
                                                                                          // and uncertainty of parameters based on the training set.

      // s * || J_invR ||
      for (int i = 0; i < m; i++) {
        resStdError[i] = 0.0;
        for (int j = 0; j < n; j++) {
          resStdError[i] += J_invR[i, j] * J_invR[i, j];
        }
        resStdError[i] = Math.Sqrt(resStdError[i]); // length of row vector in JR
        resStdError[i] *= s; // standard error for residuals  (determined on training set)
      }

      // https://en.wikipedia.org/wiki/Confidence_and_prediction_bands
      var f = alglib.invfdistribution(n, m - n, alpha);
      var t = alglib.invstudenttdistribution(m - n, 1 - alpha / 2);

      var noiseStdDev = includeNoise ? s : 0.0;
      if (m == 1) {
        // point-wise interval
        low[0] = yPred[0] - (resStdError[0] + noiseStdDev) * t;
        high[0] = yPred[0] + (resStdError[0] + noiseStdDev) * t;
      } else {
        // simultaneous interval (band)
        for (int i = 0; i < m; i++) {
          low[i] = yPred[i] - resStdError[i] * Math.Sqrt(n * f) - t * noiseStdDev; // not sure if t or f should be used for the noise part
          high[i] = yPred[i] + resStdError[i] * Math.Sqrt(n * f) + t * noiseStdDev;
        }
      }
    }
  }
}
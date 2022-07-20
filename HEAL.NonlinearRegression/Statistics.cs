using System;

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

    public double[] resStdError { get; internal set; } // standard error for residuals


    public LeastSquaresStatistics(int m, int n, double SSR, double[] yPred, double[] paramEst, Action<double[], double[], double[,]> jacobian) {
      this.m = m;
      this.n = n;
      this.SSR = SSR;
      this.yPred = (double[])yPred.Clone();
      this.paramEst = (double[])paramEst.Clone();
      CalcParameterStatistics(jacobian);
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
    private void CalcParameterStatistics(Action<double[], double[], double[,]> jacobian) {
      var pOpt = paramEst;

      var yPred = new double[m];
      var J = new double[m, n];
      jacobian(pOpt, yPred, J);
      // clone J for the QR decomposition
      var QR = (double[,])J.Clone();
      alglib.rmatrixqr(ref QR, m, n, out _);
      alglib.rmatrixqrunpackr(QR, n, n, out var R);

      // inverse of R
      alglib.rmatrixtrinverse(ref R, isupper: true, out var info, out var invReport);
      if (info < 0) throw new InvalidOperationException("Cannot invert R");

      // extract R^-1 into diag(|r1|,|r2|, ...|rp|) L where L has unit length rows
      var L = new double[n, n];
      var se = new double[n];
      for (int i = 0; i < n; i++) {
        se[i] = 0;
        for (int j = i; j < n; j++) {
          se[i] += R[i, j] * R[i, j];
        }
        se[i] = Math.Sqrt(se[i]); // length of row

        // divide each row by its length to produce L
        for (int j = i; j < n; j++) {
          L[i, j] = R[i, j] / se[i];
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

      // 1-alpha approximate inference interval for the expected response
      // (1.37), page 23
      var JR = new double[m, n];
      alglib.rmatrixgemm(m, n, n, 1.0, J, 0, 0, 0, R, 0, 0, 0, 0.0, ref JR, 0, 0);
      resStdError = new double[m];

      for (int i = 0; i < m; i++) {
        resStdError[i] = 0.0;
        for (int j = 0; j < n; j++) {
          resStdError[i] += JR[i, j] * JR[i, j];
        }
        resStdError[i] = Math.Sqrt(resStdError[i]); // length of row vector in JR
        resStdError[i] *= s; // standard error for residuals 
      }
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

    public void GetPredictionIntervals(double alpha, out double[] low, out double[] high, bool includeNoise = false) {
      low = new double[m];
      high = new double[m];

      // var f = alglib.invfdistribution(n, m - n, alpha);
      var t = alglib.invstudenttdistribution(m - n, 1 - alpha / 2);

      for (int i = 0; i < m; i++) {
        low[i] = yPred[i] - resStdError[i] * Math.Sqrt(n * t) -(includeNoise ? t*s : 0.0);
        high[i] = yPred[i] + resStdError[i] * Math.Sqrt(n * t) + (includeNoise ? t*s : 0.0);
      }
    }
  }
}
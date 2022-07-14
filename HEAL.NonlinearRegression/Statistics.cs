using System;
using System.Collections.Generic;
using System.Linq;

namespace HEAL.NonlinearRegression {
  public class Statistics {
    public int m { get; internal set; } // number of observations
    public int n { get; internal set; } // number of parameters
    public double[] yPred { get; internal set; }
    public double SSR { get; internal set; } // sum of squared residuals, S(θ) in Bates and Watts
    public double s => Math.Sqrt(SSR / (m - n)); // s²: residual mean square or variance estimate based on m-n degrees of freedom 
    public double[] paramEst { get; internal set; } // estimated values for parameters θ
    public double[] paramStdError { get; internal set; } // standard error for parameters (se(θ) in Bates and Watts)
    public double[,] correlation { get; internal set; }// correlation matrix for parameters

    public double[] resStdError { get; internal set; } // standard error for residuals

    private Tuple<double[], double[][]>[] t_profiles;
    private alglib.spline1dinterpolant[] spline_tau2p, spline_p2tau;
    private alglib.spline1dinterpolant[,] spline_p2q;


    public Statistics(int m, int n, double SSR, double[] yPred, double[] paramEst, Action<double[], double[], double[,]> jacobian) {
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
    // - t-profile prediction intervals


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

    /// <summary>
    /// Produces points on the contour in tau space (taup, tauq) and contour points in the original parameter space (p, q).
    /// (Bates and Watts, Appendix 6)
    /// </summary>
    /// <param name="pIdx">First parameter index</param>
    /// <param name="qIdx">Second parameter index</param>
    /// <param name="alpha">Approximation for 1-alpha confidence region is calculated (e.g. 0.05, value should be between 0.5 and 0.01)</param>
    /// <param name="taup">Contour values for first parameter (in tau scale)</param>
    /// <param name="tauq">Contour values for second parameter (in tau scale)</param>
    /// <param name="p">Contour values for first parameter.</param>
    /// <param name="q">Contour values for second parameter.</param>
    /// <exception cref="InvalidOperationException"></exception>
    public void ApproximateProfilePairContour(int pIdx, int qIdx, double alpha, out double[] taup, out double[] tauq, out double[] p, out double[] q) {
      if (t_profiles == null) throw new InvalidOperationException("t-profiles have not been calculate for this model. The error is too small for reliable determination of t-profiles.");
      // initialize splines of interpolation if necessary
      if (spline_p2tau == null) PrepareSplinesForProfileSketches();

      // scale tau coordinates by dividing by sqrt(n * F(n, m-n, alpha)) 
      // to get a nominal 1 - alpha joint likelihood contour
      var tauScale = Math.Sqrt(n * alglib.invfdistribution(n, m - n, alpha));

      // produce plot for two parameters
      // angles for points on traces as described in Appendix 6
      var anglePairs = new ValueTuple<double, double>[4];
      // anglePairs[0] = (0, alglib.spline1dcalc(spline_tau2gpq[pIdx, qIdx], k));
      // anglePairs[1] = (Math.PI, alglib.spline1dcalc(spline_tau2gpq[pIdx, qIdx], -k));
      // anglePairs[2] = (alglib.spline1dcalc(spline_tau2gpq[qIdx, pIdx], k), 0);
      // anglePairs[3] = (alglib.spline1dcalc(spline_tau2gpq[qIdx, pIdx], -k), Math.PI);

      // from R package 'ellipse'
      double MapTau(double tauA, int aIdx, int bIdx) {
        var a = alglib.spline1dcalc(spline_tau2p[aIdx], tauA * tauScale); // map from tau to a (using t-profile of a)
        var b = alglib.spline1dcalc(spline_p2q[aIdx, bIdx], a); // map from a to b
        var tauB = alglib.spline1dcalc(spline_p2tau[bIdx], b); // map from b to tau (using t-profile of b)
        return tauB / tauScale;
      }
      anglePairs[0] = (0, Math.Acos(MapTau(1, pIdx, qIdx)));
      anglePairs[1] = (Math.PI, Math.Acos(MapTau(-1, pIdx, qIdx)));
      anglePairs[2] = (Math.Acos(MapTau(1, qIdx, pIdx)), 0);
      anglePairs[3] = (Math.Acos(MapTau(-1, qIdx, pIdx)), Math.PI);

      var a = new double[5]; // angle 
      var d = new double[5]; // phase
      for (int j = 0; j < 4; j++) {
        var aj = (anglePairs[j].Item1 + anglePairs[j].Item2) / 2.0;
        var dj = anglePairs[j].Item1 - anglePairs[j].Item2;
        if (dj < 0) {
          dj = -dj;
          aj = -aj;
        }
        a[j] = aj;
        d[j] = dj;
      }
      Array.Sort(a, d, 0, 4);
      a[4] = a[0] + 2 * Math.PI; // period 2*pi
      d[4] = d[0];

      alglib.spline1dbuildcubic(a, d, a.Length, -1, 0, -1, 0, out var spline_ad); // periodic boundary conditions
      var nSteps = 100;
      taup = new double[nSteps]; tauq = new double[nSteps];
      p = new double[nSteps]; q = new double[nSteps];
      for (int i = 0; i < nSteps; i++) {
        var ai = i * Math.PI * 2 / nSteps - Math.PI;
        var di = alglib.spline1dcalc(spline_ad, ai);
        taup[i] = Math.Cos(ai + di / 2) * tauScale;
        tauq[i] = Math.Cos(ai - di / 2) * tauScale;
        p[i] = alglib.spline1dcalc(spline_tau2p[pIdx], taup[i]);
        q[i] = alglib.spline1dcalc(spline_tau2p[qIdx], tauq[i]);
        // Console.WriteLine($"{tau_p} {tau_q} {theta_p} {theta_q}");          
      }
    }


    // Calculate t-profiles for all parameters.
    // Bates and Watts, Appendix A3.5
    internal void CalcTProfiles(double[] y, Action<double[], double[]> func, Action<double[], double[], double[,]> jacobian) {
      var pOpt = paramEst;

      var t_profiles = new Tuple<double[], double[][]>[paramEst.Length]; // for each parameter the tau values and the matrix of parameters

      for (int pIdx = 0; pIdx < n; pIdx++) {
        t_profiles[pIdx] = CalcTProfile(y, func, jacobian, pOpt, pIdx);
      }
    }

    private Tuple<double[], double[][]> CalcTProfile(double[] y, Action<double[], double[]> func, Action<double[], double[], double[,]> jacobian, double[] pOpt, int pIdx) {
      const int kmax = 30;
      const int step = 8;
      var tmax = Math.Sqrt(alglib.invfdistribution(n, m - n, 0.01)); // limit for t (use small alpha here), book page 302

      // buffers
      var yPred_cond = new double[m];
      var jac = new double[m, n];
      var tau = new List<double>();
      var M = new List<double[]>();
      var delta = -paramStdError[pIdx] / step;
      var p_cond = (double[])pOpt.Clone();

      alglib.minlmcreatevj(m, p_cond, out var state);
      alglib.minlmsetcond(state, 0.0, maxits: 30);

      do {
        var t = 0.0; // bug fix to pseudo-code in Bates and Watts
        var invSlope = 1.0;
        for (int k = 0; k < kmax; k++) {
          t = t + invSlope;
          var curP = paramEst[pIdx] + delta * t;

          // _f and _jac wrap func and Jacobian and hold one of the parameters fixed
          void _f(double[] x, double[] fi, object o) {
            // var tup = (Tuple<int, double>)o;

            // var fixedParamIdx = tup.Item1;
            // var fixedParamVal = tup.Item2;
            // x[fixedParamIdx] = fixedParamVal; // probably not necessary if we set the gradient zero in jac
            x[pIdx] = curP;
            func(x, fi);
            for (int i = 0; i < m; i++) {
              fi[i] = fi[i] - y[i];
            }
          }

          void _jac(double[] x, double[] fi, double[,] jac, object o) {
            // var tup = (Tuple<int, double>)o;
            // var fixedParamIdx = tup.Item1;
            // var fixedParamVal = tup.Item2;
            // x[fixedParamIdx] = fixedParamVal;
            x[pIdx] = curP;
            jacobian(x, fi, jac);
            for (int i = 0; i < m; i++) {
              fi[i] = fi[i] - y[i];
              jac[i, pIdx] = 0.0;
            }
          }


          // minimize
          p_cond[pIdx] = curP;
          alglib.minlmrestartfrom(state, p_cond);
          alglib.minlmoptimize(state, _f, _jac, null, null);
          alglib.minlmresults(state, out p_cond, out var report);
          if (report.terminationtype < 0) throw new InvalidProgramException();

          jacobian(p_cond, yPred_cond, jac); // get predicted values and Jacobian for calculation of z and v_p

          var SSR_cond = 0.0; // S(\theta_p)
          var zv = 0.0; // z^T v_p

          for (int i = 0; i < m; i++) {
            var z = y[i] - yPred_cond[i];
            SSR_cond += z * z;
            zv += z * jac[i, pIdx];
          }

          var tau_i = Math.Sign(delta) * Math.Sqrt(SSR_cond - SSR) / s;

          invSlope = Math.Abs(tau_i * s * s / (paramStdError[pIdx] * zv));
          tau.Add(tau_i);
          M.Add((double[])p_cond.Clone());

          invSlope = Math.Min(4.0, Math.Max(invSlope, 1.0 / 16));

          if (Math.Abs(tau_i) > tmax) break;
        }
        delta = -delta; // repeat for other direction
      } while (delta > 0);  // exactly two iterations


      // sort M by tau
      var tauArr = tau.ToArray();
      var mArr = M.ToArray();
      Array.Sort(tauArr, mArr);

      // copy M to transposed (column-oriented) array
      var mArrTransposed = new double[n][]; // column-oriented
      for (int j = 0; j < n; j++) {
        mArrTransposed[j] = new double[tau.Count];
        for (int i = 0; i < mArrTransposed[j].Length; i++) {
          mArrTransposed[j][i] = mArr[i][j];
        }
      }

      return Tuple.Create(tauArr, mArrTransposed);
    }

    private void PrepareSplinesForProfileSketches() {
      // profile pair plots
      spline_tau2p = new alglib.spline1dinterpolant[n];
      spline_p2tau = new alglib.spline1dinterpolant[n];
      spline_p2q = new alglib.spline1dinterpolant[n, n];
      for (int pIdx = 0; pIdx < n; pIdx++) {
        // interpolating spline for p-th column of M as a function of tau
        var tau = t_profiles[pIdx].Item1; // tau

        var p = t_profiles[pIdx].Item2[pIdx]; // p-th column of M_p
        alglib.spline1dbuildcubic(tau, p, out spline_tau2p[pIdx]);   // s tau->theta
        alglib.spline1dbuildcubic(p, tau, out spline_p2tau[pIdx]);   // s theta->tau

        // from Bates and Watts
        // couldn't get the alg. from the book to work
        // for (int qIdx = 0; qIdx < n; qIdx++) {
        //   if (pIdx == qIdx) continue;
        //   var pq = t_profiles[qIdx].Item2[pIdx]; // p th column of Mq
        //   var gpq = new double[pq.Length];
        //   for (int i = 0; i < pq.Length; i++) {
        //     gpq[i] = alglib.spline1dcalc(spline_p2tau[pIdx], pq[i]);
        //     gpq[i] = Math.Acos(gpq[i] / tau[i]);
        //   }
        //   alglib.spline1dbuildcubic(tau, gpq, out spline_tau2gpq[pIdx, qIdx]);
        // }

        // this from R package 'ellipse'
        for (int qIdx = 0; qIdx < n; qIdx++) {
          if (pIdx == qIdx) continue;
          var q = t_profiles[pIdx].Item2[qIdx]; // q-th column of Mp
          alglib.spline1dbuildcubic(p, q, out spline_p2q[pIdx, qIdx]);
        }
      } // prepare splines for interpolation
    }


    public void WriteStatistics(System.IO.TextWriter writer) {
      var p = paramEst;
      var se = paramStdError;
      GetParameterIntervals(0.05, out var seLow, out var seHigh);
      writer.WriteLine($"SSR {SSR:e4} s {s:e4}");
      writer.WriteLine($"{"Para"} {"Estimate",14}  {"Std. error",14} {"Lower",14} {"Upper",14} Correlation matrix");
      for (int i = 0; i < n; i++) {
        var j = Enumerable.Range(0, i + 1);
        writer.WriteLine($"{i,5} {p[i],14:e4} {se[i],14:e4} {seLow[i],14:e4} {seHigh[i],14:e4} {string.Join(" ", j.Select(ji => correlation[i, ji].ToString("f2")))}");
      }
      writer.WriteLine();

      writer.WriteLine($"{"yPred",14} {"low",14}  {"high",14}");
      GetPredictionIntervals(0.05, out var predLow, out var predHigh, includeNoise: false);
      for (int i = 0; i < m; i++) {
        writer.WriteLine($"{yPred[i],14:e4} {predLow[i],14:e4} {predHigh[i],14:e4}");
      }
    }
  }
}
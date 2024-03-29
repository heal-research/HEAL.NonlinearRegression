﻿using HEAL.Expressions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace HEAL.NonlinearRegression {

  public class TProfile {
    private readonly double[] paramEst;
    private readonly double[] paramStdError;
    private readonly int n;
    private readonly int m;

    private readonly Tuple<double[], double[][]>[] t_profiles;
    private readonly alglib.spline1dinterpolant[] spline_tau2p;
    private readonly alglib.spline1dinterpolant[] spline_p2tau;
    private readonly alglib.spline1dinterpolant[,] spline_p2q;

    // Calculate t-profiles for all parameters.
    // Bates and Watts, Appendix A3.5

    public TProfile(double[] paramEst, ApproximateLikelihood laplaceApproximation, LikelihoodBase likelihood) {
      this.paramEst = (double[])paramEst.Clone();
      laplaceApproximation.CalcParameterStatistics(paramEst, out this.paramStdError, out _, out _);
      this.n = paramEst.Length;
      this.m = likelihood.Y.Length;

      t_profiles = new Tuple<double[], double[][]>[n]; // for each parameter the tau values and the matrix of parameters

      for (int pIdx = 0; pIdx < n; pIdx++) {
        t_profiles[pIdx] = CalcTProfile(paramEst, laplaceApproximation, likelihood, pIdx);
      }


      spline_tau2p = new alglib.spline1dinterpolant[n];
      spline_p2tau = new alglib.spline1dinterpolant[n];
      spline_p2q = new alglib.spline1dinterpolant[n, n];

      PrepareSplinesForProfileSketches();
    }

    // p_stud is studentized parameter value
    public void GetProfile(int paramIdx, out double[] p, out double[] tau, out double[] p_stud) {
      var profile = t_profiles[paramIdx];
      var n = profile.Item1.Length;
      p = new double[n];
      tau = new double[n];
      p_stud = new double[n];
      for (int i = 0; i < n; i++) {
        tau[i] = profile.Item1[i];
        p[i] = profile.Item2[paramIdx][i];
        p_stud[i] = (p[i] - paramEst[paramIdx]) / paramStdError[paramIdx];
      }
    }

    public static Tuple<double[], double[][]> CalcTProfile(double[] paramEst, ApproximateLikelihood laplaceApproximation, LikelihoodBase likelihood, int pIdx) {
      const int kmax = 300;
      const int step = 16;

    restart:
      int n = paramEst.Length;
      // TODO: slow (do not recalculate every time)
      laplaceApproximation.CalcParameterStatistics(paramEst, out var paramStdError, out var invH, out _); // approximate value, only used for scaling and to determine initial step size
      var diagH = new double[paramEst.Length];
      for (int i = 0; i < diagH.Length; i++) diagH[i] = 1.0 / invH[i, i];

      // in R: (parameterization taken from: https://github.com/wch/r-source/blob/03f8775bf4ae55129fa76318de2394059613353f/src/library/stats/R/nls-profile.R#L144)
      // > qf(1 - 0.01, 1L, 12 - 2) 10.04429
      // > qf(1 - 0.02, 1L, 12 - 2) 7.638422
      // > qf(1 - 0.05, 1L, 12 - 2) 4.964603

      // in alglib:
      // alglib.invfdistribution(1, 12 - 2, 0.01)
      // 10.044289273396592
      // alglib.invfdistribution(1, 12 - 2, 0.02)
      // 7.6384216175965465
      // alglib.invfdistribution(1, 12 - 2, 0.05)
      // 4.964602743730711
      var tmax = Math.Sqrt(alglib.invfdistribution(1, likelihood.NumberOfObservations - n, 0.001)); // use a small threshold here (alpha smaller or equal to the alpha we use for the query)

      var tau = new List<double>();
      var M = new List<double[]>();
      var delta = -paramStdError[pIdx] / step;

      var nllOpt = likelihood.NegLogLikelihood(paramEst); // calculate maximum likelihood

      #region CG configuration
      alglib.mincgcreate(paramEst, out var state);
      alglib.mincgsetcond(state, 0.0, 0.0, 0.0, 0);
      alglib.mincgsetscale(state, paramStdError);
      alglib.mincgsetprecdiag(state, diagH);
      // alglib.mincgoptguardgradient(state, 1e-8);
      #endregion

      do {
        var t = 0.0; // bug fix to pseudo-code in Bates and Watts
        var invSlope = 1.0;
        var p_cond = (double[])paramEst.Clone();
        for (int k = 0; k < kmax; k++) {
          t += invSlope;
          var curP = paramEst[pIdx] + delta * t;

          // minimize
          p_cond[pIdx] = curP;

          // objective function for alglib cgoptimize
          // likelihood but with p[pIdx] = curP fixed
          void negLogLikeFixed(double[] p, ref double f, double[] grad, object obj) {
            p[pIdx] = curP;
            likelihood.NegLogLikelihoodGradient(p, out f, grad);
            if (double.IsNaN(f)) f = 1e300;
            grad[pIdx] = 0.0;
          }

          #region parameter optimization (CG)
          alglib.mincgrestartfrom(state, p_cond);
          alglib.mincgoptimize(state, negLogLikeFixed, rep: null, obj: null);
          alglib.mincgresults(state, out p_cond, out var report);
          // alglib.mincgoptguardresults(state, out var optGuardRep);
          if (report.terminationtype < 0) break;

          var nll_grad = new double[n];
          likelihood.NegLogLikelihoodGradient(p_cond, out var nll, nll_grad);
          var zv = nll_grad[pIdx];

          if (nll < nllOpt) {
            Array.Copy(p_cond, paramEst, paramEst.Length);
            Console.Error.WriteLine($"Found a new optimum in t-profile calculation theta=({string.Join(", ", p_cond.Select(pi => pi.ToString()))}).");
            goto restart;
          }

          var deviance = 2 * nll;
          var devianceOriginal = 2 * nllOpt;


          var tau_i = Math.Sign(delta) * Math.Sqrt(deviance - devianceOriginal);
          invSlope = Math.Abs(tau_i / (paramStdError[pIdx] * zv));

          // For deviance plot
          // System.Console.WriteLine($"{pIdx},{paramEst[pIdx]},{curP},{deviance - devianceOriginal},{tau_i},{invSlope}");
          #endregion


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
      if (t_profiles == null) throw new InvalidOperationException("Call CalcTProfiles first");

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
        var _a = alglib.spline1dcalc(spline_tau2p[aIdx], tauA * tauScale); // map from tau to a (using t-profile of a)
        var _b = alglib.spline1dcalc(spline_p2q[aIdx, bIdx], _a); // map from a to b
        var tauB = alglib.spline1dcalc(spline_p2tau[bIdx], _b); // map from b to tau (using t-profile of b)
        return Math.Max(-1, Math.Min(1, tauB / tauScale));
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

    public static void GetPredictionIntervals(double[,] x, NonlinearRegression nls, out double[] low, out double[] high, double alpha = 0.05) {
      var predRows = x.GetLength(0); // the points for which we calculate the prediction interval
      var trainRows = nls.Likelihood.Y.Length;
      var n = nls.ParamEst.Length; // number of model parameters
      var d = x.GetLength(1); // number of features

      var _low = new double[predRows];
      var _high = new double[predRows];

      // calc predicted values
      var yPred = nls.Predict(x);

      // we only calculate pointwise intervals
      // in R:
      // > qt(0.01, 10) -2.763769
      // > qt(0.02, 10) -2.359315
      // > qt(0.05, 10) -1.812461
      // in alglib:
      // alglib.invstudenttdistribution(10, 0.01) -2.7637694581126961
      // alglib.invstudenttdistribution(10, 0.02) -2.3593146237365361
      // alglib.invstudenttdistribution(10, 0.05) -1.8124611228116756

      var t = -alglib.invstudenttdistribution(trainRows - n, alpha / 2); // source: https://github.com/cran/MASS/blob/1767aca83144264dac95606edff420855fac260b/R/confint.R#L80


      // prediction intervals for each point in x
      Parallel.For(0, predRows, new ParallelOptions() { MaxDegreeOfParallelism = 12 },
        (i, loopState) => {
          // buffer
          double[] paramEstExt = new double[n];
          Array.Copy(nls.ParamEst, paramEstExt, n);

          var xi = new double[d];
          Buffer.BlockCopy(x, i * d * sizeof(double), xi, 0, d * sizeof(double));

          var reparameterizedModel = Expr.ReparameterizeExpr(nls.Likelihood.ModelExpr, xi, out var outputParamIdx);
          paramEstExt[outputParamIdx] = yPred[i];

          var likelihoodExt = nls.Likelihood.Clone();
          likelihoodExt.ModelExpr = reparameterizedModel; // leads to recompilation (TODO: we can reuse one model if x parameter is extended to include x0)

          var profile = CalcTProfile(paramEstExt, likelihoodExt.LaplaceApproximation(paramEstExt), likelihoodExt, outputParamIdx); // only for the function output parameter

          var tau = profile.Item1;
          var theta = new double[tau.Length];
          for (int k = 0; k < theta.Length; k++) {
            theta[k] = profile.Item2[outputParamIdx][k]; // profile of function output parameter
          }

          // aggregate values for the same tau
          var theta_tau = tau.Zip(theta, (taui, thetai) => (taui, thetai))
          .GroupBy(tup => tup.taui)
          .Select(g => (taui: g.Key, thetai: g.Average(tup => tup.thetai)))
          .OrderBy(tup => tup.taui).ToArray();

          if (theta_tau.Length > 3) {
            alglib.spline1dbuildcubic(
              theta_tau.Select(tup => tup.taui).ToArray(),
              theta_tau.Select(tup => tup.thetai).ToArray(), out var tau2theta);

            // extrapolate the minimum value when the cut-off for the t statistic is not reached
            //  if (tau.Min() > -t) _low[i] = alglib.spline1dcalc(tau2theta, tau.Min());
            if (tau.Min() > -t) _low[i] = double.NaN;
            else _low[i] = alglib.spline1dcalc(tau2theta, -t);

            // extrapolate the maximum value
            // if (tau.Max() < t) _high[i] = alglib.spline1dcalc(tau2theta, tau.Max());
            if (tau.Max() < t) _high[i] = double.NaN;
            else _high[i] = alglib.spline1dcalc(tau2theta, t);
          } else {
            _low[i] = double.NaN;
            _high[i] = double.NaN;
          }
        });

      // cannot manipulate low and high output parameters directly in parallel.for
      low = (double[])_low.Clone();
      high = (double[])_high.Clone();
    }

    private void PrepareSplinesForProfileSketches() {
      // profile pair plots
      for (int pIdx = 0; pIdx < n; pIdx++) {
        // interpolating spline for p-th column of M as a function of tau
        var tau = t_profiles[pIdx].Item1; // tau
        var p = t_profiles[pIdx].Item2[pIdx]; // p-th column of M_p

        var tau_p = tau.Zip(p, (taui, pi) => (taui, pi))
          .GroupBy(tup => tup.taui)
          .Select(g => (taui: g.Key, pi: g.Average(tup => tup.pi)))
          .OrderBy(g => g.taui)
          .ToArray();


        alglib.spline1dbuildcubic(
          tau_p.Select(tup => tup.taui).ToArray(), 
          tau_p.Select(tup => tup.pi).ToArray(), 
          out spline_tau2p[pIdx]);   // s tau->theta
        alglib.spline1dbuildcubic(
          tau_p.Select(tup => tup.pi).ToArray(),
          tau_p.Select(tup => tup.taui).ToArray(), 
          out spline_p2tau[pIdx]);   // s theta->tau

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
  }
}

using HEAL.Expressions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace HEAL.NonlinearRegression {

  public class TProfile {
    private readonly double[] paramEst;
    private readonly double[] paramStdError;
    private readonly int m;
    private readonly int n;

    private readonly Tuple<double[], double[][]>[] t_profiles;
    private readonly alglib.spline1dinterpolant[] spline_tau2p;
    private readonly alglib.spline1dinterpolant[] spline_p2tau;
    private readonly alglib.spline1dinterpolant[,] spline_p2q;

    // Calculate t-profiles for all parameters.
    // Bates and Watts, Appendix A3.5

    public TProfile(double[] y, double[,] x, LaplaceApproximation statistics,
      Function func,
      Jacobian jacobian) {
      this.paramEst = statistics.paramEst;
      this.paramStdError = statistics.paramStdError;
      this.m = statistics.m;
      this.n = statistics.n;

      t_profiles = new Tuple<double[], double[][]>[statistics.n]; // for each parameter the tau values and the matrix of parameters

      for (int pIdx = 0; pIdx < statistics.n; pIdx++) {
        t_profiles[pIdx] = CalcTProfile(y, x, statistics, func, jacobian, pIdx);
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

    public static Tuple<double[], double[][]> CalcTProfile(double[] y, double[,] x, LaplaceApproximation statistics, Function modelFunc, Jacobian modelJac, int pIdx, double alpha = 0.05) {
    restart:
      var paramEst = statistics.paramEst;
      var paramStdError = statistics.paramStdError; // approximate value, only used for scaling and to determine initial step size
      var m = statistics.m;
      var n = statistics.n;

      const int kmax = 300;
      const int step = 16;
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
      var tmax = Math.Sqrt(alglib.invfdistribution(1, m - n, 0.001)); // use a small threshold here (alpha smaller or equal to the alpha we use for the query)

      // buffers
      var yPred_cond = new double[m];
      var J = new double[m, n];
      var tau = new List<double>();
      var M = new List<double[]>();
      var delta = -paramStdError[pIdx] / step;

      var negLogLike = Util.CreateGaussianNegLogLikelihood(modelJac, y, x, statistics.s); // TODO: should we use the noise_stdev here?
      // var negLogLike = Util.CreateBernoulliNegLogLikelihood(modelJac, y, x);
      var nllOpt = 0.0;
      var tempGrad = new double[n];
      negLogLike(paramEst, ref nllOpt, tempGrad, null); // calculate maximum likelihood

      #region CG
      alglib.mincgcreate(paramEst, out var state);
      alglib.mincgsetcond(state, 0.0, 0.0, 0.0, 0);
      alglib.mincgsetscale(state, paramStdError);
      // alglib.mincgoptguardgradient(state, 1e-8);

      #endregion


      #region Levenberg-Marquard
      // alglib.minlmcreatevj(m, p_cond, out var state);
      // // alglib.minlmsetcond(state, 1e-9, 0);
      // alglib.minlmsetscale(state, paramStdError);

      // var resFunc = Util.CreateResidualFunction(modelFunc, x, y); // fix parameter not required here because it is fixed in the Jacobian -> parameter is unchanged
      // // adapted jacobian for fixed parameter
      // var resJacForFixed = Util.FixParameter(Util.CreateResidualJacobian(modelJac, x, y), pIdx);
      // 
      // var alglibResFunc = Util.CreateAlgibResidualFunction(resFunc);
      // var alglibResJacForFixed = Util.CreateAlgibResidualJacobian(resJacForFixed);
      #endregion

      do {
        var t = 0.0; // bug fix to pseudo-code in Bates and Watts
        var invSlope = 1.0;
        var p_cond = (double[])paramEst.Clone();
        for (int k = 0; k < kmax; k++) {
          t = t + invSlope;
          var curP = paramEst[pIdx] + delta * t;

          // minimize
          p_cond[pIdx] = curP;
          var negLogLikeFixed = Util.FixParameter(negLogLike, pIdx, curP);

          #region LM / Gaussian
          // alglib.minlmrestartfrom(state, p_cond);
          // alglib.minlmoptimize(state, alglibResFunc, alglibResJacForFixed, null, null);
          // alglib.minlmresults(state, out p_cond, out var report);
          // if (report.terminationtype < 0) throw new InvalidProgramException();
          // 
          // jac(p_cond, x, yPred_cond, J); // get predicted values and Jacobian for calculation of z and v_p
          // 
          // var SSR_cond = 0.0; // S(,theta_p)
          // var zv = 0.0; // z^T v_p
          // 
          // for (int i = 0; i < m; i++) {
          //   var z = y[i] - yPred_cond[i];
          //   SSR_cond += z * z;
          //   zv += z * J[i, pIdx];
          // }
          // 
          // if (SSR_cond < SSR) throw new ArgumentException($"Found a new optimum in t-profile calculation theta=({string.Join(", ", p_cond.Select(pi => pi.ToString()))}).");
          // 
          // var tau_i = Math.Sign(delta) * Math.Sqrt(SSR_cond - SSR) / s;
          // 
          // invSlope = Math.Abs(tau_i * s * s / (paramStdError[pIdx] * zv));
          #endregion

          #region CG Gaussian
          alglib.mincgrestartfrom(state, p_cond);
          alglib.mincgoptimize(state, negLogLikeFixed, rep: null, obj: null);
          alglib.mincgresults(state, out p_cond, out var report);
          // alglib.mincgoptguardresults(state, out var optGuardRep);
          if (report.terminationtype < 0) break;

          double nll = 0.0;
          var grad = new double[n];
          negLogLike(p_cond, ref nll, grad, null);
          var zv = grad[pIdx];


          // modelJac(p_cond, x, yPred_cond, J); // get predicted values and Jacobian for calculation of z and v_p
          // 
          // var SSR_cond = 0.0; // S(,heta_p)
          // var zv = 0.0; // z^T v_p
          // 
          // for (int i = 0; i < m; i++) {
          //   var z = y[i] - yPred_cond[i];
          //   SSR_cond += z * z;
          //   zv += z * J[i, pIdx];
          // }

          // if (SSR_cond < SSR) throw new ArgumentException($"Found a new optimum in t-profile calculation theta=({string.Join(", ", p_cond.Select(pi => pi.ToString()))}).");

          if (nll < nllOpt) {
            System.Console.Error.WriteLine($"Found a new optimum in t-profile calculation theta=({string.Join(", ", p_cond.Select(pi => pi.ToString()))}).");
            goto restart;
          }

          // TODO: statistics of the NLS model should provide these (based on the likelihoods)
          // deviance is 2* log likelihood for gaussian case
          // deviance is 2 * (loglike(model) - loglike(optimalModel)) for general likelihoods where optimalModel has one parameter for each output and produces a perfect fit
          // https://en.wikipedia.org/wiki/Deviance_(statistics)
          var deviance = 2 * nll;
          var devianceOriginal = 2 * nllOpt;

           var tau_i = Math.Sign(delta) * Math.Sqrt(deviance - devianceOriginal); // TODO: double check the factor 2 here (it is required to produce exactly the same results as LaplaceApproximation for Gaussian likelihood and a linear model)
          invSlope = Math.Abs(tau_i / (paramStdError[pIdx] * zv));
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
        var a = alglib.spline1dcalc(spline_tau2p[aIdx], tauA * tauScale); // map from tau to a (using t-profile of a)
        var b = alglib.spline1dcalc(spline_p2q[aIdx, bIdx], a); // map from a to b
        var tauB = alglib.spline1dcalc(spline_p2tau[bIdx], b); // map from b to tau (using t-profile of b)
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

    public static void GetPredictionIntervals(double[,] x, NonlinearRegression nls, out double[] low, out double[] high, double alpha = 0.05, bool includeNoise = false) {
      var predRows = x.GetLength(0); // the points for which we calculate the prediction interval
      var trainRows = nls.Statistics.m;
      var n = nls.Statistics.n; // number of parameters
      var d = x.GetLength(1); // number of features

      var _low = new double[predRows];
      var _high = new double[predRows];

      // calc predicted values
      var yPred = new double[predRows];
      nls.func(nls.Statistics.paramEst, x, yPred);

      // var offsetIdx = Expr.FindOffsetParameterIndex(nls.modelExpr);
      // var scaleIdx = Expr.FindScalingParameterIndex(nls.modelExpr);
      // if (offsetIdx == -1 && scaleIdx == -1) {
      //   throw new NotSupportedException("Only models with an explicit offset or scaling parameter are supported by the t-profile prediction intervals.");
      // }

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
      // old code for pointwise and simultaneuous intervals
      // var t = alglib.invstudenttdistribution(m - n, 1 - alpha / 2);
      // var f = alglib.invfdistribution(n, m - n, alpha);
      // var s = nls.Statistics.s;
      // if (m == 1) {
      //   _low[i] = alglib.spline1dcalc(tau2theta, -t) - (includeNoise ? t * s : 0.0);
      //   _high[i] = alglib.spline1dcalc(tau2theta, t) + (includeNoise ? t * s : 0.0);
      // } else {
      //   _low[i] = alglib.spline1dcalc(tau2theta, -f) - (includeNoise ? t * s : 0.0);
      //   _high[i] = alglib.spline1dcalc(tau2theta, f) + (includeNoise ? t * s : 0.0);
      // }
      var s = nls.Statistics.s;

      // prediction intervals for each point in x
      Parallel.For(0, predRows, new ParallelOptions() { MaxDegreeOfParallelism = 1 },
        (i, loopState) => {
          // buffer
          // actually they are only needed once for the whole loop but with parallel for we need to make copies
          double[] paramEstExt = new double[n];
          Array.Copy(nls.ParamEst, paramEstExt, nls.ParamEst.Length);
          var xi = new double[d];

          Buffer.BlockCopy(x, i * d * sizeof(double), xi, 0, d * sizeof(double));
          #region old reparameterization
          // Function funcExt;
          // Jacobian jacExt;
          // int outputParamIdx;
          // if (offsetIdx > 0) {
          //   funcExt = Util.ReparameterizeFuncWithOffset(nls.func, xi, offsetIdx);
          //   jacExt = Util.ReparameterizeJacobianWithOffset(nls.jacobian, xi, offsetIdx);
          //   outputParamIdx = offsetIdx;
          // } else if (scaleIdx > 0) {
          //   funcExt = Util.ReparameterizeFuncWithScale(nls.func, xi, scaleIdx);
          //   jacExt = Util.ReparameterizeJacobianWithScale(nls.jacobian, xi, scaleIdx);
          //   outputParamIdx = scaleIdx;
          // } else {
          //   throw new NotSupportedException("Only models with an explicit offset or scaling parameter are supported by the t-profile prediction intervals.");
          // }

          // paramEstExt[outputParamIdx] = yPred[i]; // function output parameter is prediction at point xi
          #endregion 
          var reparameterizedModel = Expr.ReparameterizeExpr(nls.modelExpr, xi, out var outputParamIdx);
          paramEstExt[outputParamIdx] = yPred[i];


          // TODO compilation can be done outside of the loop if reparameterizedModel has x,x0,theta as parameters (needs new expression type)
          var _func = Expr.Broadcast(reparameterizedModel).Compile();
          void modelFunc(double[] p, double[,] X, double[] f) => _func(p, X, f); // TODO wrapper only necessary because return values are incompatible 
          var _jac = Expr.Jacobian(reparameterizedModel, n).Compile();
          void modelJac(double[] p, double[,] X, double[] f, double[,] jac) => _jac(p, X, f, jac);


          var statisticsExt = new LaplaceApproximation(trainRows, n, nls.Statistics.SSR, yPred, paramEstExt,
            Util.CreateGaussianNegLogLikelihoodHessian(modelJac, nls.y, nls.Statistics.s), nls.x);
          // var statisticsExt = new LaplaceApproximation(trainRows, n, nls.Statistics.SSR, yPred, paramEstExt,
          //   Util.CreateBernoulliNegLogLikelihoodHessian(modelJac, nls.y), nls.x); // the effort for this is small compared to the effort of the TProfile calculation below

          var profile = CalcTProfile(nls.y, nls.x, statisticsExt, modelFunc, modelJac, outputParamIdx, alpha); // only for the function output parameter

          var tau = profile.Item1;
          var theta = new double[tau.Length];
          for (int k = 0; k < theta.Length; k++) {
            theta[k] = profile.Item2[outputParamIdx][k]; // profile of function output parameter
          }
          alglib.spline1dbuildcubic(tau, theta, out var tau2theta);
          if (tau.Min() > -t) _low[i] = double.NaN;
          else _low[i] = alglib.spline1dcalc(tau2theta, -t) - (includeNoise ? t * s : 0.0);
          if (tau.Max() < t) _high[i] = double.NaN;
          else _high[i] = alglib.spline1dcalc(tau2theta, t) + (includeNoise ? t * s : 0.0);
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
  }
}

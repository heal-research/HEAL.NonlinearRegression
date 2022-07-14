using System;
using System.Collections.Generic;

namespace HEAL.NonlinearRegression {
  public static class PredictionInterval {

    // calculate t-profile for prediction output
    public static Tuple<double[],double[]> Calculate(double[] p,
      Action<double[], double[]> func,
      Action<double[], double[], double[,]> jacobian,
      double[] y, 
      double se, double s, double SSR) {

      int maxIterations = 30;
      int m = y.Length;
      int n = p.Length;

      // var tmax = Math.Sqrt(alglib.invfdistribution(m, m - n, 0.01)); // limit for t (use small alpha here)
      var tmax = alglib.invfdistribution(n, m - n, 0.01); // limit for t (use small alpha here)

      var kmax = 30;
      int step = 8;

      // alglib.minlmcreatevj(m, p, out var state);
      // alglib.minlmsetcond(state, epsx: 0.0, maxIterations);
      // if (scale != null) alglib.minlmsetscale(state, scale);
      // if (stepMax > 0.0) alglib.minlmsetstpmax(state, stepMax);



      var tau = new List<double>();
      var theta = new List<double>();
      var delta = -se / step; // standard deviation for the residuals (based on n-m DoF) 
      var p_cond = (double[])p.Clone();

      alglib.minlmcreatevj(m, p_cond, out var state);
      alglib.minlmsetcond(state, 0.0, maxits: maxIterations);

      var pred0 = p[p.Length - 1];

      do {
        var t = 0.0; // bug fix to pseudo-code in Bates and Watts
        var invSlope = 1.0;
        for (int k = 0; k < kmax; k++) {
          t = t + invSlope;
          var curP = pred0 + delta * t;

          void _f(double[] pCur, double[] fi, object o) {
            // fix last parameter (offset)
            pCur[pCur.Length - 1] = curP;
            func(pCur, fi);
            for (int i = 0; i < m; i++) {
              fi[i] = fi[i] - y[i];
            }
          }

          void _jac(double[] pCur, double[] fi, double[,] jac, object o) {
            // fix last parameter (offset)
            pCur[pCur.Length - 1] = curP;
            jacobian(pCur, fi, jac); // pass through
            for (int i = 0; i < m; i++) {
              fi[i] = fi[i] - y[i];
            }
            // fix last parameter
            for (int i = 0; i < jac.GetLength(0); i++)
              jac[i, pCur.Length - 1] = 0.0; 
          }

          // minimize
          alglib.minlmrestartfrom(state, p_cond);
          alglib.minlmoptguardgradient(state, 1e-6);
          alglib.minlmoptimize(state, _f, _jac, rep: null, obj: null);
          alglib.minlmresults(state, out p_cond, out var report);
          alglib.minlmoptguardresults(state, out var optGuardReport);
          if (optGuardReport.badgradsuspected) throw new InvalidProgramException();
          if (report.terminationtype < 0) throw new InvalidProgramException();

          // calc squared error again (could be taken from state or report)
          double[] pred_cond = new double[m];
          // double[,] jac_cond = new double[m, n];
          // jacobian(p_cond, pred_cond, jac_cond);
          func(p_cond, pred_cond);

          var SSR_cond = 0.0; // S(\theta_p)
          var zv = 0.0; // z^T v_p

          for (int i = 0; i < m; i++) {
            var z = pred_cond[i] - y[i];
            SSR_cond += z * z;
            zv += z; // * jac_cond[i, p.Length - 1];          // jac[.,pIdx] = 1
          }

          var tau_i = Math.Sign(delta) * Math.Sqrt(SSR_cond - SSR) / s;

          invSlope = Math.Abs(tau_i * s * s / (se * zv));    // Math.Abs(tau_i * s * s / (paramStdError[pIdx] * zv));
          tau.Add(tau_i);
          theta.Add(curP);

          invSlope = Math.Min(4.0, Math.Max(invSlope, 1.0 / 16));

          if (Math.Abs(tau_i) > tmax) break;
        }
        delta = -delta; // repeat for other direction
      } while (delta > 0);  // exactly two iterations


      // sort M by tau
      var tauArr = tau.ToArray();
      var thetaArr = theta.ToArray();
      Array.Sort(tauArr, thetaArr);
      // for (int i = 0; i < tauArr.Length; i++) {
      //   Console.WriteLine($"{tauArr[i]} {thetaArr[i]}");
      // }
      return Tuple.Create(tauArr, thetaArr);
    }
  }
}

using HEAL.NonlinearRegression;

namespace HEAL.NonlinearRegression {
  public class Program {
    public static void Main(string[] args) {
      DemoLinearUnivariate();
      // DemoLinear();
      // DemoExponential();
      // DemoPCB();
      // DemoBOD();
      // DemoPuromycin();
    }

    public static void DemoLinearUnivariate() {
      var rand = new System.Random(1234);

      var pOpt = new double[] { 2.5, 2 };

      int m = 20;
      var d = 2;
      var x = new double[m, d];
      var y = new double[m];
      double[] yNoise = null;

      NonlinearRegression.Report report = null;

      for (int i = 0; i < m; i++) {
        for (int j = 0; j < d - 1; j++) {
          x[i, j] = i / (double)m * 20 - 10; //  rand.NextDouble() * 20 - 10;
        }
        x[i, d - 1] = 1.0;
      }


      void F(double[] p, double[] fi) {
        for (int i = 0; i < m; i++) {
          fi[i] = 0;
          for (int j = 0; j < d; j++)
            fi[i] += p[j] * x[i, j];
        }
      }

      void Jac(double[] p, double[] fi, double[,] Jac) {
        F(p, fi);
        Array.Copy(x, Jac, x.Length);
      }

      F(pOpt, y);

      yNoise = (double[])y.Clone();
      for (int i = 0; i < m; i++) yNoise[i] += randNorm(rand, 0, 10);

      var p = new double[] { .1, .1 };
      NonlinearRegression.FitLeastSquares(p, F, Jac, yNoise, out report);

      if (report.Success) {
        Console.WriteLine($"p_opt: {string.Join(" ", p.Select(pi => pi.ToString("e5")))}");
        Console.WriteLine($"{report}");
        report.Statistics.WriteStatistics(Console.Out);

      } else {
        Console.WriteLine("There was a problem while fitting.");
      }


      // re-parameterized function F_extendend has an additional parameter which is the output in predx
      // the re-parameterized function is f(x, p) - f(x0, p) + p_ext

      // we test it here for one input point (first point in training)
      var predx = new double[] { x[0, 0], x[0, 1] };



      void F_ext(double[] p, double[] fi) {
        for (int i = 0; i < m; i++) {
          fi[i] = 0;
          for (int j = 0; j < d - 1; j++)
            fi[i] += p[j] * x[i, j] - p[j] * predx[j];
          fi[i] += p[d - 1];
        }
      }

      void Jac_ext(double[] p, double[] fi, double[,] Jac) {
        F_ext(p, fi);

        for (int i = 0; i < m; i++) {
          for (int j = 0; j < d - 1; j++)
            Jac[i, j] = x[i, j] - predx[j];

          Jac[i, d - 1] = 1;
        }
      }

      Console.WriteLine("Prediction intervals based on t-profile");
      for (int i = 0; i < m; i++) {
        var newParam = (double[])report.Statistics.paramEst.Clone();
        newParam[d - 1] = report.Statistics.yPred[i];
        predx = new double[] { x[i, 0], x[i, 1] };

        var modifiedStats = new Statistics(m, d, report.Statistics.SSR, report.Statistics.yPred, newParam, Jac_ext);

        var profile = PredictionInterval.Calculate(newParam, F_ext, Jac_ext, yNoise, modifiedStats.paramStdError.Last(), modifiedStats.s, modifiedStats.SSR);
        alglib.spline1dbuildcubic(profile.Item1, profile.Item2, out var tau2theta);
        var alpha = 0.05;
        var t = alglib.invstudenttdistribution(m - d, 1 - alpha / 2);
        Console.WriteLine($"{report.Statistics.yPred[i],14:e4} {alglib.spline1dcalc(tau2theta, -t),14:e4} {alglib.spline1dcalc(tau2theta, t),14:e4}");
      }
    }

    public static double randNorm(System.Random rand, int mean, int stdDev) {
      double u, v, s;
      do {
        u = rand.NextDouble() * 2 - 1;
        v = rand.NextDouble() * 2 - 1;
        s = u * u + v * v;
      } while (s >= 1 || s == 0);
      s = Math.Sqrt(-2.0 * Math.Log(s) / s);
      return mean + stdDev * u * s;
    }

    public static void DemoLinear() {
      var rand = new System.Random(1234);

      var pOpt = new double[] { 1, 2, 3, 4 };

      int m = 20;
      var d = 4;
      var x = new double[m, d];
      var y = new double[m];
      double[] yNoise = null;

      NonlinearRegression.Report report = null;

      for (int i = 0; i < m; i++) {
        for (int j = 0; j < d - 1; j++) {
          x[i, j] = rand.NextDouble() * 2 - 1; // u~(-1,1)
        }
        x[i, d - 1] = 1.0;
      }


      void F(double[] p, double[] fi) {
        for (int i = 0; i < m; i++) {
          fi[i] = 0;
          for (int j = 0; j < d; j++)
            fi[i] += p[j] * x[i, j];
        }
      }

      void Jac(double[] p, double[] fi, double[,] Jac) {
        F(p, fi);
        Array.Copy(x, Jac, x.Length);
      }

      F(pOpt, y);

      yNoise = (double[])y.Clone();
      for (int i = 0; i < m; i++) yNoise[i] += rand.NextDouble() * 0.2 - 0.1;

      var p = new double[] { .1, .1, .1, .1 };
      NonlinearRegression.FitLeastSquares(p, F, Jac, yNoise, out report);

      if (report.Success) {
        Console.WriteLine($"p_opt: {string.Join(" ", p.Select(pi => pi.ToString("e5")))}");
        Console.WriteLine($"{report}");
        report.Statistics.WriteStatistics(Console.Out);

      } else {
        Console.WriteLine("There was a problem while fitting.");
      }


      // re-parameterized function F_extendend has an additional parameter which is the output in predx
      // the re-parameterized function is f(x, p) - f(x0, p) + p_ext

      // we test it here for one input point (first point in training)
      var predx = new double[] { x[0, 0], x[0, 1], x[0, 2], x[0, 3] };



      void F_ext(double[] p, double[] fi) {
        for (int i = 0; i < m; i++) {
          fi[i] = 0;
          for (int j = 0; j < d - 1; j++)
            fi[i] += p[j] * x[i, j] - p[j] * predx[j];
          fi[i] += p[d - 1];
        }
      }

      void Jac_ext(double[] p, double[] fi, double[,] Jac) {
        F_ext(p, fi);

        for (int i = 0; i < m; i++) {
          for (int j = 0; j < d - 1; j++)
            Jac[i, j] = x[i, j] - predx[j];

          Jac[i, d - 1] = 1;
        }
      }



      var newParam = (double[])report.Statistics.paramEst.Clone();
      newParam[d - 1] = report.Statistics.yPred[0];


      var modifiedStats = new Statistics(m, d, report.Statistics.SSR, report.Statistics.yPred, newParam, Jac_ext);


      var profile = PredictionInterval.Calculate(newParam, F_ext, Jac_ext, yNoise, modifiedStats.paramStdError.Last(), modifiedStats.s, modifiedStats.SSR);
      alglib.spline1dbuildcubic(profile.Item1, profile.Item2, out var tau2theta);
      var alpha = 0.05;
      var t = alglib.invstudenttdistribution(m - d, 1 - alpha / 2);
      alglib.spline1dcalc(tau2theta, t);
      // TODO CONTINUE HERE
    }

    void DemoExponential() {
      var pOpt = new double[] { 0.2, -3.0 };

      int m = 20;
      var x = new double[m, 1];
      var y = new double[m];


      void F(double[] p, double[] fi) {
        for (int i = 0; i < m; i++) {
          fi[i] = p[0] * Math.Exp(x[i, 0] * p[1]);
        }
      }

      void Jac(double[] p, double[] fi, double[,] Jac) {
        F(p, fi);
        for (int i = 0; i < m; i++) {
          Jac[i, 0] = Math.Exp(x[i, 0] * p[1]);
          Jac[i, 1] = x[i, 0] * fi[i];
        }
      }

      // generate data 
      for (int i = 0; i < m; i++) {
        x[i, 0] = i / (double)m;
      }
      F(pOpt, y);

      // fit with starting point [1, 1]
      var p = new double[] { 1.0, 1.0 };
      NonlinearRegression.FitLeastSquares(p, F, Jac, y, out var report);

      if (report.Success) {
        Console.WriteLine($"p_opt: {string.Join(" ", p.Select(pi => pi.ToString("e5")))}");
        Console.WriteLine($"{report}");
        report.Statistics.WriteStatistics(Console.Out);

      } else {
        Console.WriteLine("There was a problem while fitting.");
      }
    }


    void DemoPCB() {
      // PCB example from Nonlinear Regression Analysis and Its Applications, Bates and Watts, 1988
      Console.WriteLine("-----------");
      Console.WriteLine("PCB example");
      Console.WriteLine("-----------");

      var age = new double[] {
  1,
  1,
  1,
  1,
  2,
  2,
  2,
  3,
  3,
  3,
  4,
  4,
  4,
  5,
  6,
  6,
  6,
  7,
  7,
  7,
  8,
  8,
  8,
  9,
  11,
  12,
  12,
  12
};

      var PCB = new double[] {
  0.6,
  1.6,
  0.5,
  1.2,
  2.0,
  1.3,
  2.5,
  2.2,
  2.4,
  1.2,
  3.5,
  4.1,
  5.1,
  5.7,
  3.4,
  9.7,
  8.6,
  4.0,
  5.5,
  10.5,
  17.5,
  13.4,
  4.5,
  30.4,
  12.4,
  13.4,
  26.2,
  7.4
};

      var m = PCB.Length;

      // model: y = b1 + b2 x
      var y = new double[m];
      var x = new double[m, 2];
      for (int i = 0; i < m; i++) {
        y[i] = Math.Log(PCB[i]);      // ln(PCB)
        x[i, 0] = 1.0;
        x[i, 1] = Math.Cbrt(age[i]);    // cbrt(age)
      }

      void F(double[] p, double[] fi) {
        for (int i = 0; i < m; i++) {
          fi[i] = p[0] * x[i, 0] + p[1] * x[i, 1];
        }
      }

      void Jac(double[] p, double[] fi, double[,] Jac) {
        F(p, fi);
        for (int i = 0; i < m; i++) {
          Jac[i, 0] = x[i, 0];
          Jac[i, 1] = x[i, 1];
        }
      }

      // fit with starting point [1, 1]
      var p = new double[] { 1.0, 1.0 };
      NonlinearRegression.FitLeastSquares(p, F, Jac, y, out var report);

      if (report.Success) {
        Console.WriteLine($"p_opt: {string.Join(" ", p.Select(pi => pi.ToString("e5")))}");
        Console.WriteLine($"{report}");
        report.Statistics.WriteStatistics(Console.Out);
        report.Statistics.ApproximateProfilePairContour(0, 1, alpha: 0.05, out _, out _, out var p1, out var p2);
        Console.WriteLine("Approximate profile pair contour (p0 vs p1)");
        for (int i = 0; i < p1.Length; i++) {
          Console.WriteLine($"{p1[i]} {p2[i]}");
        }

      } else {
        Console.WriteLine("There was a problem while fitting.");
      }
    }

    void DemoBOD() {
      // BOD example from Nonlinear Regression Analysis and Its Applications, Bates and Watts, 1988
      Console.WriteLine("-----------");
      Console.WriteLine("BOD example");
      Console.WriteLine("-----------");

      // A 1.4     (there are two BOD datasets)
      var days = new int[] {
    1, 2, 3, 4, 5, 7
  };

      var BOD = new double[] {
    8.3,
    10.3,
    19.0,
    16.0,
    15.6,
    19.8
  };

      var m = BOD.Length;
      double[] pOpt = null; // for prediction interval
      {
        // model: BOD = p1 * (1 - exp(-p2 * days))

        void F(double[] p, double[] fi) {
          for (int i = 0; i < m; i++) {
            fi[i] = p[0] * (1 - Math.Exp(-p[1] * days[i]));
          }
        }

        void Jac(double[] p, double[] fi, double[,] Jac) {
          F(p, fi);
          for (int i = 0; i < m; i++) {
            Jac[i, 0] = 1 - Math.Exp(-p[1] * days[i]);
            Jac[i, 1] = p[0] * days[i] * Math.Exp(-p[1] * days[i]);
          }
        }

        var p = new double[] { 20, 0.24 }; // Bates and Watts, page 41
                                           // expected results:
                                           // p* = (19.143, 0.5311), s² = 6.498, 
                                           // cor(p1, p2) = -0.85
                                           // linear approximation 95% interval p1 = [12.2, 26.1], p2 = [-0.033, 1.095]
                                           // t-profile 95% interval p1 = [14.05, 37.77], p2 = [0.132, 177]
        NonlinearRegression.FitLeastSquares(p, F, Jac, BOD, out var report);

        if (report.Success) {
          pOpt = p;
          Console.WriteLine($"p_opt: {string.Join(" ", p.Select(pi => pi.ToString("e5")))}");
          Console.WriteLine($"{report}");
          report.Statistics.WriteStatistics(Console.Out);
          report.Statistics.ApproximateProfilePairContour(0, 1, alpha: 0.10, out _, out _, out var p1, out var p2);
          Console.WriteLine("Approximate profile pair contour (p0 vs p1)");
          for (int i = 0; i < p1.Length; i++) {
            Console.WriteLine($"{p1[i]} {p2[i]}");
          }
        } else {
          Console.WriteLine("There was a problem while fitting.");
        }
      }
    }

    void DemoPuromycin() {
      // Puromycin example from Nonlinear Regression Analysis and Its Applications, Bates and Watts, 1988
      Console.WriteLine("-----------------");
      Console.WriteLine("Puromycin example");
      Console.WriteLine("-----------------");
      // substrate concentration
      var x = new double[] {
    0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56, 1.10, 1.10
  };

      var treated = new double[] {
    76, 47, 97, 107, 123, 139, 159, 152, 191, 201, 207, 200
  };

      var m = x.Length;

      // model: y = p1 x / (p2 + x)

      void F(double[] p, double[] fi) {
        for (int i = 0; i < m; i++) {
          fi[i] = p[0] * x[i] / (p[1] + x[i]);
        }
      }

      void Jac(double[] p, double[] fi, double[,] Jac) {
        F(p, fi);
        for (int i = 0; i < m; i++) {
          Jac[i, 0] = x[i] / (p[1] + x[i]);
          Jac[i, 1] = -p[0] * x[i] / Math.Pow(p[1] + x[i], 2);
        }
      }

      var p = new double[] { 205, 0.08 };  // Bates and Watts page 41
      NonlinearRegression.FitLeastSquares(p, F, Jac, treated, out var report);

      if (report.Success) {
        Console.WriteLine($"p_opt: {string.Join(" ", p.Select(pi => pi.ToString("e5")))}");
        Console.WriteLine($"{report}");
        report.Statistics.WriteStatistics(Console.Out);
        report.Statistics.ApproximateProfilePairContour(0, 1, alpha: 0.05, out _, out _, out var p1, out var p2);
        Console.WriteLine("Approximate profile pair contour (p0 vs p1)");
        for (int i = 0; i < p1.Length; i++) {
          Console.WriteLine($"{p1[i]} {p2[i]}");
        }

      } else {
        Console.WriteLine("There was a problem while fitting.");
      }
    }

  }
}
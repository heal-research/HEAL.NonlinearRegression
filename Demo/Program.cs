using System;
using System.Linq;

namespace HEAL.NonlinearRegression {
  public class Program {
    public static void Main(string[] args) {
      RunDemo(new LinearUnivariateProblem());
      RunDemo(new LinearProblem());
      RunDemo(new ExponentialProblem());
      RunDemo(new PCBProblem());
      RunDemo(new BODProblem());       // Bates and Watts, page 41
                                       // expected results:
                                       // p* = (19.143, 0.5311), s² = 6.498, 
                                       // cor(p1, p2) = -0.85
                                       // linear approximation 95% interval p1 = [12.2, 26.1], p2 = [-0.033, 1.095]
                                       // t-profile 95% interval p1 = [14.05, 37.77], p2 = [0.132, 177]
      RunDemo(new PuromycinProblem());
    }

    private static void RunDemo(INLSProblem problem) {
      Console.WriteLine("-----------------");
      Console.WriteLine($"{problem.GetType().Name}");
      Console.WriteLine("-----------------");

      RunDemo(problem.X, problem.y, problem.Func, problem.Jacobian, problem.ThetaStart);

      Console.WriteLine();
    }

    public static void DemoLinearUnivariate() {


      /*

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
      */
    }


    public static void DemoLinear() {

      /*
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
      */
    }



    /// <summary>
    /// Runs the algorithm and analysis for a nonlinear regression problem.
    /// </summary>
    /// <param name="x">The matrix of input values for f.</param>
    /// <param name="y">The vector of target values.</param>
    /// <param name="f">The function to fit.</param>
    /// <param name="jac">The Jacobian of f.</param>
    /// <param name="start">The starting point for parameter values.</param>
    private static void RunDemo(double[,] x, double[] y, Function f, Jacobian jac, double[] start) {
      NonlinearRegression.FitLeastSquares(start, f, jac, x, y, out var report);

      if (report.Success) {
        Console.WriteLine($"p_opt: {string.Join(" ", start.Select(pi => pi.ToString("e5")))}");
        Console.WriteLine($"{report}");
        report.Statistics.WriteStatistics(Console.Out);



        // TODO: extend this to produce some relevant output for all parameters instead of only a pairwise contour
        if (report.Statistics.s > 1e-6) {
          report.Statistics.ApproximateProfilePairContour(0, 1, alpha: 0.05, out _, out _, out var p1, out var p2);
          Console.WriteLine("Approximate profile pair contour (p0 vs p1)");
          for (int i = 0; i < p1.Length; i++) {
            Console.WriteLine($"{p1[i]} {p2[i]}");
          }
        }

      } else {
        Console.WriteLine("There was a problem while fitting.");
      }
    }

    #region helper


    #endregion
  }
}
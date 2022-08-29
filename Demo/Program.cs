using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using HEAL.Expressions;

namespace HEAL.NonlinearRegression {
  public class Program {
    public static void Main(string[] args) {
      RunDemo(new FriedmanProblem());
      RunDemo(new PagieProblem());
      RunDemo(new KotanchekProblem());
      RunDemo(new RatPol2DProblem());
      RunDemo(new PuromycinDSRProblem());
      RunDemo(new PCBDSRProblem());


      RunDemo(new LinearUnivariateProblem());
      RunDemo(new LinearProblem());
      RunDemo(new ExponentialProblem());
      RunDemo(new PCBProblem());
      RunDemo(new BODProblem()); // Bates and Watts, page 41
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

      // fitting and prediction intervals
      RunDemo(problem.X, problem.y, problem.Func, problem.Jacobian, problem.ThetaStart);

      // model analysis
      if (problem is SymbolicProblemBase symbProb) {
        Console.WriteLine($"Model: {symbProb.ModelExpr}");

        Console.WriteLine("Variable importance (SSR ratio)");
        var varImportance =
          ModelAnalysis.VariableImportance(symbProb.ModelExpr, symbProb.X, symbProb.y, symbProb.ThetaStart);
        foreach (var kvp in varImportance.OrderByDescending(kvp => kvp.Value)) {
          Console.WriteLine($"x{kvp.Key} {kvp.Value,-11:e4}");
        }
        Console.WriteLine();

        Console.WriteLine("Subtree importance (SSR ratio)");
        var expr = symbProb.ModelExpr;
        var subExprImportance = ModelAnalysis.SubtreeImportance(expr, symbProb.X, symbProb.y, symbProb.ThetaStart);
        var sat = new Dictionary<Expression, double>();
        sat[expr] = 0.0; // reference value for the importance
        foreach (var tup in subExprImportance.OrderByDescending(tup => tup.Item1)) {
          Console.WriteLine($"{tup.Item1} {tup.Item2,-11:e4}");
          sat[tup.Item1] = Math.Max(0, Math.Log(tup.Item2)); // use log scale for coloring
        }

        using (var writer = new System.IO.StreamWriter($"{problem.GetType().Name}.gv")) {
          writer.WriteLine(Expr.ToGraphViz(expr, saturation: sat));
        }
        Console.WriteLine();

        Console.WriteLine("Nested models (set parameters zero) (SSR ratio)");
        ModelAnalysis.NestedModelLiklihoodRatios(symbProb.ModelExpr, symbProb.X, symbProb.y, symbProb.ThetaStart);
      }


      Console.WriteLine();
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
      var theta = (double[])start.Clone();
      var nls = new NonlinearRegression();
      nls.Fit(theta, f, jac, x, y);

      if (nls.OptReport.Success) {
        Console.WriteLine($"p_opt: {string.Join(" ", theta.Select(pi => pi.ToString("e5")))}");
        Console.WriteLine($"{nls.OptReport}");
        nls.WriteStatistics();


        if (nls.Statistics.s > 1e-6) {
          var tProfile = new TProfile(y, x, nls.Statistics, f, jac);


          nls.Statistics.GetPredictionIntervals(jac, x, 0.05, out var resStdError, out var linLow, out var linHigh);
          Console.WriteLine("Prediction intervals (linear approximation);");
          for (int i = 0; i < Math.Min(linLow.Length, 10); i++) {
            Console.WriteLine($"{nls.Statistics.yPred[i],14:e4} {linLow[i],14:e4} {linHigh[i],14:e4}");
          }

          Console.WriteLine();

          // produce predictions for the first few data points
          double[,] xReduced;
          if (x.GetLength(0) > 10) {
            xReduced = new double[10, x.GetLength(1)];
            Buffer.BlockCopy(x, 0, xReduced, 0, xReduced.Length * sizeof(double));
          } else {
            xReduced = x;
          }

          try {

            TProfile.GetPredictionIntervals(xReduced, nls, out var tLow, out var tHigh);
            Console.WriteLine("Prediction intervals (t-profile);");
            for (int i = 0; i < Math.Min(linLow.Length, 10); i++) {
              Console.WriteLine($"{nls.Statistics.yPred[i],14:e4} {tLow[i],14:e4} {tHigh[i],14:e4}");
            }
          }
          catch (Exception e) {
            Console.WriteLine(e.Message);
          }
        }

        // // TODO: extend this to produce some relevant output for all parameters instead of only a pairwise contour
        // if (report.Statistics.s > 1e-6) {
        //   report.Statistics.ApproximateProfilePairContour(0, 1, alpha: 0.05, out _, out _, out var p1, out var p2);
        //   Console.WriteLine("Approximate profile pair contour (p0 vs p1)");
        //   for (int i = 0; i < p1.Length; i++) {
        //     Console.WriteLine($"{p1[i]} {p2[i]}");
        //   }
        // }
      } else {
        Console.WriteLine("There was a problem while fitting.");
      }
    }

    #region helper

    #endregion
  }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using HEAL.Expressions;

namespace HEAL.NonlinearRegression.Demo {
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
      RunDemo(problem.X, problem.y, problem.ModelExpression, problem.ThetaStart);

      // model analysis
      if (problem is SymbolicProblemBase symbProb) {
        Console.WriteLine($"Model: {symbProb.ModelExpression}");

        Console.WriteLine("Variable importance (SSR ratio)");
        var varImportance =
          ModelAnalysis.VariableImportance(symbProb.ModelExpression, LikelihoodEnum.Gaussian, noiseSigma: null, symbProb.X, symbProb.y, symbProb.ThetaStart);
        foreach (var kvp in varImportance.OrderByDescending(kvp => kvp.Value)) {
          Console.WriteLine($"x{kvp.Key} {kvp.Value,-11:e4}");
        }
        Console.WriteLine();

        Console.WriteLine("Subtree importance (SSR ratio)");
        var expr = symbProb.ModelExpression;
        var subExprImportance = ModelAnalysis.SubtreeImportance(expr, LikelihoodEnum.Gaussian, noiseSigma: null, symbProb.X, symbProb.y, symbProb.ThetaStart);
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
        ModelAnalysis.NestedModelLiklihoodRatios(symbProb.ModelExpression, LikelihoodEnum.Gaussian, noiseSigma: null, symbProb.X, symbProb.y, symbProb.ThetaStart, maxIterations: 10000);
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
    private static void RunDemo(double[,] x, double[] y, Expression<Expr.ParametricFunction> expr, double[] start) {
      var theta = (double[])start.Clone();

      var nls = new NonlinearRegression();
      nls.Fit(theta, expr, LikelihoodEnum.Gaussian, x, y);

      if (nls.OptReport.Success) {
        Console.WriteLine($"p_opt: {string.Join(" ", theta.Select(pi => pi.ToString("e5")))}");
        Console.WriteLine($"{nls.OptReport}");
        nls.WriteStatistics();


        var tProfile = new TProfile(nls.Statistics, nls.Likelihood);


        var pred = nls.PredictWithIntervals(x, IntervalEnum.LaplaceApproximation, 0.05);
        Console.WriteLine("Prediction intervals (linear approximation);");
        for (int i = 0; i < Math.Min(pred.GetLength(0), 10); i++) {
          Console.WriteLine($"{pred[i,0],14:e4} {pred[i,2],14:e4} {pred[i,3],14:e4}");
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

          pred = nls.PredictWithIntervals(x, IntervalEnum.TProfile, 0.05);
          Console.WriteLine("Prediction intervals (t-profile);");
          for (int i = 0; i < Math.Min(pred.GetLength(0), 10); i++) {
            Console.WriteLine($"{pred[i, 0],14:e4} {pred[i, 1],14:e4} {pred[i, 2],14:e4}");
          }
        } catch (Exception e) {
          Console.WriteLine(e.Message);
        }
      } else {
        Console.WriteLine("There was a problem while fitting.");
      }
    }

    #region helper

    #endregion
  }
}

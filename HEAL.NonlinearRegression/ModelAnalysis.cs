using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Linq.Expressions;
using System.Threading.Tasks;
using HEAL.Expressions;

namespace HEAL.NonlinearRegression {
  // TODO:
  // - Impact calculation for variables, parameters, and sub-trees should be unified (it is always a manipulation of the expression and refitting afterwards)
  // - Pruning should search over all possible simplifying manipulations of trees (removal of sub-trees, removal of parameters, removal of non-linear functions...)
  public static class ModelAnalysis {
    /// <summary>
    /// Replaces all references to a variable in the model by a parameter. Re-fits the model and returns the increase in R²
    /// </summary>
    /// <param name="expr">The full model expression</param>
    /// <param name="X">Input values</param>
    /// <param name="y">Target values</param>
    /// <param name="p">Initial parameter values for the full model</param>
    /// <returns></returns>
    public static Dictionary<int, double> VariableImportance(LikelihoodBase likelihood, double[] p) {
      var nlr = new NonlinearRegression();
      nlr.Fit(p, likelihood);
      var pOpt = p;
      var origExpr = likelihood.ModelExpr;
      var stats0 = nlr.LaplaceApproximation;
      var referenceDeviance = nlr.Deviance;
      var m = likelihood.X.GetLength(0);
      var d = likelihood.X.GetLength(1);

      var mean = new double[d];
      for (int i = 0; i < d; i++) {
        mean[i] = Enumerable.Range(0, m).Select(r => likelihood.X[r, i]).Average();
      }

      var varExpl = new Dictionary<int, double>();
      for (int varIdx = 0; varIdx < d; varIdx++) {
        var newExpr = Expr.ReplaceVariableWithParameter(origExpr, (double[])pOpt.Clone(),
          varIdx, mean[varIdx], out var newThetaValues);
        newExpr = Expr.FoldParameters(newExpr, newThetaValues, out newThetaValues);
        likelihood.ModelExpr = newExpr;
        nlr = new NonlinearRegression();
        nlr.Fit(newThetaValues, likelihood);
        if (nlr.LaplaceApproximation == null) {
          Console.WriteLine("Problem while fitting");
          varExpl[varIdx] = 0.0;
        } else {
          var newStats = nlr.LaplaceApproximation;
          // increase in variance for the reduced feature = variance explained by the feature
          varExpl[varIdx] = (nlr.Deviance - referenceDeviance) / likelihood.Y.Length;
        }
      }

      return varExpl;
    }


    /// <summary>
    /// Replaces each sub-tree in the model by a parameter and re-fits the model. The factor SSR_reduced / SSR_full is returned as impact.
    /// The sub-expressions depend on the structure of the model. I.e. a + b + c + d might be represented as
    /// ((a + b) + c) + d or ((a+b) + (c+d)) or (a + (b + (c + d))) as expression can have only binary operators. 
    /// </summary>
    /// <param name="expr">The full model</param>
    /// <param name="X">Input values.</param>
    /// <param name="y">Target variable values</param>
    /// <param name="p">Initial parameters for the full model</param>
    /// <returns></returns>
    public static IEnumerable<Tuple<Expression, double, double, double>> SubtreeImportance(LikelihoodBase likelihood, double[] p) {
      var expr = likelihood.ModelExpr;
      var pParam = expr.Parameters[0];
      var xParam = expr.Parameters[1];
      var subexpressions = FlattenExpressionVisitor.Execute(expr.Body);

      // fit the full model once for the baseline
      var nlr = new NonlinearRegression();
      nlr.Fit(p, likelihood);
      var referenceDeviance = nlr.Deviance;
      var stats0 = nlr.LaplaceApproximation;
      p = (double[])p.Clone();

      var fullAICc = nlr.AICc;
      var fullBIC = nlr.BIC;

      var impacts = new Dictionary<Expression, double>();

      foreach (var subExpr in subexpressions) {
        // skip parameters
        if (subExpr is BinaryExpression binExpr && binExpr.NodeType == ExpressionType.ArrayIndex && binExpr.Left == pParam)
          continue;
        var subExprInterpreter = new ExpressionInterpreter(Expression.Lambda<Expr.ParametricFunction>(subExpr, pParam, xParam), likelihood.XCol, likelihood.Y.Length);
        var eval = subExprInterpreter.Evaluate(p);

        var replValue = eval.Average();
        var reducedExpression = ReplaceSubexpressionWithParameterVisitor.Execute(expr, subExpr, p, replValue, out var newTheta);
        reducedExpression = Expr.FoldParameters(reducedExpression, newTheta, out newTheta);


        // fit reduced model
        likelihood.ModelExpr = reducedExpression;
        nlr.Fit(newTheta, likelihood);
        if (nlr.ParamEst != null) {
          var impact = nlr.Deviance / referenceDeviance;

          yield return Tuple.Create(subExpr, impact, nlr.AICc - fullAICc, nlr.BIC - fullBIC);
        }
      }
    }

    public static IEnumerable<(Expression<Expr.ParametricFunction> expr, double[] theta)> NestedModels(Expression<Expr.ParametricFunction> expr, double[] theta, ApproximateLikelihood laplaceApproximation, double alpha = 0.01) {
      var n = theta.Length;
      laplaceApproximation.GetParameterIntervals(theta, alpha, out var low, out var high);
      laplaceApproximation.CalcParameterStatistics(theta, out var se, out _, out _);
      var zScore = new double[n];
      for (int i = 0; i < n; i++) {
        zScore[i] = Math.Abs(theta[i] / se[i]);
      }

      var idx = Enumerable.Range(0, n).ToArray();
      Array.Sort(zScore, idx); // order paramIdx by zScore (smallest first)

      foreach (var i in idx) {
        // replace each parameter with zero
        if (low[i] <= 0.0 && high[i] >= 0.0) {
          var newExpr = ReplaceParameterWithConstantVisitor.Execute(expr, expr.Parameters[0], i, 0.0);
          newExpr = Expr.FoldParameters(newExpr, theta, out var newTheta);
          yield return (newExpr, newTheta);
        }

        // replace each parameter with -1
        if (theta[i] < 0 && low[i] <= -1.0 && high[i] >= -1.0) {
          var newExpr = ReplaceParameterWithConstantVisitor.Execute(expr, expr.Parameters[0], i, -1);
          newExpr = Expr.FoldParameters(newExpr, theta, out var newTheta);
          yield return (newExpr, newTheta);
        }
        // replace each parameter with 1
        if (theta[i] > 0 && low[i] <= 1.0 && high[i] >= 1.0) {
          var newExpr = ReplaceParameterWithConstantVisitor.Execute(expr, expr.Parameters[0], i, 1);
          newExpr = Expr.FoldParameters(newExpr, theta, out var newTheta);
          yield return (newExpr, newTheta);
        }

        // replace each parameter with the closest integer
        var intVal = Math.Round(theta[i]);


        if (low[i] < intVal && high[i] > intVal && intVal != 0.0 && Math.Abs(intVal) != 1.0) {
          // compare DL for constant and parameter to approximate improvement by replacing the parameter by a constant
          // var constDL = Math.Log(Math.Abs(intVal)) + Math.Log(2);
          // var paramDL = 0.5 * (-Math.Log(3) + Math.Log(laplaceApproximation.diagH[i])) + Math.Log(Math.Abs(theta[i]));
          // if (constDL < paramDL) {
          var newExpr = ReplaceParameterWithConstantVisitor.Execute(expr, expr.Parameters[0], i, Math.Round(theta[i]));
          newExpr = Expr.FoldParameters(newExpr, theta, out var newTheta);
          yield return (newExpr, newTheta);
          // }
        }
      }
    }

    /// <summary>
    /// Replaces each parameter in the model by a zero and re-fits the model for a comparison of nested models.
    /// We use a likelihood ratio test for model comparison which is exact for linear models.
    /// For nonlinear models the linear approximation is ok as long as the reduced model has similar fit as the full model.
    /// See "Bates and Watts, Nonlinear regression and its applications section on Comparing Models - Nested Models"
    /// for the argumentation. 
    /// </summary>
    /// <param name="expr">The full model</param>
    /// <param name="X">Input values.</param>
    /// <param name="y">Target variable values</param>
    /// <param name="p">Initial parameters for the full model</param>
    /// <returns></returns>
    public static IEnumerable<Tuple<int, double, double, Expression<Expr.ParametricFunction>, double[]>> NestedModelLiklihoodRatios(LikelihoodBase likelihood,
      double[] p, int maxIterations, bool verbose = false) {
      var expr = likelihood.ModelExpr;
      var pParam = expr.Parameters[0];
      var xParam = expr.Parameters[1];

      // fit the full model once for the baseline
      // TODO: we could skip this and get the baseline as parameter

      var nlr = new NonlinearRegression();
      nlr.Fit(p, likelihood, maxIterations: maxIterations);
      var refDeviance = nlr.Deviance;

      var stats0 = nlr.LaplaceApproximation;
      if (stats0 == null) return Enumerable.Empty<Tuple<int, double, double, Expression<Expr.ParametricFunction>, double[]>>(); // cannot fit the expression

      var n = p.Length; // number of parameters
      var m = likelihood.Y.Length; // number of observations
      var impacts = new List<Tuple<int, double, double, Expression<Expr.ParametricFunction>, double[]>>();

      var fullAIC = nlr.AICc;
      var fullBIC = nlr.BIC;
      if (verbose) {
        Console.WriteLine(expr.Body);
        Console.WriteLine($"theta: {string.Join(", ", p.Select(pi => pi.ToString("e2")))}");
        Console.WriteLine($"Full model deviance: {refDeviance,-11:e4}, mean deviance: {refDeviance / m,-11:e4}, AICc: {fullAIC,-11:f1} BIC: {fullBIC,-11:f1}");
        Console.WriteLine($"p{"idx",-5} {"val",-11} {"SSR_factor",-11} {"deltaDoF",-6} {"deltaSSR",-11} {"s2Extra":e3} {"fRatio",-11} {"p value",10} {"deltaAICc"} {"deltaBIC"}");
      }


      // for(int paramIdx = 0; paramIdx<p.Length;paramIdx++) { 
      Parallel.For(0, p.Length, (paramIdx) => {
        var v = new ReplaceParameterWithConstantVisitor(pParam, paramIdx, 0.0);
        var reducedExpression = (Expression<Expr.ParametricFunction>)v.Visit(expr);
        //Console.WriteLine($"Reduced: {reducedExpression}");
        // initial values for the reduced expression are the optimal parameters from the full model
        reducedExpression = Expr.SimplifyAndRemoveParameters(reducedExpression, p, out var newP);

        //Console.WriteLine($"Simplified: {reducedExpression}");
        var newSimplifiedStr = reducedExpression.ToString();
        var exprSet = new HashSet<string>();
        // simplify until no change (TODO: this shouldn't be necessary if visitors are implemented carefully)
        do {
          exprSet.Add(newSimplifiedStr);
          reducedExpression = Expr.FoldParameters(reducedExpression, newP, out newP);
          newSimplifiedStr = reducedExpression.ToString();
        } while (!exprSet.Contains(newSimplifiedStr));


        // fit reduced model
        try {
          var localNlr = new NonlinearRegression();
          var reducedLikelihood = likelihood.Clone();
          reducedLikelihood.ModelExpr = reducedExpression;
          localNlr.Fit(newP, reducedLikelihood, maxIterations: maxIterations);
          var reducedStats = localNlr.LaplaceApproximation;

          var devianceFactor = localNlr.Deviance / refDeviance;

          /* TODO: implement this based on "In All Likelihood" Section 6.6
          if (verbose) {
            var reducedN = newP.Length; // number of parameters
                                        // likelihood ratio test
            var fullDoF = m - n;
            var deltaDoF = n - reducedN; // number of fewer parameters
            var deltaDeviance = nlr.Deviance - refDeviance;
            var devianceExtra = deltaDeviance / deltaDoF; // increase in deviance per parameter
            var fRatio = devianceExtra / (localNlr.Dispersion * localNlr.Dispersion);

            // "accept the partial value if the calculated ratio is lower than the table value"
            var f = alglib.fdistribution(deltaDoF, fullDoF, fRatio);

            Console.WriteLine($"p{paramIdx,-5} {p[paramIdx],-11:e2} {devianceFactor,-11:e3} {deltaDoF,-6} {deltaDeviance,-11:e3} {devianceExtra,-11:e3} {fRatio,-11:e4}, {1 - f,-10:e3}, " +
              $"{localNlr.AICc - fullAIC,-11:f1} " +
              $"{localNlr.BIC - fullBIC,-11:f1}");
          }
          */
          lock (impacts) {
            impacts.Add(Tuple.Create(paramIdx, devianceFactor, localNlr.AICc - fullAIC, reducedExpression, newP));
          }
        } catch (Exception e) {
          // Console.WriteLine($"Exception {e.Message} for {reducedExpression}");
        }
      }
      );

      return impacts; // TODO improve interface
    }
  }
}

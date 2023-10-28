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
    /// <returns></returns>
    public static Dictionary<int, double> VariableImportance(LikelihoodBase likelihood, double[] p) {
      var nlr = new NonlinearRegression();
      nlr.Fit(p, likelihood);
      var pOpt = p;
      var origExpr = likelihood.ModelExpr;
      var refSSR = Util.SSR(likelihood.Y, nlr.Predict(likelihood.X));
      var m = likelihood.X.GetLength(0);
      var d = likelihood.X.GetLength(1);

      var mean = new double[d];
      for (int i = 0; i < d; i++) {
        mean[i] = Enumerable.Range(0, m).Select(r => likelihood.X[r, i]).Average(); // mean of each variable (as initial value for the parameters)
      }

      var ssrRatio = new Dictionary<int, double>();
      for (int varIdx = 0; varIdx < d; varIdx++) {
        var newExpr = Expr.ReplaceVariableWithParameter(origExpr, (double[])pOpt.Clone(), varIdx, mean[varIdx], out var newThetaValues);
        newExpr = Expr.SimplifyRepeated(newExpr, newThetaValues, out newThetaValues);
        var localLikelihood = likelihood.Clone();
        localLikelihood.ModelExpr = newExpr;
        nlr = new NonlinearRegression();
        nlr.Fit(newThetaValues, localLikelihood);
        if (nlr.LaplaceApproximation == null) {
          Console.WriteLine("Problem while fitting");
          ssrRatio[varIdx] = 1.0;
        }
        else {
          // only for Gaussian (TODO: support other likelihoods)
          var reducedSSR = Util.SSR(localLikelihood.Y, nlr.Predict(localLikelihood.X));

          ssrRatio[varIdx] = reducedSSR / refSSR;
        }
      }

      return ssrRatio;
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
      nlr.WriteStatistics();
      var refSSR = Util.SSR(likelihood.Y, nlr.Predict(likelihood.X));
      var m = likelihood.NumberOfObservations;
      var n = p.Length;
      var fullDoF = m - n;
      var s2Full = refSSR / fullDoF;


      var referenceDeviance = nlr.Deviance;
      var stats0 = nlr.LaplaceApproximation;
      p = (double[])p.Clone();

      var fullAICc = nlr.AICc;
      var fullBIC = nlr.BIC;

      var impacts = new Dictionary<Expression, double>();
      var eval = new double[likelihood.NumberOfObservations];

      Console.WriteLine($"{"SSR_factor",-11} {"deltaDoF",-6} {"deltaSSR",-11} {"s2Extra",-11} {"fRatio",-11} {"p value",10} {"deltaAICc",-11} {"deltaBIC",-11} {"MSE",-11} sub-expression");


      foreach (var subExpr in subexpressions) {
        if (subExpr is ConstantExpression) continue;
        Expression<Expr.ParametricFunction> reducedExpression = null;
        double[] newTheta = null;
        if (subExpr is BinaryExpression binExpr && binExpr.NodeType == ExpressionType.ArrayIndex && binExpr.Left == pParam) {
          // replace parameters with constant zero (see nested models)
          var arrIdx = (int)((ConstantExpression)binExpr.Right).Value;
          reducedExpression = ReplaceParameterWithConstantVisitor.Execute(expr, expr.Parameters[0], arrIdx, 0);
          reducedExpression = CollectParametersVisitor.Execute(reducedExpression, p, out newTheta);
        }
        else {
          // replace expressions with average value
          var subExprInterpreter = new ExpressionInterpreter(Expression.Lambda<Expr.ParametricFunction>(subExpr, pParam, xParam), likelihood.XCol, likelihood.Y.Length);
          subExprInterpreter.Evaluate(p, eval);

          var replValue = eval.Average();
          reducedExpression = ReplaceSubexpressionWithParameterVisitor.Execute(expr, subExpr, p, replValue, out newTheta);
        }
        reducedExpression = Expr.SimplifyRepeated(reducedExpression, newTheta, out newTheta);

        // fit reduced model
        var localNlr = new NonlinearRegression();
        var localLikelihood = likelihood.Clone();
        localLikelihood.ModelExpr = reducedExpression;
        try {
          localNlr.Fit(newTheta, localLikelihood);
        }
        catch (Exception) {
          Console.WriteLine($"exception when fitting {subExpr} ");
          continue;
        }
        // System.Console.WriteLine($"SSR after fitting: {Util.SSR(localLikelihood.Y, localNlr.Predict(localLikelihood.X))}");
        // 
        // System.Console.WriteLine(localLikelihood.ModelExpr);
        // System.Console.WriteLine(string.Join(" ", newTheta.Select(pi => pi.ToString("e4"))));
        // localNlr.WriteStatistics();
        var reducedSSR = Util.SSR(localLikelihood.Y, localNlr.Predict(localLikelihood.X));
        var ssrFactor = reducedSSR / refSSR;
        // this implements model statistics for Gaussian likelihood (SSR)
        var reducedN = newTheta.Length; // number of parameters
                                        // likelihood ratio test
        var deltaDoF = n - reducedN; // number of fewer parameters
        var deltaSSR = reducedSSR - refSSR;
        var s2Extra = deltaSSR / deltaDoF; // increase in deviance per parameter
        var fRatio = s2Extra / s2Full;
        var mse = reducedSSR / m;

        // "accept the partial value if the calculated ratio is lower than the table value"
        var f = double.IsInfinity(fRatio) || double.IsNaN(fRatio) ? 1.0 : alglib.fdistribution(deltaDoF, fullDoF, fRatio);
        // var f = 0.0;

        Console.WriteLine($"{ssrFactor,-11:e3} {deltaDoF,-6} {deltaSSR,-11:e3} {s2Extra,-11:e3} {fRatio,-11:e4} {1 - f,-10:e3} " +
          $"{localNlr.AICc - fullAICc,-11:f1} " +
          $"{localNlr.BIC - fullBIC,-11:f1}" +
          $"{mse,-11:e2} {subExpr} ");

        var impact = ssrFactor;

        yield return Tuple.Create(subExpr, impact, nlr.AICc - fullAICc, nlr.BIC - fullBIC);
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
          newExpr = Expr.Simplify(newExpr, theta, out var newTheta);
          yield return (newExpr, newTheta);
        }

        // replace each parameter with -1
        if (theta[i] < 0 && low[i] <= -1.0 && high[i] >= -1.0) {
          var newExpr = ReplaceParameterWithConstantVisitor.Execute(expr, expr.Parameters[0], i, -1);
          newExpr = Expr.Simplify(newExpr, theta, out var newTheta);
          yield return (newExpr, newTheta);
        }
        // replace each parameter with 1
        if (theta[i] > 0 && low[i] <= 1.0 && high[i] >= 1.0) {
          var newExpr = ReplaceParameterWithConstantVisitor.Execute(expr, expr.Parameters[0], i, 1);
          newExpr = Expr.Simplify(newExpr, theta, out var newTheta);
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
          newExpr = Expr.Simplify(newExpr, theta, out var newTheta);
          yield return (newExpr, newTheta);
          // }
        }
      }
    }

    /// <summary>
    /// Replaces each parameter in the model by a zero and re-fits the model for a comparison of nested models.
    /// </summary>
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
      var refSSR = Util.SSR(likelihood.Y, nlr.Predict(likelihood.X));

      var n = p.Length; // number of parameters
      var m = likelihood.Y.Length; // number of observations
      var impacts = new List<Tuple<int, double, double, Expression<Expr.ParametricFunction>, double[]>>();
      var s2Full = refSSR / (m - n);

      var fullAICc = nlr.AICc;
      var fullBIC = nlr.BIC;
      if (verbose) {
        Console.WriteLine(expr.Body);
        Console.WriteLine($"theta: {string.Join(", ", p.Select(pi => pi.ToString("e2")))}");
        Console.WriteLine($"Full model deviance: {refDeviance,-11:e4}, mean deviance: {refDeviance / m,-11:e4} SSR: {refSSR} MSE: {refSSR / m} AICc: {fullAICc,-11:f1} BIC: {fullBIC,-11:f1}");
        Console.WriteLine($"p{"idx",-5} {"val",-11} {"SSR_factor",-11} {"deltaDoF",-6} {"deltaSSR",-11} {"s2Extra":e3} {"fRatio",-11} {"p value",10} {"deltaAICc"} {"deltaBIC"} {"MSE"}");
      }


      // for(int paramIdx = 0; paramIdx<p.Length;paramIdx++) { 
      Parallel.For(0, p.Length, new ParallelOptions() { MaxDegreeOfParallelism = 1 }, (paramIdx) => {
        var v = new ReplaceParameterWithConstantVisitor(pParam, paramIdx, 0.0);
        var reducedExpression = (Expression<Expr.ParametricFunction>)v.Visit(expr);
        //Console.WriteLine($"Reduced: {reducedExpression}");
        // initial values for the reduced expression are the optimal parameters from the full model
        reducedExpression = Expr.SimplifyRepeated(reducedExpression, p, out var newP);

        // fit reduced model
        try {
          var localNlr = new NonlinearRegression();
          var reducedLikelihood = likelihood.Clone();
          reducedLikelihood.ModelExpr = reducedExpression;
          localNlr.Fit(newP, reducedLikelihood, maxIterations: maxIterations);
          //localNlr.WriteStatistics();


          /// We use a likelihood ratio test for model comparison which is exact for linear models.
          /// For nonlinear models the linear approximation is ok as long as the reduced model has similar fit as the full model.
          /// See "Bates and Watts, Nonlinear regression and its applications section on Comparing Models - Nested Models"
          /// for the argumentation. 
          // TODO: implement this based on "In All Likelihood" Section 6.6
          /* 
          if (verbose) {
            var reducedN = newP.Length; // number of parameters
                                        // likelihood ratio test
            var fullDoF = m - n;
            var deltaDoF = n - reducedN; // number of fewer parameters
            var deltaDeviance = nlr.Deviance - refDeviance;
            var devianceExtra = deltaDeviance / deltaDoF; // increase in deviance per parameter
            var fRatio = devianceExtra;

            // "accept the partial value if the calculated ratio is lower than the table value"
            var f = alglib.fdistribution(deltaDoF, fullDoF, fRatio);

            Console.WriteLine($"p{paramIdx,-5} {p[paramIdx],-11:e2} {devianceFactor,-11:e3} {deltaDoF,-6} {deltaDeviance,-11:e3} {devianceExtra,-11:e3} {fRatio,-11:e4}, {1 - f,-10:e3}, " +
              $"{localNlr.AICc - fullAICc,-11:f1} " +
              $"{localNlr.BIC - fullBIC,-11:f1}");
          }
          */

          var reducedSSR = Util.SSR(likelihood.Y, localNlr.Predict(likelihood.X));
          var ssrFactor = reducedSSR / refSSR;
          if (verbose) {
            // this implements model statistics for Gaussian likelihood (SSR)
            var reducedN = newP.Length; // number of parameters
                                        // likelihood ratio test
            var fullDoF = m - n;
            var deltaDoF = n - reducedN; // number of fewer parameters
            var deltaSSR = reducedSSR - refSSR;
            var s2Extra = deltaSSR / deltaDoF; // increase in deviance per parameter
            var fRatio = s2Extra / s2Full;
            var mse = reducedSSR / m;

            // "accept the partial value if the calculated ratio is lower than the table value"
            var f = alglib.fdistribution(deltaDoF, fullDoF, fRatio);

            Console.WriteLine($"p{paramIdx,-5} {p[paramIdx],-11:e2} {ssrFactor,-11:e3} {deltaDoF,-6} {deltaSSR,-11:e3} {s2Extra,-11:e3} {fRatio,-11:e4}, {1 - f,-10:e3}, " +
              $"{localNlr.AICc - fullAICc,-11:f1} " +
              $"{localNlr.BIC - fullBIC,-11:f1}" +
              $"{mse}");
          }

          lock (impacts) {
            impacts.Add(Tuple.Create(paramIdx, ssrFactor, localNlr.AICc - fullAICc, reducedExpression, newP));
          }
        }
        catch (Exception e) {
          // Console.WriteLine($"Exception {e.Message} for {reducedExpression}");
        }
      }
      );

      return impacts; // TODO improve interface
    }
  }
}

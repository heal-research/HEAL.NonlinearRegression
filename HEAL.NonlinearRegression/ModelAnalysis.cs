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
    public static Dictionary<int, double> VariableImportance(Expression<Expr.ParametricFunction> expr, double[,] X, double[] y, double[] p) {
      var nlr = new NonlinearRegression();
      nlr.Fit(p, expr, X, y);
      var stats0 = nlr.Statistics;
      var m = y.Length;
      var d = X.GetLength(1);

      var mean = new double[d];
      for (int i = 0; i < d; i++) {
        mean[i] = Enumerable.Range(0, m).Select(r => X[r, i]).Average();
      }

      var varExpl = new Dictionary<int, double>();
      for (int varIdx = 0; varIdx < d; varIdx++) {
        var newExpr = Expr.ReplaceVariableWithParameter(expr, (double[])stats0.paramEst.Clone(),
          varIdx, mean[varIdx], out var newThetaValues);

        newExpr = Expr.FoldParameters(newExpr, newThetaValues, out newThetaValues);

        nlr = new NonlinearRegression();
        nlr.Fit(newThetaValues, newExpr, X, y);
        if (nlr.Statistics == null) {
          Console.WriteLine("Problem while fitting");
          varExpl[varIdx] = 0.0;
        } else {
          var newStats = nlr.Statistics;
          // increase in variance for the reduced feature = variance explained by the feature
          varExpl[varIdx] = (newStats.SSR - stats0.SSR) / (y.Length * Util.Variance(y));
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
    public static IEnumerable<Tuple<Expression, double, double, double>> SubtreeImportance(Expression<Expr.ParametricFunction> expr, double[,] X, double[] y, double[] p) {
      var m = X.GetLength(0);
      var pParam = expr.Parameters[0];
      var xParam = expr.Parameters[1];
      var expressions = FlattenExpressionVisitor.Execute(expr.Body);
      var subexpressions = expressions.Where(e => !IsParameter(e, pParam) &&
                                                  !(e is ParameterExpression) &&
                                                  !(e is ConstantExpression));

      // fit the full model once for the baseline
      // TODO: we could skip this and get the baseline as parameter
      var nlr = new NonlinearRegression();
      nlr.Fit(p, expr, X, y);
      var stats0 = nlr.Statistics;
      p = (double[])stats0.paramEst.Clone();

      // we assume noiseSigma is the RMSE of the original model
      var noiseSigma = stats0.s;

      var fullAICc = ModelSelection.AICc(y, stats0.yPred, p.Length, noiseSigma);
      var fullBIC = ModelSelection.AICc(y, stats0.yPred, p.Length, noiseSigma);

      var impacts = new Dictionary<Expression, double>();

      foreach (var subExpr in subexpressions) {
        var subExprForEval = Expr.Broadcast(Expression.Lambda<Expr.ParametricFunction>(subExpr, pParam, xParam)).Compile();
        var eval = new double[m];
        subExprForEval(p, X, eval);
        var replValue = eval.Average();
        var reducedExpression = ReplaceSubexpressionWithParameterVisitor.Execute(expr, subExpr, p, replValue, out var newTheta);
        reducedExpression = Expr.FoldParameters(reducedExpression, newTheta, out newTheta);


        // fit reduced model
        nlr.Fit(newTheta, reducedExpression, X, y); // TODO make CLI parameter
        var reducedStats = nlr.Statistics;

        var impact = reducedStats.SSR / stats0.SSR;

        yield return Tuple.Create(subExpr, impact, ModelSelection.AICc(y, reducedStats.yPred, p.Length, noiseSigma) - fullAICc,
          ModelSelection.BIC(y, reducedStats.yPred, p.Length, noiseSigma) - fullBIC);
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
    public static IEnumerable<Tuple<int, double, double, Expression<Expr.ParametricFunction>, double[]>> NestedModelLiklihoodRatios(Expression<Expr.ParametricFunction> expr, double[,] X, double[] y, double[] p, int maxIterations, bool verbose = false) {
      var m = X.GetLength(0);
      var pParam = expr.Parameters[0];
      var xParam = expr.Parameters[1];

      // fit the full model once for the baseline
      // TODO: we could skip this and get the baseline as parameter

      var nlr = new NonlinearRegression();
      nlr.Fit(p, expr, X, y, maxIterations);

      var stats0 = nlr.Statistics;
      var noiseSigma = stats0.s;
      if (stats0 == null) return Enumerable.Empty<Tuple<int, double, double, Expression<Expr.ParametricFunction>, double[]>>(); // cannot fit the expression

      p = stats0.paramEst;
      var impacts = new List<Tuple<int, double, double, Expression<Expr.ParametricFunction>, double[]>>();

      var fullAIC = ModelSelection.AICc(y, stats0.yPred, p.Length, noiseSigma);
      var fullBIC = ModelSelection.BIC(y, stats0.yPred, p.Length, noiseSigma);
      if (verbose) {
        Console.WriteLine(expr.Body);
        Console.WriteLine($"theta: {string.Join(", ", p.Select(p => p.ToString("e2")))}");
        Console.WriteLine($"Full model SSR: {stats0.SSR,-11:e4}, MSE: {stats0.SSR / stats0.m,-11:e4}, AICc: {fullAIC,-11:f1} BIC: {fullBIC,-11:f1}");
        Console.WriteLine($"p{"idx",-5} {"val",-11} {"SSR_factor",-11} {"deltaDoF",-6} {"deltaSSR",-11} {"s2Extra":e3} {"fRatio",-11} {"p value",10} {"deltaAICc"} {"deltaBIC"}");
      }


      // for(int paramIdx = 0; paramIdx<p.Length;paramIdx++) { 
      Parallel.For(0, p.Length, (paramIdx) => {
        var v = new ReplaceParameterWithZeroVisitor(pParam, paramIdx);
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
          var nlr = new NonlinearRegression();
          nlr.Fit(newP, reducedExpression, X, y, maxIterations); // TODO make parameter
          var reducedStats = nlr.Statistics;

          var ssrFactor = reducedStats.SSR / stats0.SSR;

          // likelihood ratio test
          var fullDoF = stats0.m - stats0.n;
          var deltaDoF = stats0.n - reducedStats.n; // number of fewer parameters
          var deltaSSR = reducedStats.SSR - stats0.SSR;
          var s2Extra = deltaSSR / deltaDoF; // increase in SSR per parameter
          var fRatio = s2Extra / Math.Pow(stats0.s, 2);

          // "accept the partial value if the calculated ratio is lower than the table value"
          // var f = alglib.invfdistribution(deltaDoF, fullDoF, 0.05);  // TODO make alpha CLI parameter
          var f = alglib.fdistribution(deltaDoF, fullDoF, fRatio);

          if(verbose)
            Console.WriteLine($"p{paramIdx,-5} {p[paramIdx],-11:e2} {ssrFactor,-11:e3} {deltaDoF,-6} {deltaSSR,-11:e3} {s2Extra,-11:e3} {fRatio,-11:e4}, {1-f,-10:e3}, {ModelSelection.AICc(y, reducedStats.yPred, reducedStats.paramEst.Length, noiseSigma) - fullAIC,-11:f1} {ModelSelection.BIC(y, reducedStats.yPred, reducedStats.paramEst.Length, noiseSigma) - fullBIC,-11:f1}");

          lock (impacts) {
            impacts.Add(Tuple.Create(paramIdx, ssrFactor, ModelSelection.AICc(y, reducedStats.yPred, reducedStats.paramEst.Length, noiseSigma) - fullAIC, reducedExpression, newP));
          }
        }
        catch (Exception e) {
          // Console.WriteLine($"Exception {e.Message} for {reducedExpression}");
        }
        // yield return Tuple.Create(impact, (Expression)reducedExpression);
      }
      );

      return impacts; // TODO improve interface
    }

    private static bool IsParameter(Expression expr, ParameterExpression p) {
      return expr is BinaryExpression binExpr && binExpr.Left == p;
    }

  }
}

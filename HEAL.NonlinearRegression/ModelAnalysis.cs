using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Linq.Expressions;
using HEAL.Expressions;

namespace HEAL.NonlinearRegression {
  public static class ModelAnalysis {
    /// <summary>
    /// Replaces all references to a variable in the model by a parameter. Re-fits the model and returns the factor
    /// SSR_reduced / SSR_full as impact.
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

      var relSSR = new Dictionary<int, double>();
      for (int varIdx = 0; varIdx < d; varIdx++) {
        // TODO: merge replace and remove parameters
        var newExpr = Expr.ReplaceVariableWithParameter(expr, (double[])stats0.paramEst.Clone(),
          varIdx, mean[varIdx], out var newThetaValues);
        newExpr = Expr.FoldParameters(newExpr, newThetaValues, out newThetaValues);

        nlr = new NonlinearRegression();
        nlr.Fit(newThetaValues, newExpr, X, y);
        var newStats = nlr.Statistics;
	Console.WriteLine($"{newStats.SSR} {Util.Variance(y) * y.Length}");
	// increase in variance for the reduced feature = variance explained by the feature
        relSSR[varIdx] = (newStats.SSR - stats0.SSR) / (y.Length*Util.Variance(y)); // newStats.SSR / stats0.SSR;
      }

      return relSSR;
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
    public static IEnumerable<Tuple<double, Expression>> SubtreeImportance(Expression<Expr.ParametricFunction> expr, double[,] X, double[] y, double[] p) {
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
      p = stats0.paramEst;

      var impacts = new Dictionary<Expression, double>();

      foreach (var subExpr in subexpressions) {
        var subExprForEval = Expr.Broadcast(Expression.Lambda<Expr.ParametricFunction>(subExpr, pParam, xParam)).Compile();
        var eval = new double[m];
        subExprForEval(p, X, eval);
        var replValue = eval.Average();
        var reducedExpression = ReplaceSubexpressionWithParameterVisitor.Execute(expr, subExpr, p, replValue, out var newTheta);
        reducedExpression = Expr.FoldParameters(reducedExpression, newTheta, out newTheta);
        
        
        // fit reduced model
        nlr.Fit(newTheta, reducedExpression, X, y);
        var reducedStats = nlr.Statistics;

        var impact = reducedStats.SSR / stats0.SSR;
        
        yield return Tuple.Create(impact, subExpr);
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
    public static IEnumerable<Tuple<double, Expression>> NestedModelLiklihoodRatios(Expression<Expr.ParametricFunction> expr, double[,] X, double[] y, double[] p) {
      var m = X.GetLength(0);
      var pParam = expr.Parameters[0];
      var xParam = expr.Parameters[1];

      // fit the full model once for the baseline
      // TODO: we could skip this and get the baseline as parameter
      
      var nlr = new NonlinearRegression();
      nlr.Fit(p, expr, X, y);
      var stats0 = nlr.Statistics;
      p = stats0.paramEst;
      var impacts = new List<Tuple<double, Expression>>();

      for(int paramIdx = 0;paramIdx < p.Length;paramIdx++) {
        var v = new ReplaceParameterWithZeroVisitor(pParam, paramIdx);
        var reducedExpression = (Expression<Expr.ParametricFunction>)v.Visit(expr);
        //Console.WriteLine($"Reduced: {reducedExpression}");
        // initial values for the reduced expression are the optimal parameters from the full model
        reducedExpression = Expr.SimplifyAndRemoveParameters(reducedExpression, p, out var newP);
        //Console.WriteLine($"Simplified: {reducedExpression}");
        reducedExpression = Expr.FoldParameters(reducedExpression, newP, out newP);
        //Console.WriteLine($"Folded: {reducedExpression}");
        
        
        // fit reduced model
        nlr.Fit(newP, reducedExpression, X, y);
        var reducedStats = nlr.Statistics;

        var impact = reducedStats.SSR / stats0.SSR;
        
        // likelihood ratio test
        var fullDoF = stats0.m - stats0.n;
        var deltaDoF = stats0.n - reducedStats.n; // number of fewer parameters
        var deltaSSR = reducedStats.SSR - stats0.SSR;
        var s2Extra = deltaSSR / deltaDoF; // increase in SSR per parameter
        var fRatio = s2Extra / Math.Pow(stats0.s, 2); 
        // TODO check alglib nuget license
        var f = alglib.invfdistribution(deltaDoF, fullDoF, 0.05); // "accept the partial value if the calculated ratio is lower than the table value
        
        Console.WriteLine($"p{paramIdx,-5} {impact,-11:e4} {deltaDoF,-6} {fRatio,-11:e4} {f,11:e4} accept: {fRatio<f}");
      
        impacts.Add(Tuple.Create(impact, (Expression)reducedExpression));
        // yield return Tuple.Create(impact, (Expression)reducedExpression);
      }

      return impacts;
    }

    private static bool IsParameter(Expression expr, ParameterExpression p) {
      return expr is BinaryExpression binExpr && binExpr.Left == p;
    }

  }
}

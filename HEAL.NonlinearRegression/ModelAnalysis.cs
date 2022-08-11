using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Linq.Expressions;
using HEAL.Expressions;

namespace HEAL.NonlinearRegression {
  public static class ModelAnalysis {
    public static Dictionary<int, double> VariableImportance(Expression<Expr.ParametricFunction> expr, double[,] X, double[] y, double[] p) {
      var _func = Expr.Broadcast(expr).Compile();
      void func(double[] p, double[,] X, double[] f) => _func(p, X, f);
      
      var _jac = Expr.Jacobian(expr, p.Length).Compile();
      void jac(double[] p, double[,] X, double[] f, double[,] jac) => _jac(p, X, f, jac);
      
      var nlr = new NonlinearRegression();
      nlr.Fit(p, func, jac, X, y);
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
        
        var _newFunc = Expr.Broadcast(newExpr).Compile();
        void newFunc(double[] p, double[,] X, double[] f) => _newFunc(p, X, f);
      
        var _newJac = Expr.Jacobian(newExpr, newThetaValues.Length).Compile();
        void newJac(double[] p, double[,] X, double[] f, double[,] jac) => _newJac(p, X, f, jac);
        nlr = new NonlinearRegression(); // TODO does nlr store state in particular the number of parameters?
        nlr.Fit(newThetaValues, newFunc, newJac, X, y);
        var newStats = nlr.Statistics;
        relSSR[varIdx] = newStats.SSR / stats0.SSR;
      }

      return relSSR;
    }
    
    public static Dictionary<Expression, double> SubtreeImportance(Expression<Expr.ParametricFunction> expr, double[,] X, double[] y, double[] p) {
      var m = X.GetLength(0);
      var pParam = expr.Parameters[0];
      var xParam = expr.Parameters[1];
      var expressions = FlattenExpressionVisitor.Execute(expr.Body);
      var subexpressions = expressions.Where(e => !IsParameter(e, pParam) && 
                                                  !(e is ParameterExpression) && 
                                                  !(e is ConstantExpression));

      // fit the full model once for the baseline
      // TODO: we could skip this and get the baseline as parameter
      var _func = Expr.Broadcast(expr).Compile();
      void func(double[] p, double[,] X, double[] f) => _func(p, X, f);
      
      var _jac = Expr.Jacobian(expr, p.Length).Compile();
      void jac(double[] p, double[,] X, double[] f, double[,] jac) => _jac(p, X, f, jac);
      
      var nlr = new NonlinearRegression();
      nlr.Fit(p, func, jac, X, y);
      var stats0 = nlr.Statistics;
      p = stats0.paramEst;

      var impacts = new Dictionary<Expression, double>();

      foreach (var subExpr in subexpressions) {
        var subExprForEval = Expr.Broadcast(Expression.Lambda<Expr.ParametricFunction>(subExpr, pParam, xParam)).Compile();
        var eval = new double[m];
        subExprForEval(p, X, eval);
        var replValue = eval.Average();
        var reducedExpression = RemoveSubexpressionVisitor.Execute(expr, subExpr, p, replValue, out var newTheta);
        reducedExpression = Expr.FoldParameters(reducedExpression, newTheta, out newTheta);
        
        // TODO: allow to call NLR fit directly with an expression
        var _reducedFunc = Expr.Broadcast(reducedExpression).Compile();
        void reducedFunc(double[] p, double[,] X, double[] f) => _reducedFunc(p, X, f);
      
        var _reducedJac = Expr.Jacobian(reducedExpression, newTheta.Length).Compile();
        void reducedJac(double[] p, double[,] X, double[] f, double[,] jac) => _reducedJac(p, X, f, jac);

        // fit reduced model
        nlr = new NonlinearRegression();
        nlr.Fit(newTheta, reducedFunc, reducedJac, X, y);
        var reducedStats = nlr.Statistics;

        var impact = reducedStats.SSR / stats0.SSR;
        impacts.Add(subExpr, impact);
      }

      return impacts;
    }

    private static bool IsParameter(Expression expr, ParameterExpression p) {
      return expr is BinaryExpression binExpr && binExpr.Left == p;
    }

  }
}

using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Linq.Expressions;
using HEAL.Expressions;

namespace HEAL.NonlinearRegression {
  public static class VariableImportance {
    public static Dictionary<int, double> Calculate(Expression<Expr.ParametricFunction> expr, double[,] X, double[] y, double[] p) {
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
  }
}

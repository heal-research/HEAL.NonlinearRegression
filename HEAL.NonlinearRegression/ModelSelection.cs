using HEAL.Expressions;
using System;
using System.Linq;
using System.Linq.Expressions;

namespace HEAL.NonlinearRegression {
  public static class ModelSelection {
    public static double AIC(double logLikelihood, double dof) {
      return 2 * dof - 2 * logLikelihood;
    }
    public static double AICc(double logLikelihood, double dof, double numObservations) {
      return AIC(logLikelihood, dof) + 2 * dof * (dof + 1) / (numObservations - dof - 1);
    }

    public static double BIC(double logLikelihood, double dof, double numObservations) {
      return dof * Math.Log(numObservations) - 2 * logLikelihood;
    }

    // as described in https://arxiv.org/abs/2211.11461
    // Deaglan J. Bartlett, Harry Desmond, Pedro G. Ferreira, Exhaustive Symbolic Regression, 2022
    public static double MDL(double[] paramEst, LikelihoodBase likelihood) {
      // total description length:
      // L(D) = L(D|H) + L(H)

      // c_j are constants
      // theta_i are parameters
      // k is the number of nodes
      // n is the number of different symbols
      // Delta_i is inverse precision of parameter i
      // Delta_i are optimized to find minimum total description length
      // The paper shows that the optima for delta_i are sqrt(12/I_ii)
      // The formula implemented here is Equation (7).

      // L(D) = -log(L(theta)) + k log n - p/2 log 3
      //        + sum_j (1/2 log I_ii + log |theta_i| )
      var expr = likelihood.ModelExpr;
      int numNodes = Expr.NumberOfNodes(expr);
      var constants = Expr.CollectConstants(expr).ToList();
      var allSymbols = Expr.CollectSymbols(expr).ToList();
      int numParam = paramEst.Length;
      var fisherInfo = likelihood.FisherInformation(paramEst);

      for (int i = 0; i < numParam; i++) {
        // if the parameter estimate is not significantly different from zero
        if (Math.Abs(paramEst[i] / Math.Sqrt(12.0 / fisherInfo[i,i])) < 1.0) {
          // set param to zero (and skip in MDL calculation below)
          System.Console.WriteLine($"param[{i}] = 0 {expr}"); // for debugging
          paramEst[i] = 0.0;

          var v = new ReplaceParameterWithZeroVisitor(expr.Parameters[0], i);
          var reducedExpr = (Expression<Expr.ParametricFunction>)v.Visit(expr);
          var simplifiedExpr = Expr.SimplifyAndRemoveParameters(reducedExpr, paramEst, out var newParamEst);
          likelihood.ModelExpr = simplifiedExpr;

          if (newParamEst.Length == 0) return double.MaxValue; // no parameters left for fitting
          var nlr = new NonlinearRegression();
          nlr.Fit(newParamEst, likelihood); // TODO: here we can use FisherDiag for the scale for improved perf
          return MDL(nlr.ParamEst, nlr.Likelihood);
        } 
        // else if (Math.Round(paramEst[i]) != 0.0 && paramCodeLength(i) > constCodeLength(Math.Round(paramEst[i]))) {
        //   constants.Add(Math.Round(paramEst[i]));
        //   allSymbols.Add("const");
        //   paramEst[i] = 0.0;
        // }
      }

      int numSymbols = allSymbols.Distinct().Count();

      double constCodeLength(double val) {
        return Math.Log(Math.Abs(val)) + Math.Log(2);
      }

      double paramCodeLength(int idx) {
        return 0.5 * (-Math.Log(3) + Math.Log(fisherInfo[idx, idx])) + Math.Log(Math.Abs(paramEst[idx]));
      }

      var t1 = likelihood.NegLogLikelihood(paramEst);
      var t2 = numNodes * Math.Log(numSymbols) + constants.Sum(constCodeLength);
      var t3 = Enumerable.Range(0, numParam)
          .Where(i => paramEst[i] != 0.0) // skip parameter which are deactivated above
          .Sum(i => paramCodeLength(i));

      System.Console.WriteLine($"expr: {expr} nNodes: {numNodes}  nSym: {numSymbols} nPar: {numParam} " +
        $"DL(res): {t1:f2} " +
        $"DL(func): {t2:f2} " +
        $"DL(param): {t3:f2} " +
        $"constants: {string.Join(" ", constants.Select(ci => ci.ToString()))} " +
        $"params: {string.Join(" ", paramEst.Select(pi => pi.ToString("g4")))} " +
        $"diag(FI): {string.Join(" ", Enumerable.Range(0, numParam).Select(i => fisherInfo[i, i].ToString("g4")))}");

      return t1 + t2 + t3;
    }
  }
}

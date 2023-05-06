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
    public static double MDL(Expression<Expr.ParametricFunction> modelExpr, double[] paramEst, double logLikelihood, double[] diagFisherInfo) {
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
      int numNodes = Expr.NumberOfNodes(modelExpr);
      var constants = Expr.CollectConstants(modelExpr).ToList();
      var allSymbols = Expr.CollectSymbols(modelExpr).ToList();
      int numParam = paramEst.Length;

      for (int i = 0; i < numParam; i++) {
        // if the parameter estimate is not significantly different from zero
        if (Math.Abs(paramEst[i] / Math.Sqrt(12.0 / diagFisherInfo[i])) < 1.0) {
          // set param to zero (and skip in MDL calculation below)
          // TODO: this is an approximation. We should actually simplify the expression, re-optimize and call MDL method again.
          paramEst[i] = 0.0;
        } else if (Math.Round(paramEst[i]) != 0.0 && paramCodeLength(i) > constCodeLength(Math.Round(paramEst[i]))) {
          constants.Add(Math.Round(paramEst[i]));
          allSymbols.Add("const");
          paramEst[i] = 0.0;
        }
      }

      int numSymbols = allSymbols.Distinct().Count();

      // System.Console.WriteLine($"numNodes {numNodes}");
      // System.Console.WriteLine($"constants {string.Join(" ", constants.Select(ci => ci.ToString()))}");
      // System.Console.WriteLine($"numSymbols {numSymbols}");
      // System.Console.WriteLine($"symbols {string.Join(" ", Expr.CollectSymbols(modelExpr).Distinct().Select(s => s.ToString()))}");
      // System.Console.WriteLine($"numParam {numParam}");
      // System.Console.WriteLine($"diagFisherInfo {string.Join(" ", diagFisherInfo.Select(di => di.ToString()))}");

      double constCodeLength(double val) {
        return Math.Log(Math.Abs(val)) + Math.Log(2);
      }

      double paramCodeLength(int idx) {
        return 0.5 * (-Math.Log(3)) * Math.Log(diagFisherInfo[idx]) + Math.Log(Math.Abs(paramEst[idx]));
      }

      return -logLikelihood
        + numNodes * Math.Log(numSymbols)
        + constants.Sum(constCodeLength)
        + Enumerable.Range(0, numParam)
          .Where(i => paramEst[i] != 0.0) // skip parameter which are deactivated above
          .Sum(i => paramCodeLength(i));
    }
  }
}

using HEAL.Expressions;
using System;
using System.Collections.Generic;
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
    // This method simplifies the model by removing weakly determined parameters to optimize the DL.
    public static double DL(double[] paramEst, LikelihoodBase likelihood) {
      return DL(paramEst, likelihood, out _, out _, out _);
    }
    public static double DL(double[] paramEst, LikelihoodBase likelihood, out double log_func, out double log_param, out double[] fisherDiag) {
      // total description length: L(D) = L(D|H) + L(H)
      // c_j are constants
      // theta_i are parameters
      // k is the number of nodes
      // n is the number of different symbols
      // Delta_i is inverse precision of parameter i
      // Delta_i are optimized to find minimum total description length
      // The paper shows that the optima for delta_i are sqrt(12/I_ii)
      // The formula implemented here is Equation (7).

      // L(D) = -log(L(theta)) + k log n + sum_i (log|const_i| + log(2))
      //  - p/2 log 3 + sum_j (1/2 log I_ii + log |theta_i| )


      var expr = likelihood.ModelExpr;
      var numNodes = Expr.NumberOfNodes(expr);
      var constants = Expr.CollectConstants(expr).ToList();
      var allSymbols = Expr.CollectSymbols(expr).ToList();
      var numParam = paramEst.Length;
      var fisherInfo = likelihood.FisherInformation(paramEst);

      for (int i = 0; i < numParam; i++) {
        // If the parameter estimate is not significantly different from zero
        // then set it to zero and update the likelihood.
        if (Math.Abs(paramEst[i] / Math.Sqrt(12.0 / fisherInfo[i, i])) < 1.0) {
          // System.Console.Error.WriteLine($"Warning assuming param[{i}] = 0 in DL calculation for {expr}"); // for debugging
          paramEst = (double[])paramEst.Clone();
          paramEst[i] = 0.0;
        }
      }

      double paramCodeLength(double val, double fi) {
        return 0.5 * (-Math.Log(3) + Math.Log(fi)) + Math.Log(Math.Abs(val));
      }

      int numSymbols = allSymbols.Distinct().Count();

      var t1 = likelihood.NegLogLikelihood(paramEst);
      var t2 = numNodes * Math.Log(numSymbols) + constants.Sum(ConstCodeLength);
      var t3 = Enumerable.Range(0, numParam)
          .Where(i => paramEst[i] != 0.0) // skip parameter which are deactivated above
          .Sum(i => paramCodeLength(paramEst[i], fisherInfo[i, i]));

      // for debugging
      // System.Console.WriteLine($"expr: {expr} nNodes: {numNodes}  nSym: {numSymbols} nPar: {numParam} " +
      //   $"DL(res): {t1:g5} " +
      //   $"DL(func): {t2:g5} " +
      //   $"DL(param): {t3:g5} " +
      //   $"constants: {string.Join(" ", constants.Select(ci => ci.ToString()))} " +
      //   $"params: {string.Join(" ", paramEst.Select(pi => pi.ToString("g4")))} " +
      //   $"diag(FI): {string.Join(" ", Enumerable.Range(0, numParam).Select(i => fisherInfo[i, i].ToString("g4")))}");

      // for debugging
      log_func = t2;
      log_param = t3;
      fisherDiag = new double[paramEst.Length];
      for (int i = 0; i < paramEst.Length; i++) fisherDiag[i] = fisherInfo[i, i];

      return t1 + t2 + t3;
    }

    private static double ConstCodeLength(double val) {
      return Math.Log(Math.Abs(val)) + Math.Log(2);
    }

    // as described in https://arxiv.org/pdf/2304.06333.pdf Eq 6.
    // Deaglan J. Bartlett, Harry Desmond, Pedro G. Ferreira, Priors for Symbolic Regression, 2022
    public static double DLLattice(double[] paramEst, LikelihoodBase likelihood) {
      int numParam = paramEst.Length;
      var fisherInfo = likelihood.FisherInformation(paramEst);

      var expr = likelihood.ModelExpr;
      int numNodes = Expr.NumberOfNodes(expr);
      var constants = Expr.CollectConstants(expr).ToList();
      var allSymbols = Expr.CollectSymbols(expr).ToList();
      // fisherInfo = likelihood.FisherInformation(paramEst);

      var detFI = double.MaxValue;
      try {
        detFI = alglib.rmatrixdet(fisherInfo);
        if (detFI <= 0) {
          // System.Console.Error.WriteLine("FI not positive in MDLLattice. ");
          return double.MaxValue;
        }
      } catch (Exception e) {
        return double.MaxValue;
      }

      int numSymbols = allSymbols.Distinct().Count();


      double paramCodeLength(int idx) {
        return 0.5 * (1 - Math.Log(3));
      }

      var t1 = likelihood.NegLogLikelihood(paramEst);
      var t2 = numNodes * Math.Log(numSymbols) + constants.Sum(ConstCodeLength);
      var t3 = Enumerable.Range(0, numParam)
          .Where(i => paramEst[i] != 0.0) // skip parameter which are deactivated above
          .Sum(i => paramCodeLength(i))
          + 0.5 * Math.Log(detFI);

      return t1 + t2 + t3;
    }

    // as described in https://arxiv.org/pdf/2304.06333.pdf Eq 11.
    // Deaglan J. Bartlett, Harry Desmond, Pedro G. Ferreira, Priors for Symbolic Regression, 2022
    // posterior for parameters only (neglects function complexity)
    public static double NegativeEvidence(double[] paramEst, LikelihoodBase likelihood) {
      int numParam = paramEst.Length;

      var N = likelihood.Y.Length;
      var b = 1.0 / Math.Sqrt(N);

      return (1 - b) * likelihood.NegLogLikelihood(paramEst) - numParam / 2.0 * Math.Log(b);
    }
  }
}

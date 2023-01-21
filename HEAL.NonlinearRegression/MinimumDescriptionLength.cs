using System;
using System.Linq;

namespace HEAL.NonlinearRegression {
  public static class MinimumDescriptionLength {
    // as described in https://arxiv.org/abs/2211.11461
    // Deaglan J. Bartlett, Harry Desmond, Pedro G. Ferreira, Exhaustive Symbolic Regression, 2022
    public static double MDL(LeastSquaresStatistics stats, int numNodes, int numSymbols, double[] constants) {
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
      // TODO: it should be clarified whether equation 7 is missing the term for the constants.

      // L(D) = -log(L(theta)) + k log n - p/2 log 3
      //        + sum_j (1/2 log I_ii + log |theta_i| )

      var I = stats.FisherInformation;
      int numParam = stats.n;
      return -stats.LogLikelihood
        + numNodes * Math.Log(numSymbols) + constants.Sum(ci => Math.Log(Math.Abs(ci)))
        - numParam / 2.0 * Math.Log(3.0)
        + Enumerable.Range(0, numParam).Sum(i => 0.5 * Math.Log(I[i, i]) + Math.Log(Math.Abs(stats.paramEst[i])));
    }
  }
}

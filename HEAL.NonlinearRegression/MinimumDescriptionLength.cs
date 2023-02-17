using HEAL.Expressions;
using System;
using System.Linq;
using System.Linq.Expressions;


namespace HEAL.NonlinearRegression {
  public static class MinimumDescriptionLength {
    // as described in https://arxiv.org/abs/2211.11461
    // Deaglan J. Bartlett, Harry Desmond, Pedro G. Ferreira, Exhaustive Symbolic Regression, 2022
    public static double MDL(Expression<Expr.ParametricFunction> modelExpr, double[] paramEst, double[] y, double[,] x, bool approxHessian = false) {
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
      var constants = Expr.CollectConstants(modelExpr);
      var numSymbols = Expr.CollectSymbols(modelExpr).Distinct().Count();
      int numParam = paramEst.Length;
      var yPred = new double[y.Length];
      Expr.Broadcast(modelExpr).Compile()(paramEst, x, yPred);
      var FIdiag = FisherInformationDiag(x, y, yPred, modelExpr, paramEst, approxHessian);
      
      // TODO: for negative constants and negative parameters we would need to account for an unary sign in the expression
      return -LogLikelihood(y, yPred)
        + numNodes * Math.Log(numSymbols) + constants.Sum(ci => Math.Log(Math.Abs(ci)))
        - numParam / 2.0 * Math.Log(3.0)
        + Enumerable.Range(0, numParam).Sum(i => 0.5 * Math.Log(FIdiag[i]) + Math.Log(Math.Abs(paramEst[i])));
    }

    private static double LogLikelihood(double[] y, double[] yPred) {
      // this assumes that parameters are at the optimum
      var m = y.Length;
      var SSR = y.Zip(yPred, (yi, ypi) => Math.Pow(yi - ypi, 2)).Sum();
      var s2 = SSR / m; // optimal value for s2 in the likelihood function for the maximum likelihood parameter estimate. s2 is assumed to be fixed
      return -m / 2 * Math.Log(2 * Math.PI) - m / 2 * Math.Log(s2) - SSR / (2.0 * s2);
      // return -m / 2.0 * Math.Log(2 * Math.PI) - m / 2.0 * Math.Log(s2) - m / 2.0;
      // return -m / 2.0 * (Math.Log(2 * Math.PI) + Math.Log(s2) + 1);
    }

    // diag(I) for the normal log likelihood
    public static double[] FisherInformationDiag(double[,] X, double[] y, double[] yPred, Expression<Expr.ParametricFunction> modelExpr, double[] paramEst, bool approxHessian = false) {
      var n = paramEst.Length;
      Expr.ParametricHessianDiagFunction hess = null;
      if (!approxHessian) {
        hess = Expr.HessianDiag(modelExpr, n).Compile();
      }
      var grad = Expr.Gradient(modelExpr, n).Compile();

      var SSR = y.Zip(yPred, (yi, ypi) => Math.Pow(yi - ypi, 2)).Sum();
      var s2 = SSR / yPred.Length; // estimated error variance

      var fimDiag = new double[n]; // diagonal of Fisher information matrix of log likelihood
      var hDiagModel = new double[n]; // hessian of the model
      var gradModel = new double[n]; // gradient of the model
      var d = X.GetLength(1);
      var xi = new double[d];
      for (int r = 0; r < y.Length; r++) {
        Buffer.BlockCopy(X, r * d * sizeof(double), xi, 0, d * sizeof(double)); // copy one row
        var res = y[r] - yPred[r];
        if(!approxHessian) hess(paramEst, xi, hDiagModel); // else hDiagModel = zeros
        grad(paramEst, xi, gradModel);
        for (int pIdx = 0; pIdx < n; pIdx++) {
          fimDiag[pIdx] += (res * hDiagModel[pIdx] + gradModel[pIdx] * gradModel[pIdx]) / s2;
        }
      }

      return fimDiag;
    }
  }
}

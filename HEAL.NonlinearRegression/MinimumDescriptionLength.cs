using HEAL.Expressions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;


namespace HEAL.NonlinearRegression {
  public static class MinimumDescriptionLength {
    // as described in https://arxiv.org/abs/2211.11461
    // Deaglan J. Bartlett, Harry Desmond, Pedro G. Ferreira, Exhaustive Symbolic Regression, 2022
    public static double MDL(Expression<Expr.ParametricFunction> modelExpr, double[] paramEst, double logLikelihood, double[] y, double noiseSigma, double[,] x, bool approxHessian = false) {
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
      var FIdiag = FisherInformationDiag(x, y, yPred, modelExpr, paramEst, noiseSigma, approxHessian); // here we would actually need the Hessian of the log likelihood (should be a parameter)
      
      // TODO: for negative constants and negative parameters we would need to account for an unary sign in the expression
      return -logLikelihood
        + numNodes * Math.Log(numSymbols) + constants.Sum(ci => Math.Log(Math.Abs(ci)))
        - numParam / 2.0 * Math.Log(3.0)
        + Enumerable.Range(0, numParam).Sum(i => 0.5 * Math.Log(FIdiag[i]) + Math.Log(Math.Abs(paramEst[i])));
    }

    public static double MDLFreq(Expression<Expr.ParametricFunction> modelExpr, double[] paramEst, double logLikelihood, double[] y, double noiseSigma, double[,] x, bool approxHessian = false) {
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
      int numParam = paramEst.Length;
      var yPred = new double[y.Length];
      Expr.Broadcast(modelExpr).Compile()(paramEst, x, yPred);
      var FIdiag = FisherInformationDiag(x, y, yPred, modelExpr, paramEst, noiseSigma, approxHessian);

      var codeLen = new Dictionary<string, double>() {
        { "var", 0.66},
        { "param", 0.66},
        { "const", 0.66},
        { "+", 2.50},
        { "-", 3.4},
        { "*", 1.72},
        { "/", 2.60},
        { "Math.Log()", 4.76},
        { "Math.Exp()", 4.78},
        { "Math.Pow()", 2.53},
        { "Math.Sin()", 6},
        { "Math.Cos()", 5.5},
        { "Math.Sqrt()", 4.78},
        { "Functions.Cbrt()", 6},
        { "Functions.AQ()", 6},
      };

      double CodeLen(string sy) {
        if (sy.StartsWith("var")) return codeLen["var"];
        else return codeLen[sy];
      }


      var usedVariables = Expr.CollectSymbols(modelExpr).Where(sy => sy.StartsWith("var")).ToArray();
      var distinctVariables = usedVariables.Distinct().ToArray();

      // TODO: for negative constants and negative parameters we would need to account for an unary sign in the expression
      return -logLikelihood
        + Expr.CollectSymbols(modelExpr).Select(sy => CodeLen(sy)).Sum() // symbols in the expr
        + usedVariables.Length * Math.Log(distinctVariables.Length) // fixed length encoding for variables
        + constants.Sum(ci => Math.Log(Math.Abs(ci))) // constants
        - numParam / 2.0 * Math.Log(3.0) + Enumerable.Range(0, numParam).Sum(i => 0.5 * Math.Log(FIdiag[i]) + Math.Log(Math.Abs(paramEst[i]))) // parameter values
        ;
    }

    // diag(I) for the normal log likelihood
    public static double[] FisherInformationDiag(double[,] X, double[] y, double[] yPred, Expression<Expr.ParametricFunction> modelExpr, double[] paramEst, double noiseSigma, bool approxHessian = false) {
      var n = paramEst.Length;
      Expr.ParametricHessianDiagFunction hess = null;
      if (!approxHessian) {
        hess = Expr.HessianDiag(modelExpr, n).Compile();
      }
      var grad = Expr.Gradient(modelExpr, n).Compile();

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
          fimDiag[pIdx] += (res * hDiagModel[pIdx] + gradModel[pIdx] * gradModel[pIdx]) / (noiseSigma * noiseSigma);
        }
      }

      return fimDiag;
    }
  }
}

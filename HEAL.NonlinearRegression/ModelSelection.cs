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

    public static double DLWithIntegerSnap(double[] paramEst, LikelihoodBase likelihood) {

      // clone parameters and likelihood for pruning and evaluation (caller continues to work with original expression)
      paramEst = (double[])paramEst.Clone();
      likelihood = likelihood.Clone();
      IntegerSnapPruning(ref paramEst, likelihood, DL);
      return DL(paramEst, likelihood);
    }

    // as described in https://arxiv.org/abs/2211.11461
    // Deaglan J. Bartlett, Harry Desmond, Pedro G. Ferreira, Exhaustive Symbolic Regression, 2022
    // This method simplifies the model by removing weakly determined parameters to optimize the DL.
    public static double DL(double[] paramEst, LikelihoodBase likelihood) {
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
          System.Console.Error.WriteLine($"Warning assuming param[{i}] = 0 in DL calculation for {expr}"); // for debugging
          paramEst = (double[])paramEst.Clone();
          paramEst[i] = 0.0;
        }
      }

      static double paramCodeLength(double val, double fisherInfo) {
        return 0.5 * (-Math.Log(3) + Math.Log(fisherInfo)) + Math.Log(Math.Abs(val));
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

      return t1 + t2 + t3;
    }

    private static double ConstCodeLength(double val) {
      return Math.Log(Math.Abs(val)) + Math.Log(2);
    }


    public static double DLLatticeWithIntegerSnap(double[] paramEst, LikelihoodBase likelihood) {
      // clone parameters and likelihood for pruning and evaluation (caller continues to work with original expression)
      paramEst = (double[])paramEst.Clone();
      likelihood = likelihood.Clone();
      IntegerSnapPruning(ref paramEst, likelihood, DLLattice);
      return DLLattice(paramEst, likelihood);
    }

    // as described in https://arxiv.org/pdf/2304.06333.pdf Eq 6.
    // Deaglan J. Bartlett, Harry Desmond, Pedro G. Ferreira, Priors for Symbolic Regression, 2022
    public static double DLLattice(double[] paramEst, LikelihoodBase likelihood) {
      int numParam = paramEst.Length;
      var fisherInfo = likelihood.FisherInformation(paramEst);

      for (int i = 0; i < numParam; i++) {
        // if the parameter estimate is not significantly different from zero
        if (Math.Abs(paramEst[i] / Math.Sqrt(12.0 / fisherInfo[i, i])) < 1.0) {
          System.Console.Error.WriteLine($"Warning setting param[{i}] = 0 in DL (lattice) calculation for {likelihood.ModelExpr}"); // for debugging
          paramEst = (double[])paramEst.Clone();
          paramEst[i] = 0.0;
        }
      }

      var expr = likelihood.ModelExpr;
      int numNodes = Expr.NumberOfNodes(expr);
      var constants = Expr.CollectConstants(expr).ToList();
      var allSymbols = Expr.CollectSymbols(expr).ToList();


      var detFI = alglib.rmatrixdet(fisherInfo);
      if (detFI <= 0) throw new InvalidOperationException("FI not positive in MDLLattice. ");

      int numSymbols = allSymbols.Distinct().Count();


      static double paramCodeLength(int idx) {
        return 0.5 * (1 - Math.Log(3));
      }

      var t1 = likelihood.NegLogLikelihood(paramEst);
      var t2 = numNodes * Math.Log(numSymbols) + constants.Sum(ConstCodeLength);
      var t3 = Enumerable.Range(0, numParam)
          .Where(i => paramEst[i] != 0.0) // skip parameter which are deactivated above
          .Sum(i => paramCodeLength(i))
          + 0.5 * Math.Log(detFI);

      // System.Console.WriteLine($"nNodes: {numNodes}  nSym: {numSymbols} nPar: {numParam} " +
      //   $"DL(sum): {t1 + t2 + t3:g5} " +
      //   $"DL(res): {t1:g5} " +
      //   $"DL(func): {t2:g5} " +
      //   $"DL(param): {t3:g5} " +
      //   $"expr: {expr} " +
      //   $"constants: {string.Join(" ", constants.Select(ci => ci.ToString()))} " +
      //   $"params: {string.Join(" ", paramEst.Select(pi => pi.ToString("g4")))} " +
      //   $"diag(FI): {string.Join(" ", Enumerable.Range(0, numParam).Select(i => fisherInfo[i, i].ToString("g4")))}");

      return t1 + t2 + t3;
    }

    // as described in https://arxiv.org/pdf/2304.06333.pdf Eq 11.
    // Deaglan J. Bartlett, Harry Desmond, Pedro G. Ferreira, Priors for Symbolic Regression, 2022
    // posterior for parameters only (neglects function complexity)
    public static double Evidence(double[] paramEst, LikelihoodBase likelihood) {
      int numParam = paramEst.Length;

      var N = likelihood.Y.Length;
      var b = 1.0 / Math.Sqrt(N);

      return (1 - b) * likelihood.NegLogLikelihood(paramEst) + numParam / 2.0 * Math.Log(b);
    }

    // Algorithm for Pruning (replacing parameters by integers):
    // Replace parameters by constants greedily.
    // 1. Calc parameter statistics and select parameters with the largest zScore (least certain)
    // 2. Generate integer constant values towards zero by repeatedly halving the rounded parameter value
    // 3. For each constant value calculate the DL (ignoring the DL of the function because it is unchanged)
    // 4. Use the value with best DL as replacement, simplify and re-fit the function.
    // 5. If the simplified function has better DL accept as new model and goto 1.
    //
    // The description length function is not convex over theta (sum of negLogLik (assumed to be convex around the MLE) and logarithms of theta (concave)).
    // -> multiple local optima in DL
    // This method modifies the parameter vector and the likelihood
    private static void IntegerSnapPruning(ref double[] paramEst, LikelihoodBase likelihood, Func<double[], LikelihoodBase, double> DL) {
      var fisherInfo = likelihood.FisherInformation(paramEst);

      var n = paramEst.Length;
      var precision = new double[n];
      for (int i = 0; i < n; i++) {
        precision[i] = Math.Abs(paramEst[i]) * Math.Sqrt(fisherInfo[i, i]);
      }

      var idx = Enumerable.Range(0, n).ToArray();
      Array.Sort(precision, idx); // order paramIdx by zScore (smallest first)

      var paramIdx = idx[0];
      // generate integer alternatives for the parameter by repeatedly halving the value (TODO: other schemes useful / better here?)
      var constValues = new List<int> {
          (int)Math.Round(paramEst[paramIdx])
        };
      while (constValues.Last() > 0) {
        constValues.Add(constValues.Last() / 2); // integer divison
      }

      var origDL = DL(paramEst, likelihood);

      var origParamValue = paramEst[paramIdx];
      var origExpr = likelihood.ModelExpr;
      var bestDL = double.MaxValue; // likelihood and DL of constant for best replacement
      var bestConstValue = double.NaN;

      // try all constant values and find best 
      foreach (var constValue in constValues) {
        paramEst[paramIdx] = constValue;
        double curDL;
        if (constValue == 0) {
          curDL = likelihood.NegLogLikelihood(paramEst);
        } else {
          curDL = likelihood.NegLogLikelihood(paramEst) + ConstCodeLength(constValue);
        }
        // System.Console.WriteLine($"param: {paramIdx} const:{constValue} DL:{curDL} negLL:{likelihood.NegLogLikelihood(paramEst)} DL(const):{ConstCodeLength(constValue)}");
        if (curDL < bestDL) {
          bestDL = curDL;
          bestConstValue = constValue;
        }
      }


      // replace parameter with best constant value, simplify and re-fit
      paramEst[paramIdx] = bestConstValue;

      var v = new ReplaceParameterWithConstantVisitor(origExpr.Parameters[0], paramIdx, bestConstValue);
      var reducedExpr = (Expression<Expr.ParametricFunction>)v.Visit(origExpr);
      var simplifiedExpr = Expr.SimplifyAndRemoveParameters(reducedExpr, paramEst, out var simplifiedParamEst);
      if (simplifiedParamEst.Length > 0) {
        likelihood.ModelExpr = simplifiedExpr;
        var nlr = new NonlinearRegression();
        nlr.Fit(simplifiedParamEst, likelihood); // TODO: here we could use FisherDiag for the scale for improved perf
        if (nlr.ParamEst == null) {
          System.Console.Error.WriteLine("Problem while re-fitting pruned expression in DL calculation.");
          likelihood.ModelExpr = origExpr;
          paramEst[paramIdx] = origParamValue;
        } else {
          var newDL = DL(nlr.ParamEst, likelihood);
          // if the new DL is shorter then continue with next parameter
          if (newDL < origDL) {
            System.Console.WriteLine("######################################");
            System.Console.WriteLine($"In DL: replaced parameter[{paramIdx}]={origParamValue} by constant {bestConstValue}:");
            System.Console.WriteLine($"Pruned model: {likelihood.ModelExpr}");

            likelihood.LaplaceApproximation(nlr.ParamEst).WriteStatistics(System.Console.Out);

            paramEst = nlr.ParamEst;
            IntegerSnapPruning(ref paramEst, likelihood, DL);
          } else {
            // no improvement by replacing the parameter with a constant -> restore original expression and return
            likelihood.ModelExpr = origExpr;
            paramEst[paramIdx] = origParamValue;
          }
        }
      }
    }


  }
}

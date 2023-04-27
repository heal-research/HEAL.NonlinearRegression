﻿using HEAL.Expressions;
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
      var constants = Expr.CollectConstants(modelExpr);
      var numSymbols = Expr.CollectSymbols(modelExpr).Distinct().Count();
      int numParam = paramEst.Length;

      // TODO: check if parameter estimate is significantly different from zero
      for (int i = 0; i < numParam; i++) {
        // if the parameter estimate is not significanlty different from zero
        if (paramEst[i] / Math.Sqrt(12.0 / diagFisherInfo[i]) < 1.0) {
          // TODO: set param to zero and calculate MDL for the manipulated expression
          
        }
      }

      System.Console.WriteLine($"numNodes {numNodes}");
      System.Console.WriteLine($"constants {string.Join(" ", constants.Select(ci => ci.ToString()))}");
      System.Console.WriteLine($"numSymbols {numSymbols}");
      System.Console.WriteLine($"numParam {numParam}");
      System.Console.WriteLine($"diagFisherInfo {string.Join(" ", diagFisherInfo.Select(di => di.ToString()))}");

      // TODO: for negative constants we would need to account for an unary sign in the expression
      return -logLikelihood
        + numNodes * Math.Log(numSymbols) + constants.Sum(ci => Math.Log(Math.Abs(ci)))
        - numParam / 2.0 * Math.Log(3.0)
        + Enumerable.Range(0, numParam).Sum(i => 0.5 * Math.Log(diagFisherInfo[i]) + Math.Log(Math.Abs(paramEst[i])));
    }


    // for experimental code which considers frequencies of symbols occuring in named expressions
    private static Dictionary<string, double> codeLen = new Dictionary<string, double>() {
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
        { "Functions.Logistic()", 6 }
      };
    // for experimental code which considers frequencies of symbols occuring in named expressions
    public static double MDLFreq(Expression<Expr.ParametricFunction> modelExpr, double[] paramEst, double logLikelihood, double[] diagFisherInfo) {
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

      var usedVariables = Expr.CollectSymbols(modelExpr).Where(sy => sy.StartsWith("var")).ToArray();
      var distinctVariables = usedVariables.Distinct().ToArray();

      // TODO: for negative constants and negative parameters we would need to account for an unary sign in the expression
      return -logLikelihood
        + Expr.CollectSymbols(modelExpr).Select(sy => CodeLen(sy)).Sum() // symbols in the expr
        + usedVariables.Length * Math.Log(distinctVariables.Length) // fixed length encoding for variables
        + constants.Sum(ci => Math.Log(Math.Abs(ci))) // constants
        - numParam / 2.0 * Math.Log(3.0) + Enumerable.Range(0, numParam).Sum(i => 0.5 * Math.Log(diagFisherInfo[i]) + Math.Log(Math.Abs(paramEst[i]))) // parameter values
        ;
    }

    private static double CodeLen(string sy) {
      if (sy.StartsWith("var")) return codeLen["var"];
      else return codeLen[sy];
    }
  }
}

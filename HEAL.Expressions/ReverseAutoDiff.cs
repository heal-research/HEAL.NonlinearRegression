using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;

namespace HEAL.Expressions {

  // reverse mode autodiff (in combination with EvaluationVisitor)
  public static class ReverseAutoDiff {
    public static void CalculateJac(Expression expr, ParameterExpression param, Dictionary<Expression, double[]> exprValues, double[,] jac) {
      int batchSize = jac.GetLength(0); // TODO: batched evaluation
      List<(Expression expr, double[] diff)> diffs;
      List<int> lengths;

      Array.Clear(jac, 0, jac.Length);

      diffs = new List<(Expression expr, double[] diff)>();
      lengths = new List<int>();

      // FlattenExpressionVisitor returns post-order representation
      // TODO: skip arguments of arrayIndex expressions for perf
      foreach (var subexpr in FlattenExpressionVisitor.Execute(expr)) {
        var numChildren = 0;
        if (subexpr is UnaryExpression) {
          numChildren = 1;
        } else if (subexpr is BinaryExpression) {
          numChildren = 2;
        } else if (subexpr is MethodCallExpression callExpr) {
          numChildren = callExpr.Arguments.Count;
        }

        var len = 1;
        var cIdx = lengths.Count - 1;
        for (int c = 0; c < numChildren; c++) {
          len += lengths[cIdx];
          cIdx = cIdx - lengths[cIdx];
        }

        diffs.Add((subexpr, new double[batchSize]));
        lengths.Add(len);
      }

      // init backpropagate
      var lastDiff = diffs.Last().diff;
      for (int i = 0; i < batchSize; i++) lastDiff[i] = 1.0;

      // backpropagate
      for (int exprIdx = diffs.Count - 1; exprIdx >= 0; exprIdx--) {
        var curExpr = diffs[exprIdx].expr;
        var curDiff = diffs[exprIdx].diff;
        if (curExpr is UnaryExpression unaryExpr) {
          var opDiff = diffs[exprIdx - 1].diff;
          if (unaryExpr.NodeType == ExpressionType.Negate) {
            for (int i = 0; i < batchSize; i++) opDiff[i] = -1.0 * curDiff[i];
          } else if (unaryExpr.NodeType == ExpressionType.UnaryPlus) {
            for (int i = 0; i < batchSize; i++) opDiff[i] = curDiff[i];
          } else throw new NotSupportedException("Unknown operation");
        } else if (curExpr is BinaryExpression binExpr) {

          if (binExpr.NodeType == ExpressionType.ArrayIndex) {
            var index = binExpr.Right;
            if (index.NodeType != ExpressionType.Constant) throw new NotSupportedException("only constant indices for parameter are allowed");

            var idx = (int)((ConstantExpression)index).Value;
            if (binExpr.Left == param) {
              for (int i = 0; i < batchSize; i++) { jac[i, idx] += curDiff[i]; } // parameter (add in case parameters would appear multiple times)
            } else {
              // nothing to do (until we want to calculate the gradient for variables)
            }
          } else {
            var leftDiff = diffs[exprIdx - 1 - lengths[exprIdx - 1]].diff;
            var rightDiff = diffs[exprIdx - 1].diff;
            var leftEval = exprValues[binExpr.Left];
            var rightEval = exprValues[binExpr.Right];

            switch (binExpr.NodeType) {
              case ExpressionType.Add: { for (int i = 0; i < batchSize; i++) { leftDiff[i] = curDiff[i]; rightDiff[i] = curDiff[i]; } break; }
              case ExpressionType.Subtract: { for (int i = 0; i < batchSize; i++) { leftDiff[i] = curDiff[i]; rightDiff[i] = -curDiff[i]; } break; }
              case ExpressionType.Multiply: {
                  for (int i = 0; i < batchSize; i++) {
                    leftDiff[i] = curDiff[i] * rightEval[i];
                    rightDiff[i] = curDiff[i] * leftEval[i];
                  }
                  break;
                }
              case ExpressionType.Divide: {
                  for (int i = 0; i < batchSize; i++) {
                    leftDiff[i] = curDiff[i] / rightEval[i];
                    rightDiff[i] = -curDiff[i] * leftEval[i] / (rightEval[i] * rightEval[i]);
                  }
                  break;
                }
              default: throw new NotSupportedException(binExpr.ToString());
            }
          }
        } else if (curExpr is MethodCallExpression callExpr) {
          var arg0Eval = exprValues[callExpr.Arguments[0]];
          var arg0Diff = diffs[exprIdx - 1].diff;
          if (callExpr.Method == sin) {
            for (int i = 0; i < batchSize; i++) { arg0Diff[i] = curDiff[i] * Math.Cos(arg0Eval[i]); }
          } else if (callExpr.Method == cos) {
            for (int i = 0; i < batchSize; i++) { arg0Diff[i] = curDiff[i] * -Math.Sin(arg0Eval[i]); }
          } else if (callExpr.Method == exp) {
            for (int i = 0; i < batchSize; i++) { arg0Diff[i] = curDiff[i] * Math.Exp(arg0Eval[i]); }
          } else if (callExpr.Method == log) {
            for (int i = 0; i < batchSize; i++) { arg0Diff[i] = curDiff[i] / arg0Eval[i]; }
          } else if (callExpr.Method == tanh) {
            for (int i = 0; i < batchSize; i++) { arg0Diff[i] = curDiff[i] * 2.0 / (Math.Cosh(2.0 * arg0Eval[i]) + 1); }
          } else if (callExpr.Method == sqrt) {
            for (int i = 0; i < batchSize; i++) { arg0Diff[i] = curDiff[i] * 0.5 / Math.Sqrt(arg0Eval[i]); }
          } else if (callExpr.Method == cbrt) {
            for (int i = 0; i < batchSize; i++) { arg0Diff[i] = curDiff[i] / 3.0 / Math.Pow(Math.Cbrt(arg0Eval[i]), 2); }
          } else if (callExpr.Method == pow) {
            var exponentDiff = diffs[exprIdx - 1].diff;
            var exponentEval = exprValues[callExpr.Arguments[1]];
            var baseDiff = diffs[exprIdx - 1 - lengths[exprIdx - 1]].diff;
            var baseEval = exprValues[callExpr.Arguments[0]];
            for (int i = 0; i < batchSize; i++) { baseDiff[i] = curDiff[i] * exponentEval[i] * Math.Pow(baseEval[i], exponentEval[i] - 1); }
            for (int i = 0; i < batchSize; i++) { exponentDiff[i] = curDiff[i] * Math.Pow(baseEval[i], exponentEval[i]) * Math.Log(baseEval[i]); }
          } else if (callExpr.Method == abs) {
            for (int i = 0; i < batchSize; i++) { arg0Diff[i] = curDiff[i] * Math.Sign(arg0Eval[i]); }
          } else if (callExpr.Method == sign) {
            for (int i = 0; i < batchSize; i++) { arg0Diff[i] = 0; }
          } else if (callExpr.Method == logistic) {
            for (int i = 0; i < batchSize; i++) { arg0Diff[i] = curDiff[i] * Functions.LogisticPrime(arg0Eval[i]); }
          } else if (callExpr.Method == invlogistic) {
            for (int i = 0; i < batchSize; i++) { arg0Diff[i] = curDiff[i] * Functions.InvLogisticPrime(arg0Eval[i]); }
          } else if (callExpr.Method == logisticPrime) {
            for (int i = 0; i < batchSize; i++) { arg0Diff[i] = curDiff[i] * Functions.LogisticPrimePrime(arg0Eval[i]); }
          } else if (callExpr.Method == invlogisticPrime) {
            for (int i = 0; i < batchSize; i++) { arg0Diff[i] = curDiff[i] * Functions.InvLogisticPrimePrime(arg0Eval[i]); }
          } else throw new NotSupportedException($"Unsupported method call {callExpr.Method.Name}");
        }
      }
    }

    private static readonly MethodInfo abs = typeof(Math).GetMethod("Abs", new[] { typeof(double) });
    private static readonly MethodInfo sin = typeof(Math).GetMethod("Sin", new[] { typeof(double) });
    private static readonly MethodInfo cos = typeof(Math).GetMethod("Cos", new[] { typeof(double) });
    private static readonly MethodInfo exp = typeof(Math).GetMethod("Exp", new[] { typeof(double) });
    private static readonly MethodInfo log = typeof(Math).GetMethod("Log", new[] { typeof(double) });
    private static readonly MethodInfo tanh = typeof(Math).GetMethod("Tanh", new[] { typeof(double) });
    private static readonly MethodInfo cosh = typeof(Math).GetMethod("Cosh", new[] { typeof(double) });
    private static readonly MethodInfo sqrt = typeof(Math).GetMethod("Sqrt", new[] { typeof(double) });
    private static readonly MethodInfo cbrt = typeof(Functions).GetMethod("Cbrt", new[] { typeof(double) });
    private static readonly MethodInfo pow = typeof(Math).GetMethod("Pow", new[] { typeof(double), typeof(double) });
    private static readonly MethodInfo sign = typeof(Functions).GetMethod("Sign", new[] { typeof(double) }); // for deriv abs(x)
    private static readonly MethodInfo logistic = typeof(Functions).GetMethod("Logistic", new[] { typeof(double) });
    private static readonly MethodInfo invlogistic = typeof(Functions).GetMethod("InvLogistic", new[] { typeof(double) });
    private static readonly MethodInfo logisticPrime = typeof(Functions).GetMethod("LogisticPrime", new[] { typeof(double) }); // deriv of logistic
    private static readonly MethodInfo logisticPrimePrime = typeof(Functions).GetMethod("LogisticPrimePrime", new[] { typeof(double) }); // deriv of logistic
    private static readonly MethodInfo invlogisticPrime = typeof(Functions).GetMethod("InvLogisticPrime", new[] { typeof(double) });
    private static readonly MethodInfo invlogisticPrimePrime = typeof(Functions).GetMethod("InvLogisticPrimePrime", new[] { typeof(double) });

  }
}

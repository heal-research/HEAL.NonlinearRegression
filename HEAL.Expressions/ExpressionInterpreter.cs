using System;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;

namespace HEAL.Expressions {
  // prepares data structures for repeated efficient evaluation of a single expression
  public class ExpressionInterpreter {
    private readonly int batchSize;
    private readonly int m;
    private readonly double[][] x;
    private readonly double[][] xBuf;
    private readonly ParameterExpression thetaParam;
    private readonly ParameterExpression xParam;
    private readonly Expression[] exprArr;
    private readonly Instruction[] instrArr;

    // x is column oriented
    public ExpressionInterpreter(Expression<Expr.ParametricFunction> expr, double[][] x, int nRows, int batchSize = 256) {
      foreach (var xi in x) if (xi.Length != nRows) throw new ArgumentException("len(x_i) != nRows");
      if (x.Length > 0 && batchSize > x.First().Length) batchSize = x.First().Length;
      this.batchSize = batchSize;
      this.m = nRows;

      this.x = x;
      this.xBuf = x.Select(_ => new double[batchSize]).ToArray(); // buffer of batchSize for each variable
      this.thetaParam = expr.Parameters[0];
      this.xParam = expr.Parameters[1];
      // prepare a postfix representation of the expression
      exprArr = FlattenExpressionVisitor.Execute(expr.Body).ToArray();
      instrArr = new Instruction[exprArr.Length];
      for (int i = 0; i < exprArr.Length; i++) {
        var curExpr = exprArr[i];
        var curInstr = new Instruction() {
          opc = OpCode(curExpr)
        };

        // determine length
        var c = i - 1;
        int len = 1;
        for (int ch = 0; ch < NumChildren(exprArr[i]); ch++) {
          len += instrArr[c].length;
          c -= instrArr[c].length;
        }

        curInstr.length = len;
        switch (curInstr.opc) {
          case Instruction.OpcEnum.Var: {
              curInstr.idx = ExtractArrIndex(curExpr);
              curInstr.values = xBuf[curInstr.idx];
              curInstr.diffValues = new double[batchSize];
              break;
            }
          case Instruction.OpcEnum.Const: {
              var val = (double)((ConstantExpression)curExpr).Value;
              curInstr.values = new double[batchSize];
              curInstr.diffValues = new double[batchSize];
              for (int j = 0; j < batchSize; j++) curInstr.values[j] = val;
              break;
            }
          case Instruction.OpcEnum.Param: {
              curInstr.idx = ExtractArrIndex(curExpr);
              curInstr.values = new double[batchSize];
              curInstr.diffValues = new double[batchSize];
              break;
            }
          default: {
              curInstr.values = new double[batchSize];
              curInstr.diffValues = new double[batchSize];
              break;
            }
        }
        instrArr[i] = curInstr;
      }
    }

    // all rows
    public double[] Evaluate(double[] theta) {
      var remainderStart = (m / batchSize) * batchSize; // integer divison
      var f = new double[m];
      for (int startRow = 0; startRow < remainderStart; startRow += batchSize) {
        var fi = Evaluate(theta, startRow, batchSize);
        Array.Copy(fi, 0, f, startRow, batchSize);
      }

      // remainder
      if (m - remainderStart > 0) {
        var fi = Evaluate(theta, remainderStart, m - remainderStart);
        Array.Copy(fi, 0, f, remainderStart, m - remainderStart);
      }
      return f;
    }

    private double[] Evaluate(double[] theta, int startRow, int batchSize) {
      // copy variable values into batch buffer
      for (int i = 0; i < x.Length; i++) {
        Buffer.BlockCopy(x[i], startRow * sizeof(double), xBuf[i], 0, batchSize * sizeof(double));
      }

      for (int instrIdx = 0; instrIdx < instrArr.Length; instrIdx++) {
        var curInstr = instrArr[instrIdx];
        var curVal = curInstr.values;
        var ch0Idx = instrIdx - 1;
        var rightVal = ch0Idx >= 0 ? instrArr[ch0Idx].values : null;
        var ch1Idx = ch0Idx >= 0 ? ch0Idx - instrArr[ch0Idx].length : -1; // index of second child if the op has two children
        var leftVal = ch1Idx >= 0 ? instrArr[ch1Idx].values : null;
        switch (curInstr.opc) {
          case Instruction.OpcEnum.Var: /* nothing to do */ break;
          case Instruction.OpcEnum.Const: /* nothing to do */ break;
          case Instruction.OpcEnum.Param: for (int i = 0; i < batchSize; i++) { curVal[i] = theta[curInstr.idx]; } break;
          case Instruction.OpcEnum.Neg: for (int i = 0; i < batchSize; i++) { curVal[i] = -rightVal[i]; } break;
          case Instruction.OpcEnum.Add: for (int i = 0; i < batchSize; i++) { curVal[i] = leftVal[i] + rightVal[i]; } break;
          case Instruction.OpcEnum.Sub: for (int i = 0; i < batchSize; i++) { curVal[i] = leftVal[i] - rightVal[i]; } break;
          case Instruction.OpcEnum.Mul: for (int i = 0; i < batchSize; i++) { curVal[i] = leftVal[i] * rightVal[i]; } break;
          case Instruction.OpcEnum.Div: for (int i = 0; i < batchSize; i++) { curVal[i] = leftVal[i] / rightVal[i]; } break;

          case Instruction.OpcEnum.Log: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Log(rightVal[i]); } break;
          case Instruction.OpcEnum.Abs: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Abs(rightVal[i]); } break;
          case Instruction.OpcEnum.Exp: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Exp(rightVal[i]); } break;
          case Instruction.OpcEnum.Sin: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Sin(rightVal[i]); } break;
          case Instruction.OpcEnum.Cos: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Cos(rightVal[i]); } break;
          case Instruction.OpcEnum.Cosh: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Cosh(rightVal[i]); } break;
          case Instruction.OpcEnum.Tanh: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Tanh(rightVal[i]); } break;
          case Instruction.OpcEnum.Pow: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Pow(leftVal[i], rightVal[i]); } break;
          case Instruction.OpcEnum.Sqrt: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Sqrt(rightVal[i]); } break;
          case Instruction.OpcEnum.Cbrt: for (int i = 0; i < batchSize; i++) { curVal[i] = Functions.Cbrt(rightVal[i]); } break;
          case Instruction.OpcEnum.Sign: for (int i = 0; i < batchSize; i++) { curVal[i] = double.IsNaN(rightVal[i]) ? double.NaN : Math.Sign(rightVal[i]); } break;
          // case Instruction.OpcEnum.AQ: for (int i = 0; i < batchSize; i++) { curVal[i] = Functions.AQ(leftVal[i], rightVal[i]); } break;
          case Instruction.OpcEnum.Logistic: for (int i = 0; i < batchSize; i++) { curVal[i] = Functions.Logistic(rightVal[i]); } break;
          case Instruction.OpcEnum.InvLogistic: for (int i = 0; i < batchSize; i++) { curVal[i] = Functions.InvLogistic(rightVal[i]); } break;
          case Instruction.OpcEnum.LogisticPrime: for (int i = 0; i < batchSize; i++) { curVal[i] = Functions.LogisticPrime(rightVal[i]); } break;
          case Instruction.OpcEnum.InvLogisticPrime: for (int i = 0; i < batchSize; i++) { curVal[i] = Functions.InvLogisticPrime(rightVal[i]); } break;
          default: throw new InvalidOperationException();
        }
      }
      return instrArr.Last().values;
    }
    public double[] EvaluateWithJac(double[] theta, double[,] jacX, double[,] jacTheta) {
      var remainderStart = (m / batchSize) * batchSize; // integer divison
      var f = new double[m];
      for (int startRow = 0; startRow < remainderStart; startRow += batchSize) {
        var fi = EvaluateWithJac(theta, startRow, batchSize, jacX, jacTheta);
        Array.Copy(fi, 0, f, startRow, batchSize);
      }

      // remainder
      if (m - remainderStart > 0) {
        var fi = EvaluateWithJac(theta, remainderStart, m - remainderStart, jacX, jacTheta);
        Array.Copy(fi, 0, f, remainderStart, m - remainderStart);
      }
      return f;
    }

    private double[] EvaluateWithJac(double[] theta, int startRow, int batchSize, double[,] jacX, double[,] jacTheta) {
      // evaluate forward
      var f = (double[])Evaluate(theta, startRow, batchSize).Clone();

      if (jacX == null && jacTheta == null) return f; // backprop not necessary;

      // clear arrays
      if (jacX != null) Array.Clear(jacX, startRow * jacX.GetLength(1), batchSize * jacX.GetLength(1));
      if (jacTheta != null) Array.Clear(jacTheta, startRow * jacTheta.GetLength(1), batchSize * jacTheta.GetLength(1));

      // backpropagate
      var lastInstr = instrArr.Last();
      for (int i = 0; i < batchSize; i++) lastInstr.diffValues[i] = 1.0;

      for (int instrIdx = instrArr.Length - 1; instrIdx >= 0; instrIdx--) {
        var curInstr = instrArr[instrIdx];
        var curDiff = curInstr.diffValues;
        var curVal = curInstr.values;
        var ch0Idx = instrIdx - 1;
        var rightDiff = ch0Idx >= 0 ? instrArr[ch0Idx].diffValues : null;
        var rightVal = ch0Idx >= 0 ? instrArr[ch0Idx].values : null;
        var ch1Idx = ch0Idx >= 0 ? ch0Idx - instrArr[ch0Idx].length : -1; // index of second child if the op has two children
        var leftDiff = ch1Idx >= 0 ? instrArr[ch1Idx].diffValues : null;
        var leftVal = ch1Idx >= 0 ? instrArr[ch1Idx].values : null;
        switch (curInstr.opc) {
          case Instruction.OpcEnum.Var: if (jacX != null) for (int i = 0; i < batchSize; i++) { jacX[startRow + i, curInstr.idx] += curInstr.diffValues[i]; } break;
          case Instruction.OpcEnum.Const: /* nothing to do */ break;
          case Instruction.OpcEnum.Param: if (jacTheta != null) for (int i = 0; i < batchSize; i++) { jacTheta[startRow + i, curInstr.idx] += curInstr.diffValues[i]; } break;
          case Instruction.OpcEnum.Neg: for (int i = 0; i < batchSize; i++) { rightDiff[i] = -curDiff[i]; } break;
          case Instruction.OpcEnum.Add: for (int i = 0; i < batchSize; i++) { leftDiff[i] = curDiff[i]; rightDiff[i] = curDiff[i]; } break;
          case Instruction.OpcEnum.Sub: for (int i = 0; i < batchSize; i++) { leftDiff[i] = curDiff[i]; rightDiff[i] = -curDiff[i]; } break;
          case Instruction.OpcEnum.Mul:
            for (int i = 0; i < batchSize; i++) {
              leftDiff[i] = curDiff[i] * rightVal[i];
              rightDiff[i] = curDiff[i] * leftVal[i];
            }
            break;
          case Instruction.OpcEnum.Div:
            for (int i = 0; i < batchSize; i++) {
              leftDiff[i] = curDiff[i] / rightVal[i];
              rightDiff[i] = -curDiff[i] * leftVal[i] / (rightVal[i] * rightVal[i]);
            }
            break;

          case Instruction.OpcEnum.Log: for (int i = 0; i < batchSize; i++) { rightDiff[i] = curDiff[i] / rightVal[i]; } break;
          case Instruction.OpcEnum.Abs: for (int i = 0; i < batchSize; i++) { rightDiff[i] = curDiff[i] * (double.IsNaN(rightVal[i]) ? double.NaN : Math.Sign(rightVal[i])); } break;
          case Instruction.OpcEnum.Exp: for (int i = 0; i < batchSize; i++) { rightDiff[i] = curDiff[i] * curVal[i]; } break;
          case Instruction.OpcEnum.Sin: for (int i = 0; i < batchSize; i++) { rightDiff[i] = curDiff[i] * Math.Cos(rightVal[i]); } break;
          case Instruction.OpcEnum.Cos: for (int i = 0; i < batchSize; i++) { rightDiff[i] = curDiff[i] * -Math.Sin(rightVal[i]); } break;
          case Instruction.OpcEnum.Cosh: for (int i = 0; i < batchSize; i++) { rightDiff[i] = curDiff[i] * Math.Sinh(rightVal[i]); } break;
          case Instruction.OpcEnum.Tanh: for (int i = 0; i < batchSize; i++) { rightDiff[i] = curDiff[i] * 2.0 / (Math.Cosh(2.0 * rightVal[i]) + 1); } break;
          case Instruction.OpcEnum.Pow:
            for (int i = 0; i < batchSize; i++) {
              leftDiff[i] = curDiff[i] * rightVal[i] * curVal[i] / leftVal[i]; // curDiff[i] * rightVal[i] * Math.Pow(leftVal[i], rightVal[i] - 1);
              rightDiff[i] = curDiff[i] * curVal[i] * Math.Log(leftVal[i]);
            }
            break;
          case Instruction.OpcEnum.Sqrt: for (int i = 0; i < batchSize; i++) { rightDiff[i] = curDiff[i] * 0.5 / curVal[i]; } break;
          case Instruction.OpcEnum.Cbrt: for (int i = 0; i < batchSize; i++) { rightDiff[i] = curDiff[i] / (3.0 * curVal[i] * curVal[i]); } break;
          case Instruction.OpcEnum.Sign: for (int i = 0; i < batchSize; i++) { rightDiff[i] = 0.0; } break;
          // case Instruction.OpcEnum.AQ:
          //   for (int i = 0; i < batchSize; i++) {
          //     leftDiff[i] = curDiff[i] / rightVal[i];
          //     rightDiff[i] = -curDiff[i] * leftVal[i] / (rightVal[i] * rightVal[i]);
          //   }
          //   break;
          case Instruction.OpcEnum.Logistic: for (int i = 0; i < batchSize; i++) { rightDiff[i] = curDiff[i] * Functions.LogisticPrime(rightVal[i]); } break;
          case Instruction.OpcEnum.InvLogistic: for (int i = 0; i < batchSize; i++) { rightDiff[i] = curDiff[i] * Functions.InvLogisticPrime(rightVal[i]); } break;
          case Instruction.OpcEnum.LogisticPrime: for (int i = 0; i < batchSize; i++) { rightDiff[i] = curDiff[i] * Functions.LogisticPrimePrime(rightVal[i]); } break;
          case Instruction.OpcEnum.InvLogisticPrime: for (int i = 0; i < batchSize; i++) { rightDiff[i] = curDiff[i] * Functions.InvLogisticPrimePrime(rightVal[i]); } break;
          default: throw new InvalidOperationException();
        }
      }

      return f;
    }


    private int NumChildren(Expression expression) {
      if (expression is BinaryExpression binExpr) {
        // array index x[0], p[0] are leaf nodes for the interpreter
        if (binExpr.NodeType == ExpressionType.ArrayIndex) {
          return 0;
        } else
          return 2;
      } else if (expression is UnaryExpression) return 1;
      else if (expression is MethodCallExpression callExpr) return callExpr.Arguments.Count;
      else return 0;
    }

    private int ExtractArrIndex(Expression expression) {
      if (expression.NodeType != ExpressionType.ArrayIndex) throw new InvalidProgramException();
      var index = ((BinaryExpression)expression).Right;
      return (int)((ConstantExpression)index).Value; // only integer constants allowed
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
    // private static readonly MethodInfo aq = typeof(Functions).GetMethod("AQ", new[] { typeof(double), typeof(double) });
    private static readonly MethodInfo sign = typeof(Functions).GetMethod("Sign", new[] { typeof(double) }); // for deriv abs(x)
    private static readonly MethodInfo logistic = typeof(Functions).GetMethod("Logistic", new[] { typeof(double) });
    private static readonly MethodInfo invlogistic = typeof(Functions).GetMethod("InvLogistic", new[] { typeof(double) });
    private static readonly MethodInfo logisticPrime = typeof(Functions).GetMethod("LogisticPrime", new[] { typeof(double) }); // deriv of logistic
    // private static readonly MethodInfo logisticPrimePrime = typeof(Functions).GetMethod("LogisticPrimePrime", new[] { typeof(double) }); // deriv of logistic
    private static readonly MethodInfo invlogisticPrime = typeof(Functions).GetMethod("InvLogisticPrime", new[] { typeof(double) });
    // private static readonly MethodInfo invlogisticPrimePrime = typeof(Functions).GetMethod("InvLogisticPrimePrime", new[] { typeof(double) });

    private Instruction.OpcEnum OpCode(Expression expression) {
      switch (expression.NodeType) {
        case ExpressionType.ArrayIndex: {
            var binExpr = (BinaryExpression)expression;
            if (binExpr.Left == thetaParam) return Instruction.OpcEnum.Param;
            else if (binExpr.Left == xParam) return Instruction.OpcEnum.Var;
            else throw new InvalidProgramException();
          }
        case ExpressionType.Constant: { return Instruction.OpcEnum.Const; }
        case ExpressionType.Negate: return Instruction.OpcEnum.Neg;
        case ExpressionType.Add: return Instruction.OpcEnum.Add;
        case ExpressionType.Subtract: return Instruction.OpcEnum.Sub;
        case ExpressionType.Multiply: return Instruction.OpcEnum.Mul;
        case ExpressionType.Divide: return Instruction.OpcEnum.Div;
        case ExpressionType.Call: {
            var callExpr = (MethodCallExpression)expression;
            if (callExpr.Method == log) return Instruction.OpcEnum.Log;
            if (callExpr.Method == abs) return Instruction.OpcEnum.Abs;
            if (callExpr.Method == exp) return Instruction.OpcEnum.Exp;
            if (callExpr.Method == sin) return Instruction.OpcEnum.Sin;
            if (callExpr.Method == cos) return Instruction.OpcEnum.Cos;
            if (callExpr.Method == cosh) return Instruction.OpcEnum.Cosh;
            if (callExpr.Method == tanh) return Instruction.OpcEnum.Tanh;
            if (callExpr.Method == pow) return Instruction.OpcEnum.Pow;
            if (callExpr.Method == sqrt) return Instruction.OpcEnum.Sqrt;
            if (callExpr.Method == cbrt) return Instruction.OpcEnum.Cbrt;
            if (callExpr.Method == sign) return Instruction.OpcEnum.Sign;
            // if (callExpr.Method == aq) return Instruction.OpcEnum.AQ;
            if (callExpr.Method == logistic) return Instruction.OpcEnum.Logistic;
            if (callExpr.Method == invlogistic) return Instruction.OpcEnum.InvLogistic;
            if (callExpr.Method == logisticPrime) return Instruction.OpcEnum.LogisticPrime;
            if (callExpr.Method == invlogisticPrime) return Instruction.OpcEnum.InvLogisticPrime;
            else throw new InvalidProgramException();
          }
        default: throw new InvalidProgramException();
      }
    }

    private struct Instruction {
      public enum OpcEnum { None, Const, Param, Var, Neg, Add, Sub, Mul, Div, Log, Abs, Exp, Sin, Cos, Cosh, Tanh, Pow, Sqrt, Cbrt, Sign, Logistic, InvLogistic, LogisticPrime, InvLogisticPrime };
      public int length;
      public OpcEnum opc;
      public double[] values;
      public double[] diffValues; // for reverse autodiff
      public int idx; // only for parameters and variables

    }
  }
}

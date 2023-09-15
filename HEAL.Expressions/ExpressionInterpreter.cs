using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;


namespace HEAL.Expressions {
  // TODO: length of code does not have to match size of expression tree. (allow multiple instructions for a single tree node e.g. for powabs or logabs)

  // prepares data structures for repeated efficient evaluation of a single expression
  public class ExpressionInterpreter {
    private readonly int batchSize;
    private readonly int m;
    private readonly double[][] x;
    private readonly double[][] xBuf;
    private readonly ParameterExpression thetaParam;
    private readonly ParameterExpression xParam;
    private readonly List<Instruction> instuctions;
    private readonly Dictionary<string, int> expr2tapeIdx = new Dictionary<string, int>();

    // x is column oriented
    public ExpressionInterpreter(Expression<Expr.ParametricFunction> expression, double[][] x, int nRows, int batchSize = 256) {
      foreach (var xi in x) if (xi.Length != nRows) throw new ArgumentException("len(x_i) != nRows");
      if (x.Length > 0 && batchSize > x.First().Length) batchSize = x.First().Length;
      this.batchSize = batchSize;
      this.m = nRows;
      this.x = x;
      this.xBuf = x.Select(_ => new double[batchSize]).ToArray(); // buffer of batchSize for each variable
      this.thetaParam = expression.Parameters[0];
      this.xParam = expression.Parameters[1];
      // prepare a postfix representation of the expression
      instuctions = new List<Instruction>();
      foreach (var curExpr in FlattenExpressionVisitor.Execute(expression.Body)) {
        var exprStr = curExpr.ToString();
        if (expr2tapeIdx.ContainsKey(exprStr)) continue;

        var curInstr = new Instruction() {
          opc = OpCode(curExpr)
        };

        switch (curExpr) {
          case BinaryExpression binary:
            // operators +, -, *, /,
            if (binary.NodeType != ExpressionType.ArrayIndex) {
              curInstr.values = new double[batchSize];
              curInstr.diffValues = new double[batchSize];

              curInstr.idx1 = expr2tapeIdx[binary.Left.ToString()];
              curInstr.idx2 = expr2tapeIdx[binary.Right.ToString()];
            } else {
              // array access x[] or p[]
              if (binary.Left == thetaParam) {
                curInstr.idx1 = ExtractArrIndex(curExpr);
                curInstr.values = new double[1]; // the value is constant
                curInstr.diffValues = new double[batchSize];
              } else if (binary.Left == xParam) {
                curInstr.idx1 = ExtractArrIndex(curExpr);
                curInstr.values = xBuf[curInstr.idx1];
                curInstr.diffValues = new double[batchSize];
              } else throw new NotSupportedException("unknown array variable.");
            }
            break;
          case UnaryExpression unary:
            if (unary.NodeType == ExpressionType.UnaryPlus) continue; // this can be ignored completely (and should never occur anyway)

            curInstr.values = new double[batchSize];
            curInstr.diffValues = new double[batchSize];
            curInstr.idx1 = expr2tapeIdx[unary.Operand.ToString()];
            break;
          case MethodCallExpression call:
            curInstr.values = new double[batchSize];
            curInstr.diffValues = new double[batchSize];

            // only unary or binary
            curInstr.idx1 = expr2tapeIdx[call.Arguments[0].ToString()];
            if (call.Arguments.Count == 2) {
              curInstr.idx2 = expr2tapeIdx[call.Arguments[1].ToString()];
            } else if (call.Arguments.Count > 2)
              throw new NotSupportedException("Method call with more than two arguments");
            break;
          case ConstantExpression constExpr:
            curInstr.values = new[] { (double)((ConstantExpression)curExpr).Value };
            break;
        }

        expr2tapeIdx.Add(exprStr, instuctions.Count);
        instuctions.Add(curInstr);
      }
    }

    // all rows
    public void Evaluate(double[] theta, double[] f) {
      var remainderStart = (m / batchSize) * batchSize; // integer divison
      for (int startRow = 0; startRow < remainderStart; startRow += batchSize) {
        Evaluate(theta, f, startRow, batchSize);
      }

      // remainder
      if (m - remainderStart > 0) {
        Evaluate(theta, f, remainderStart, m - remainderStart);
      }
    }

    private void Evaluate(double[] theta, double[] f, int startRow, int batchSize) {
      // copy variable values into batch buffer
      for (int i = 0; i < x.Length; i++) {
        Buffer.BlockCopy(x[i], startRow * sizeof(double), xBuf[i], 0, batchSize * sizeof(double));
      }

      for (int instrIdx = 0; instrIdx < instuctions.Count; instrIdx++) {
        var curInstr = instuctions[instrIdx];
        var curVal = curInstr.values;
        Instruction left = default, right = default;
        if (curInstr.opc != Instruction.OpcEnum.Var && curInstr.opc != Instruction.OpcEnum.Param) {
          left = instuctions[curInstr.idx1];
          right = instuctions[curInstr.idx2];
        }
        switch (curInstr.opc) {
          case Instruction.OpcEnum.Var: /* nothing to do */ break;
          case Instruction.OpcEnum.Const: /* nothing to do */ break;
          case Instruction.OpcEnum.Param: curVal[0] = theta[curInstr.idx1]; break;
          case Instruction.OpcEnum.Neg: for (int i = 0; i < batchSize; i++) { curVal[i] = -left.GetValue(i); } break;
          case Instruction.OpcEnum.Add: for (int i = 0; i < batchSize; i++) { curVal[i] = left.GetValue(i) + right.GetValue(i); } break;
          case Instruction.OpcEnum.Sub: for (int i = 0; i < batchSize; i++) { curVal[i] = left.GetValue(i) - right.GetValue(i); } break;
          case Instruction.OpcEnum.Mul: for (int i = 0; i < batchSize; i++) { curVal[i] = left.GetValue(i) * right.GetValue(i); } break;
          case Instruction.OpcEnum.Div: for (int i = 0; i < batchSize; i++) { curVal[i] = left.GetValue(i) / right.GetValue(i); } break;

          case Instruction.OpcEnum.Log: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Log(left.GetValue(i)); } break;
          case Instruction.OpcEnum.Abs: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Abs(left.GetValue(i)); } break;
          case Instruction.OpcEnum.Exp: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Exp(left.GetValue(i)); } break;
          case Instruction.OpcEnum.Sin: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Sin(left.GetValue(i)); } break;
          case Instruction.OpcEnum.Cos: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Cos(left.GetValue(i)); } break;
          case Instruction.OpcEnum.Cosh: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Cosh(left.GetValue(i)); } break;
          case Instruction.OpcEnum.Tanh: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Tanh(left.GetValue(i)); } break;
          case Instruction.OpcEnum.Pow: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Pow(left.GetValue(i), right.GetValue(i)); } break;
          case Instruction.OpcEnum.PowAbs: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Pow(Math.Abs(left.GetValue(i)), right.GetValue(i)); } break;
          case Instruction.OpcEnum.Sqrt: for (int i = 0; i < batchSize; i++) { curVal[i] = Math.Sqrt(left.GetValue(i)); } break;
          case Instruction.OpcEnum.Cbrt: for (int i = 0; i < batchSize; i++) { curVal[i] = Functions.Cbrt(left.GetValue(i)); } break;
          case Instruction.OpcEnum.Sign: for (int i = 0; i < batchSize; i++) { curVal[i] = double.IsNaN(left.GetValue(i)) ? double.NaN : Math.Sign(left.GetValue(i)); } break;
          // case Instruction.OpcEnum.AQ: for (int i = 0; i < batchSize; i++) { curVal[i] = Functions.AQ(left.GetValue(i), right.GetValue(i)); } break;
          case Instruction.OpcEnum.Logistic: for (int i = 0; i < batchSize; i++) { curVal[i] = Functions.Logistic(left.GetValue(i)); } break;
          case Instruction.OpcEnum.InvLogistic: for (int i = 0; i < batchSize; i++) { curVal[i] = Functions.InvLogistic(left.GetValue(i)); } break;
          case Instruction.OpcEnum.LogisticPrime: for (int i = 0; i < batchSize; i++) { curVal[i] = Functions.LogisticPrime(left.GetValue(i)); } break;
          case Instruction.OpcEnum.InvLogisticPrime: for (int i = 0; i < batchSize; i++) { curVal[i] = Functions.InvLogisticPrime(left.GetValue(i)); } break;
          default: throw new InvalidOperationException();
        }
      }

      var result = instuctions.Last();

      if (result.values.Length == 1) {
        // parameters or constants
        Array.Fill(f, result.GetValue(0), startRow, batchSize);
      } else {
        Array.Copy(result.values, 0, f, startRow, batchSize);
      }
    }
    public double[] EvaluateWithJac(double[] theta, double[] f, double[,] jacX, double[,] jacTheta) {
      var remainderStart = (m / batchSize) * batchSize; // integer divison
      for (int startRow = 0; startRow < remainderStart; startRow += batchSize) {
        EvaluateWithJac(theta, f, startRow, batchSize, jacX, jacTheta);
      }

      // remainder
      if (m - remainderStart > 0) {
        EvaluateWithJac(theta, f, remainderStart, m - remainderStart, jacX, jacTheta);
      }
      return f;
    }

    private void EvaluateWithJac(double[] theta, double[] f, int startRow, int batchSize, double[,] jacX, double[,] jacTheta) {
      // evaluate forward
      Evaluate(theta, f, startRow, batchSize);

      if (jacX == null && jacTheta == null) return; // backprop not necessary;

      // clear arrays
      if (jacX != null) Array.Clear(jacX, startRow * jacX.GetLength(1), batchSize * jacX.GetLength(1));
      if (jacTheta != null) Array.Clear(jacTheta, startRow * jacTheta.GetLength(1), batchSize * jacTheta.GetLength(1));
      for (int i = 0; i < instuctions.Count; i++) if (instuctions[i].diffValues != null) Array.Clear(instuctions[i].diffValues);


      // backpropagate
      var lastInstr = instuctions.Last();
      if (lastInstr.diffValues != null) Array.Fill(lastInstr.diffValues, 1.0);

      for (int instrIdx = instuctions.Count - 1; instrIdx >= 0; instrIdx--) {
        var curInstr = instuctions[instrIdx];
        var curDiff = curInstr.diffValues;

        double[] leftDiff = null, rightDiff = null;
        Instruction left = default, right = default;
        if (curInstr.opc != Instruction.OpcEnum.Var && curInstr.opc != Instruction.OpcEnum.Param) {
          var ch0Idx = curInstr.idx1;
          leftDiff = ch0Idx >= 0 ? instuctions[ch0Idx].diffValues : null;
          left = ch0Idx >= 0 ? instuctions[ch0Idx] : default;

          var ch1Idx = curInstr.idx2;
          rightDiff = ch1Idx >= 0 ? instuctions[ch1Idx].diffValues : null;
          right = ch1Idx >= 0 ? instuctions[ch1Idx] : default;
        }
        switch (curInstr.opc) {
          case Instruction.OpcEnum.Var: if (jacX != null) for (int i = 0; i < batchSize; i++) { jacX[startRow + i, curInstr.idx1] += curDiff[i]; } break;
          case Instruction.OpcEnum.Const: /* nothing to do */ break;
          case Instruction.OpcEnum.Param: if (jacTheta != null) for (int i = 0; i < batchSize; i++) { jacTheta[startRow + i, curInstr.idx1] += curDiff[i]; } break;
          case Instruction.OpcEnum.Neg:
            if (leftDiff != null) for (int i = 0; i < batchSize; i++) { leftDiff[i] -= curDiff[i]; }
            break;
          case Instruction.OpcEnum.Add:
            if (leftDiff != null) for (int i = 0; i < batchSize; i++) { leftDiff[i] += curDiff[i]; }
            if (rightDiff != null) for (int i = 0; i < batchSize; i++) { rightDiff[i] += curDiff[i]; }
            break;
          case Instruction.OpcEnum.Sub:
            if (leftDiff != null) for (int i = 0; i < batchSize; i++) { leftDiff[i] += curDiff[i]; }
            if (rightDiff != null) for (int i = 0; i < batchSize; i++) { rightDiff[i] -= curDiff[i]; }
            break;
          case Instruction.OpcEnum.Mul:
            if (leftDiff != null) for (int i = 0; i < batchSize; i++) { leftDiff[i] += curDiff[i] * right.GetValue(i); }
            if (rightDiff != null) for (int i = 0; i < batchSize; i++) { rightDiff[i] += curDiff[i] * left.GetValue(i); }
            break;
          case Instruction.OpcEnum.Div:
            if (leftDiff != null) for (int i = 0; i < batchSize; i++) { leftDiff[i] += curDiff[i] / right.GetValue(i); }
            if (rightDiff != null) for (int i = 0; i < batchSize; i++) { rightDiff[i] += -curDiff[i] * left.GetValue(i) / (right.GetValue(i) * right.GetValue(i)); }
            break;

          // TODO: for unary operations we can re-use the curDiff array
          case Instruction.OpcEnum.Log:
            if (leftDiff != null) for (int i = 0; i < batchSize; i++) { leftDiff[i] += curDiff[i] / left.GetValue(i); }
            break;
          case Instruction.OpcEnum.Abs:
            if (leftDiff != null) for (int i = 0; i < batchSize; i++) { leftDiff[i] += curDiff[i] * (double.IsNaN(left.GetValue(i)) ? double.NaN : Math.Sign(left.GetValue(i))); }
            break;
          case Instruction.OpcEnum.Exp:
            if (leftDiff != null) for (int i = 0; i < batchSize; i++) { leftDiff[i] += curDiff[i] * curInstr.GetValue(i); }
            break;
          case Instruction.OpcEnum.Sin: if (leftDiff != null) for (int i = 0; i < batchSize; i++) { leftDiff[i] += curDiff[i] * Math.Cos(left.GetValue(i)); } break;
          case Instruction.OpcEnum.Cos: if (leftDiff != null) for (int i = 0; i < batchSize; i++) { leftDiff[i] += curDiff[i] * -Math.Sin(left.GetValue(i)); } break;
          case Instruction.OpcEnum.Cosh: if (leftDiff != null) for (int i = 0; i < batchSize; i++) { leftDiff[i] += curDiff[i] * Math.Sinh(left.GetValue(i)); } break;
          case Instruction.OpcEnum.Tanh: if (leftDiff != null) for (int i = 0; i < batchSize; i++) { leftDiff[i] += curDiff[i] * 2.0 / (Math.Cosh(2.0 * left.GetValue(i)) + 1); } break;
          case Instruction.OpcEnum.Pow:
            if (leftDiff != null)
              for (int i = 0; i < batchSize; i++) {
                leftDiff[i] += curDiff[i] * right.GetValue(i) * curInstr.GetValue(i) / left.GetValue(i); // curDiff[i] * right.GetValue(i) * Math.Pow(left.GetValue(i), right.GetValue(i) - 1);
              }
            if (rightDiff != null)
              for (int i = 0; i < batchSize; i++) {
                rightDiff[i] += curDiff[i] * curInstr.GetValue(i) * Math.Log(left.GetValue(i));
              }
            break;
          case Instruction.OpcEnum.PowAbs:
            if (leftDiff != null)
              for (int i = 0; i < batchSize; i++) {
                leftDiff[i] += curDiff[i] * right.GetValue(i) * left.GetValue(i) * Math.Pow(Math.Abs(left.GetValue(i)), right.GetValue(i) - 2);  //f'(x) * f(x) * g(y) * abs(f(x))^(g(y) - 2)
              }
            if (rightDiff != null)
              for (int i = 0; i < batchSize; i++) {
                rightDiff[i] += curDiff[i] * curInstr.GetValue(i) * Math.Log(Math.Abs(left.GetValue(i))); // check
              }
            break;
          case Instruction.OpcEnum.Sqrt:
            if (leftDiff != null) for (int i = 0; i < batchSize; i++) { leftDiff[i] += curDiff[i] * 0.5 / curInstr.GetValue(i); }
            break;
          case Instruction.OpcEnum.Cbrt:
            if (leftDiff != null) for (int i = 0; i < batchSize; i++) { leftDiff[i] += curDiff[i] / (3.0 * curInstr.GetValue(i) * curInstr.GetValue(i)); }
            break;
          case Instruction.OpcEnum.Sign: /* nothing to do */ break;
          // case Instruction.OpcEnum.AQ:
          //   for (int i = 0; i < batchSize; i++) {
          //     leftDiff[i] = curDiff[i] / right.GetValue(i);
          //     rightDiff[i] = -curDiff[i] * left.GetValue(i) / (right.GetValue(i) * right.GetValue(i));
          //   }
          //   break;
          case Instruction.OpcEnum.Logistic: if (leftDiff != null) for (int i = 0; i < batchSize; i++) { leftDiff[i] += curDiff[i] * Functions.LogisticPrime(left.GetValue(i)); } break;
          case Instruction.OpcEnum.InvLogistic: if (leftDiff != null) for (int i = 0; i < batchSize; i++) { leftDiff[i] += curDiff[i] * Functions.InvLogisticPrime(left.GetValue(i)); } break;
          case Instruction.OpcEnum.LogisticPrime: if (leftDiff != null) for (int i = 0; i < batchSize; i++) { leftDiff[i] += curDiff[i] * Functions.LogisticPrimePrime(left.GetValue(i)); } break;
          case Instruction.OpcEnum.InvLogisticPrime: if (leftDiff != null) for (int i = 0; i < batchSize; i++) { leftDiff[i] += curDiff[i] * Functions.InvLogisticPrimePrime(left.GetValue(i)); } break;
          default: throw new InvalidOperationException();
        }
      }
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
    private static readonly MethodInfo powabs = typeof(Functions).GetMethod("PowAbs", new[] { typeof(double), typeof(double) });
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
            if (callExpr.Method == powabs) return Instruction.OpcEnum.PowAbs;
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
      public enum OpcEnum { None, Const, Param, Var, Neg, Add, Sub, Mul, Div, Log, Abs, Exp, Sin, Cos, Cosh, Tanh, Pow, PowAbs, Sqrt, Cbrt, Sign, Logistic, InvLogistic, LogisticPrime, InvLogisticPrime };

      public int idx1; // child idx1 for internal nodes, index into p or x for parameters or variables
      public int idx2; // child idx2 for internal nodes (only for binary operations)
      public OpcEnum opc;
      public double[] values; // for internal nodes and variables
      public double[] diffValues; // for reverse autodiff

      public double GetValue(int idx) => values.Length == 1 ? values[0] : values[idx];
    }
  }
}

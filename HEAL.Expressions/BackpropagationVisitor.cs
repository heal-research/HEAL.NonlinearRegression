using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using System.Reflection;

namespace HEAL.Expressions {

  // reverse mode autodiff (in combination with EvaluationVisitor)
  public class BackpropagationVisitor : ExpressionVisitor {
    private readonly ParameterExpression param;
    private readonly int batchSize;
    private readonly double[,] jac;
    private readonly Dictionary<Expression, double[]> nodeValues; // forward evaluation result for each node

    internal BackpropagationVisitor(ParameterExpression param, Dictionary<Expression, double[]> nodeValues, double[,] jac) {
      this.param = param;
      this.batchSize = jac.GetLength(0);
      this.nodeValues = nodeValues;
      this.jac = jac;
      Array.Clear(jac, 0, jac.Length);
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      var res = nodeValues[node];
      var opRes = nodeValues[node.Operand];

      if (node.NodeType == ExpressionType.Negate) {
        for (int i = 0; i < batchSize; i++) opRes[i] = -1.0 * res[i];
      } else if (node.NodeType == ExpressionType.UnaryPlus) {
        for (int i = 0; i < batchSize; i++) opRes[i] = res[i];
      } else throw new NotSupportedException("Unknown operation");

      Visit(node.Operand);
      return node;
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      var res = nodeValues[node];

      if (node.NodeType == ExpressionType.ArrayIndex) {
        var index = node.Right;
        if (index.NodeType != ExpressionType.Constant) throw new NotSupportedException("only constant indices for parameter are allowed");

        var idx = (int)((ConstantExpression)index).Value;
        if (node.Left == param) {
          for (int i = 0; i < batchSize; i++) { jac[i, idx] += res[i]; } // parameter (add in case parameters would appear multiple times)
        } else {
          // nothing to do (until we want to calculate the gradient for variables)
        }
        return node;
      } else {
        var left = nodeValues[node.Left];
        var right = nodeValues[node.Right];

        switch (node.NodeType) {
          case ExpressionType.Add: { for (int i = 0; i < batchSize; i++) { left[i] = res[i]; right[i] = res[i]; } break; }
          case ExpressionType.Subtract: { for (int i = 0; i < batchSize; i++) { left[i] = res[i]; right[i] = -res[i]; } break; }
          case ExpressionType.Multiply: {
              var newLeft = new double[batchSize];
              var newRight = new double[batchSize];
              for (int i = 0; i < batchSize; i++) {
                newLeft[i] = res[i] * right[i];
                newRight[i] = res[i] * left[i];
              }
              nodeValues[node.Left] = newLeft;
              nodeValues[node.Right] = newRight;
              break;
            }
          case ExpressionType.Divide: {
              var newLeft = new double[batchSize];
              var newRight = new double[batchSize];
              for (int i = 0; i < batchSize; i++) {
                newLeft[i] = res[i] / right[i];
                newRight[i] = -res[i] * left[i] / (right[i] * right[i]);
              }
              nodeValues[node.Left] = newLeft;
              nodeValues[node.Right] = newRight;
              break;
            }
          default: throw new NotSupportedException(node.ToString());
        }
        Visit(node.Left); Visit(node.Right);
        return node;

      }
    }

    private readonly MethodInfo abs = typeof(Math).GetMethod("Abs", new[] { typeof(double) });
    private readonly MethodInfo sin = typeof(Math).GetMethod("Sin", new[] { typeof(double) });
    private readonly MethodInfo cos = typeof(Math).GetMethod("Cos", new[] { typeof(double) });
    private readonly MethodInfo exp = typeof(Math).GetMethod("Exp", new[] { typeof(double) });
    private readonly MethodInfo log = typeof(Math).GetMethod("Log", new[] { typeof(double) });
    private readonly MethodInfo tanh = typeof(Math).GetMethod("Tanh", new[] { typeof(double) });
    private readonly MethodInfo cosh = typeof(Math).GetMethod("Cosh", new[] { typeof(double) });
    private readonly MethodInfo sqrt = typeof(Math).GetMethod("Sqrt", new[] { typeof(double) });
    private readonly MethodInfo cbrt = typeof(Functions).GetMethod("Cbrt", new[] { typeof(double) });
    private readonly MethodInfo pow = typeof(Math).GetMethod("Pow", new[] { typeof(double), typeof(double) });
    private readonly MethodInfo sign = typeof(Math).GetMethod("Sign", new[] { typeof(double) }); // for deriv abs(x)
    private readonly MethodInfo logistic = typeof(Functions).GetMethod("Logistic", new[] { typeof(double) });
    private readonly MethodInfo invlogistic = typeof(Functions).GetMethod("InvLogistic", new[] { typeof(double) });
    private readonly MethodInfo logisticPrime = typeof(Functions).GetMethod("LogisticPrime", new[] { typeof(double) }); // deriv of logistic
    private readonly MethodInfo logisticPrimePrime = typeof(Functions).GetMethod("LogisticPrimePrime", new[] { typeof(double) }); // deriv of logistic
    private readonly MethodInfo invlogisticPrime = typeof(Functions).GetMethod("InvLogisticPrime", new[] { typeof(double) });
    private readonly MethodInfo invlogisticPrimePrime = typeof(Functions).GetMethod("InvLogisticPrimePrime", new[] { typeof(double) });

    protected override Expression VisitMethodCall(MethodCallExpression node) {
      var res = nodeValues[node];
      var a1 = nodeValues[node.Arguments[0]];
      if (node.Method == sin) {
        for (int i = 0; i < batchSize; i++) { a1[i] = res[i] * Math.Cos(a1[i]); }
      } else if (node.Method == cos) {
        for (int i = 0; i < batchSize; i++) { a1[i] = res[i] * -Math.Sin(a1[i]); }
        //dfx = Expression.Negate(Expression.Call(sin, x));
      } else if (node.Method == exp) {
        for (int i = 0; i < batchSize; i++) { a1[i] = res[i] * Math.Exp(a1[i]); }
        //dfx = node;
      } else if (node.Method == log) {
        for (int i = 0; i < batchSize; i++) { a1[i] = res[i] / a1[i]; }
        //dfx = Expression.Divide(Expression.Constant(1.0), x);
      } else if (node.Method == tanh) {
        for (int i = 0; i < batchSize; i++) { a1[i] = res[i] * 2.0 / (Math.Cosh(2.0 * a1[i]) + 1); }
        //dfx = Expression.Divide(
        //  Expression.Constant(2.0),
        //  Expression.Add(
        //    Expression.Call(cosh,
        //      Expression.Multiply(Expression.Constant(2.0), x)),
        //    Expression.Constant(1.0)));
      } else if (node.Method == sqrt) {
        for (int i = 0; i < batchSize; i++) { a1[i] = res[i] * 0.5  / Math.Sqrt(a1[i]); }
        //dfx = Expression.Multiply(Expression.Constant(0.5), Expression.Divide(Expression.Constant(1.0), node));
      } else if (node.Method == cbrt) {
        for (int i = 0; i < batchSize; i++) { a1[i] = res[i] / 3.0 / Math.Pow(Math.Cbrt(a1[i]), 2); }
        // 1/3 * 1/cbrt(...)^2
        //dfx = Expression.Divide(Expression.Constant(1.0 / 3.0),
        //  Expression.Call(pow, node, Expression.Constant(2.0)));
      } else if (node.Method == pow) {
        var exponent = node.Arguments[1];
        var a2 = nodeValues[exponent];
        var newA1 = new double[a1.Length];
        for (int i = 0; i < batchSize; i++) { newA1[i] = res[i] * a2[i] * Math.Pow(a1[i], a2[i] - 1); }
        for (int i = 0; i < batchSize; i++) { a2[i] = res[i] * res[i] * Math.Log(a1[i]); }
        nodeValues[node.Arguments[0]] = newA1;
        Visit(exponent);
      } else if (node.Method == abs) {
        for (int i = 0; i < batchSize; i++) { a1[i] = res[i] * Math.Sign(a1[i]); }
        // dfx = Expression.Multiply(Expression.Call(sign, x), Expression.Constant(1.0)); // int -> double
      } else if (node.Method == logistic) {
        for (int i = 0; i < batchSize; i++) { a1[i] = res[i] * Functions.LogisticPrime(a1[i]); }
        // dfx = Expression.Call(logisticPrime, x);
      } else if (node.Method == invlogistic) {
        for (int i = 0; i < batchSize; i++) { a1[i] = res[i] * Functions.InvLogisticPrime(a1[i]); }
        // dfx = Expression.Call(invlogisticPrime, x);
      } else if (node.Method == logisticPrime) {
        for (int i = 0; i < batchSize; i++) { a1[i] = res[i] * Functions.LogisticPrimePrime(a1[i]); }
        // dfx = Expression.Call(logisticPrimePrime, x);
      } else if (node.Method == invlogisticPrime) {
        for (int i = 0; i < batchSize; i++) { a1[i] = res[i] * Functions.InvLogisticPrimePrime(a1[i]); }
        // dfx = Expression.Call(invlogisticPrimePrime, x);
      } else throw new NotSupportedException($"Unsupported method call {node.Method.Name}");

      Visit(node.Arguments[0]); // backpropagate
      return node;
    }
  }
}

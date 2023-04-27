using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using System.Reflection;

namespace HEAL.Expressions {
  public class EvaluationVisitor : ExpressionVisitor {
    private readonly ParameterExpression param;
    private readonly int batchSize;
    private readonly double[,] x;
    private readonly double[] theta;
    private readonly Dictionary<Expression, double[]> evalResult; // forward evaluation result for each node
    public Dictionary<Expression, double[]> NodeValues => evalResult;

    internal EvaluationVisitor(ParameterExpression param, double[] theta, double[,] x) {
      this.param = param;
      this.batchSize = x.GetLength(0);
      this.x = x;
      this.theta = theta;
      this.evalResult = new Dictionary<Expression, double[]>();
    }

    protected override Expression VisitConstant(ConstantExpression node) {
      var res = new double[batchSize];
      evalResult[node] = res;

      var val = (double)node.Value;
      for (int i = 0; i < batchSize; i++) res[i] = val;
      return node;
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      var res = new double[batchSize];
      evalResult[node] = res;

      Visit(node.Operand);
      if (node.NodeType == ExpressionType.Negate) {
        for (int i = 0; i < batchSize; i++) res[i] = -1.0;
      } else if (node.NodeType == ExpressionType.UnaryPlus) {
        for (int i = 0; i < batchSize; i++) res[i] = 1.0;
      } else throw new NotSupportedException("Unknown operation");
      return node;
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      var res = new double[batchSize];
      evalResult[node] = res;

      if (node.NodeType == ExpressionType.ArrayIndex) {
        var index = node.Right;
        if (index.NodeType != ExpressionType.Constant) throw new NotSupportedException("only constant indices for parameter are allowed");

        var idx = (int)((ConstantExpression)index).Value;
        if (node.Left == param) {
          for (int i = 0; i < batchSize; i++) { res[i] = theta[idx]; } // parameter
        } else {
          for (int i = 0; i < batchSize; i++) { res[i] = x[i, idx]; } // variable
        }
        return node;
      } else {
        Visit(node.Left); Visit(node.Right);
        var left = evalResult[node.Left];
        var right = evalResult[node.Right];

        switch (node.NodeType) {
          case ExpressionType.Add: { for (int i = 0; i < batchSize; i++) { res[i] = left[i] + right[i]; } return node; }
          case ExpressionType.Subtract: { for (int i = 0; i < batchSize; i++) { res[i] = left[i] - right[i]; } return node; }
          case ExpressionType.Multiply: { for (int i = 0; i < batchSize; i++) { res[i] = left[i] * right[i]; } return node; }
          case ExpressionType.Divide: { for (int i = 0; i < batchSize; i++) { res[i] = left[i] / right[i]; } return node; }
          default: throw new NotSupportedException(node.ToString());
        }
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
      var res = new double[batchSize];
      evalResult[node] = res;

      Visit(node.Arguments[0]);
      var a1 = evalResult[node.Arguments[0]];
      if (node.Method == sin) {
        for (int i = 0; i < batchSize; i++) { res[i] = Math.Sin(a1[i]); }
        //dfx = Expression.Call(cos, x);
      } else if (node.Method == cos) {
        for (int i = 0; i < batchSize; i++) { res[i] = Math.Cos(a1[i]); }
        //dfx = Expression.Negate(Expression.Call(sin, x));
      } else if (node.Method == exp) {
        for (int i = 0; i < batchSize; i++) { res[i] = Math.Exp(a1[i]); }
        //dfx = node;
      } else if (node.Method == log) {
        for (int i = 0; i < batchSize; i++) { res[i] = Math.Log(a1[i]); }
        //dfx = Expression.Divide(Expression.Constant(1.0), x);
      } else if (node.Method == tanh) {
        for (int i = 0; i < batchSize; i++) { res[i] = Math.Tanh(a1[i]); }
        //dfx = Expression.Divide(
        //  Expression.Constant(2.0),
        //  Expression.Add(
        //    Expression.Call(cosh,
        //      Expression.Multiply(Expression.Constant(2.0), x)),
        //    Expression.Constant(1.0)));
      } else if (node.Method == sqrt) {
        for (int i = 0; i < batchSize; i++) { res[i] = Math.Sqrt(a1[i]); }
        //dfx = Expression.Multiply(Expression.Constant(0.5), Expression.Divide(Expression.Constant(1.0), node));
      } else if (node.Method == cbrt) {
        for (int i = 0; i < batchSize; i++) { res[i] = Math.Cbrt(a1[i]); }
        // 1/3 * 1/cbrt(...)^2
        //dfx = Expression.Divide(Expression.Constant(1.0 / 3.0),
        //  Expression.Call(pow, node, Expression.Constant(2.0)));
      } else if (node.Method == pow) {
        Visit(node.Arguments[1]);
        var a2 = evalResult[node.Arguments[1]];
        for (int i = 0; i < batchSize; i++) { res[i] = Math.Pow(a1[i], a2[i]); }
        // var exponent = node.Arguments[1];
        // if (exponent.NodeType == ExpressionType.Constant) {
        //   var expVal = (double)((ConstantExpression)exponent).Value;
        //   dfx = Expression.Multiply(exponent, Expression.Call(pow, b, Expression.Constant(expVal - 1)));
        // } else if (exponent is BinaryExpression binaryExpression && binaryExpression.Left == param) {
        //   return Expression.Multiply(node, Expression.Add(Expression.Divide(Expression.Multiply(exponent, dx), b), Expression.Call(log, b)));
        // } else {
        //   throw new NotSupportedException("Exponents can only be parameters or constants.");
        // }

      } else if (node.Method == abs) {
        for (int i = 0; i < batchSize; i++) { res[i] = Math.Abs(a1[i]); }
        // dfx = Expression.Multiply(Expression.Call(sign, x), Expression.Constant(1.0)); // int -> double
      } else if (node.Method == logistic) {
        for (int i = 0; i < batchSize; i++) { res[i] = Functions.Logistic(a1[i]); }
        // dfx = Expression.Call(logisticPrime, x);
      } else if (node.Method == invlogistic) {
        for (int i = 0; i < batchSize; i++) { res[i] = Functions.InvLogistic(a1[i]); }
        // dfx = Expression.Call(invlogisticPrime, x);
      } else if (node.Method == logisticPrime) {
        for (int i = 0; i < batchSize; i++) { res[i] = Functions.LogisticPrime(a1[i]); }
        // dfx = Expression.Call(logisticPrimePrime, x);
      } else if (node.Method == invlogisticPrime) {
        for (int i = 0; i < batchSize; i++) { res[i] = Functions.InvLogisticPrime(a1[i]); }
        // dfx = Expression.Call(invlogisticPrimePrime, x);
      } else throw new NotSupportedException($"Unsupported method call {node.Method.Name}");

      return node;
      // return Expression.Multiply(dfx, dx);
    }
  }
}

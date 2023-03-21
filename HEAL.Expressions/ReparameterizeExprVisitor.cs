using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;

namespace HEAL.Expressions {
  // used by t-profile code for prediction intervals
  // takes an expression y = f(x, theta)
  // and returns expression for reparameterized f 
  // theta_i = f'(x,theta) = f( g(x,theta) + invF(theta_i) - g(x0, theta) )

  // Returns the expression and the index of the parameter theta_i
  internal class ReparameterizeExprVisitor : ExpressionVisitor {
    private readonly ParameterExpression theta;
    private readonly ParameterExpression x;
    private readonly double[] x0;
    private int outParamIdx;
    public int OutParamIdx => outParamIdx;
    private Stack<Expression> funcStack = new Stack<Expression>(); // collect all monotonic functions to create inverse transformation

    // monotonic functions and inverses
    private readonly MethodInfo exp = typeof(Math).GetMethod("Exp", new[] { typeof(double) });
    private readonly MethodInfo log = typeof(Math).GetMethod("Log", new[] { typeof(double) });
    private readonly MethodInfo tanh = typeof(Math).GetMethod("Tanh", new[] { typeof(double) });
    private readonly MethodInfo sqrt = typeof(Math).GetMethod("Sqrt", new[] { typeof(double) });
    private readonly MethodInfo cbrt = typeof(Functions).GetMethod("Cbrt", new[] { typeof(double) });
    private readonly MethodInfo logistic = typeof(Functions).GetMethod("Logistic", new[] { typeof(double) });
    private readonly MethodInfo invlogistic = typeof(Functions).GetMethod("InvLogistic", new[] { typeof(double) });


    internal ReparameterizeExprVisitor(ParameterExpression theta, ParameterExpression x, double[] x0) {
      this.theta = theta;
      this.x = x;
      this.x0 = (double[])x0.Clone();
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      if (node.NodeType == ExpressionType.Add) {
        if (IsParameter(node.Left)) {
          this.outParamIdx = ParameterIndex(node.Left);
          var g0 = ReplaceParameterWithNumberVisitor.Replace(node.Right, x, x0); // bind x to x0 values
          return Expression.Subtract(Expression.Add(node.Right, CreateInverseFunction(node.Left)), g0);
        } else if (IsParameter(node.Right)) {
          this.outParamIdx = ParameterIndex(node.Right);
          var g0 = ReplaceParameterWithNumberVisitor.Replace(node.Left, x, x0); // bind x to x0 values
          return Expression.Subtract(Expression.Add(node.Left, CreateInverseFunction(node.Right)), g0);
        } else {
          return node.Update(Visit(node.Left), null, Visit(node.Right));
        }
      } else if (node.NodeType == ExpressionType.Subtract) {
        // TODO: similar to case for addition
        throw new NotSupportedException($"node type {node.Type} is not supported for reparameterization.");
      } else if (node.NodeType == ExpressionType.Multiply) {
        if (IsParameter(node.Left)) {
          this.outParamIdx = ParameterIndex(node.Left);
          var g0 = ReplaceParameterWithNumberVisitor.Replace(node.Right, x, x0); // bind x to x0 values
          return Expression.Multiply(node.Right, Expression.Divide(node.Left, g0));
        } else if (IsParameter(node.Right)) {
          this.outParamIdx = ParameterIndex(node.Right);
          var g0 = ReplaceParameterWithNumberVisitor.Replace(node.Left, x, x0); // bind x to x0 values
          return Expression.Multiply(node.Left, Expression.Divide(node.Right, g0));
        } else {
          return node.Update(Visit(node.Left), null, Visit(node.Right));
        }
      } else throw new NotSupportedException($"node type {node.NodeType} is not supported for reparameterization.");
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      if (node.NodeType == ExpressionType.Negate || node.NodeType == ExpressionType.UnaryPlus) {
        return node.Update(Visit(node.Operand));
      } else {
        throw new NotSupportedException($"ReparameterizeExpression {node.NodeType}");
      }
    }

    protected override Expression VisitMethodCall(MethodCallExpression node) {
      if (node.Method == exp || node.Method == log || node.Method == logistic || node.Method == sqrt || node.Method == cbrt || node.Method == tanh) {
        funcStack.Push(node);
        return node.Update(node.Object, new[] { Visit(node.Arguments[0]) });
      } else {
        // TODO implement remaining invertible functions
        throw new NotSupportedException($"ReparameterizeExpression unsupported method call {node.NodeType}");
      }
    }

    // returns the inverse of the functions on the stack 
    private Expression CreateInverseFunction(Expression operand) {
      while (funcStack.Any()) {
        var f = funcStack.Pop();
        if (f.NodeType == ExpressionType.UnaryPlus) operand = operand; // do nothing 
        else if (f.NodeType == ExpressionType.Negate) operand = Expression.Negate(operand);
        else if (f is MethodCallExpression methCallExpr && methCallExpr.Method == logistic) operand = Expression.Call(invlogistic, new[] { operand });
        else throw new NotSupportedException($"inverse of {f} is not supported");
      }
      return operand;
    }

    private int ParameterIndex(Expression expr) {
      var binExpr = (BinaryExpression)expr;
      var constExpr = (ConstantExpression)binExpr.Right;
      return (int)constExpr.Value;
    }

    private bool IsParameter(Expression expr) {
      return expr is BinaryExpression binExpr && expr.NodeType == ExpressionType.ArrayIndex && binExpr.Left == theta;
    }
  }
}

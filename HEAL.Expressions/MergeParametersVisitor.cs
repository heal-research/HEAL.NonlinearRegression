using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  public class RemoveRedundantParametersVisitor : ExpressionVisitor {
    private readonly ParameterExpression thetaParam;
    private readonly double[] thetaValues;
    private readonly List<double> newThetaValues = new List<double>();

    // TODO make thetaValues optional
    public RemoveRedundantParametersVisitor(ParameterExpression theta, double[] thetaValues) {
      this.thetaParam = theta;
      this.thetaValues = thetaValues;
    }

    public double[] GetNewParameterValues => newThetaValues.ToArray();

    protected override Expression VisitBinary(BinaryExpression node) {
      var left = Visit(node.Left);
      var right = Visit(node.Right);
      var leftIsParam = IsParam(left, out var leftArrIdxExpr, out var leftIdx);
      var rightIsParam = IsParam(right, out var rightArrIdxExpr, out var rightIdx); 
      if ( leftIsParam && rightIsParam) {
        // IsParam => left, right are BinaryExpressions (theta[.])
        switch (node.NodeType) {
          case ExpressionType.Add: {
            return CombineParam(leftIdx, rightIdx, (a, b) => a + b);
          }
          case ExpressionType.Subtract: {
            return CombineParam(leftIdx, rightIdx, (a, b) => a + b);
          }
          case ExpressionType.Multiply: {
            return CombineParam(leftIdx, rightIdx, (a, b) => a * b);
          }
          case ExpressionType.Divide: {
            return CombineParam(leftIdx, rightIdx, (a, b) => a / b);
          }
          default: throw new NotSupportedException(node.ToString());
        }
      } else if (leftIsParam) {
        return UpdateParameterIndex(leftArrIdxExpr);
      } else if (rightIsParam) {
        return UpdateParameterIndex(rightArrIdxExpr);
      }

      return node.Update(left, null, right);
    }

    private Expression UpdateParameterIndex(BinaryExpression arrIdxExpr) {
      var idx = (int)((ConstantExpression)arrIdxExpr.Right).Value;
      return NewParameter(thetaValues[idx]);
    }
    private Expression NewParameter(double newVal) {
      newThetaValues.Add(newVal);
      return Expression.ArrayIndex(thetaParam, Expression.Constant(newThetaValues.Count - 1));
    }


    protected override Expression VisitUnary(UnaryExpression node) {
      var operand = Visit(node.Operand);
      if (IsParam(operand, out _, out var idx)) {
        switch (node.NodeType) {
          case ExpressionType.Negate: {
            // negate parameter value
            return NewParameter(thetaValues != null ? -thetaValues[idx] : 0.0);
          } default: throw new NotSupportedException(node.ToString());
        }
      }

      return node.Update(operand);
    }

    protected override Expression VisitMethodCall(MethodCallExpression node) {
      var args = node.Arguments.Select(Visit).ToArray();
      
      // this should work as soon as we are only calling static methods and the method returns double
      if (!node.Method.IsStatic || node.Method.ReturnType != typeof(double))
        throw new NotSupportedException(node.Method.Name);
      
      // all values are parameters -> call the method for the parameter values to fold function and return as new parameter
      if (args.All(arg => IsParam(arg, out _, out _))) {
        double newVal = 0.0;
        if (thetaValues != null) {
          var paramValues = args.Select(arg => {
            IsParam(arg, out _, out var idx); // get idx
            return (object)thetaValues[idx]; // and retrieve value for it
          }).ToArray();
          newVal = (double)node.Method.Invoke(node.Object, paramValues);
        }

        return NewParameter(newVal);
      } else {
        var newArgs = new Expression[args.Length];
        for(int argIdx = 0; argIdx < args.Length; argIdx++) {
          var arg = args[argIdx];
          if (IsParam(arg, out var arrIdxExpr, out var paramIdx)) {
            // collect value and update parameter idx
            newThetaValues.Add(thetaValues[paramIdx]);
            newArgs[argIdx] = arrIdxExpr.Update(arrIdxExpr.Left, null, Expression.Constant(newThetaValues.Count - 1));
          } else {
            newArgs[argIdx] = arg;
          }
        }

        return node.Update(node.Object, newArgs);
      }
    }
    
    
    private Expression CombineParam(int leftIdx, int rightIdx, Func<double, double, double> op) {
      var newValue = thetaValues != null
        ? op(thetaValues[leftIdx], thetaValues[rightIdx])
        : 0.0;
      return NewParameter(newValue);
    }

    private bool IsParam(Expression expr, out BinaryExpression arrayIdxExpr, out int paramIdx) {
      arrayIdxExpr = null;
      paramIdx = -1;
      if (expr.NodeType == ExpressionType.ArrayIndex) {
        arrayIdxExpr = (BinaryExpression)expr;
        if (arrayIdxExpr.Left == thetaParam) {
          paramIdx = (int)((ConstantExpression)arrayIdxExpr.Right).Value;
          return true;
        }

        arrayIdxExpr = null;
      }
      return false;
    }
  }
}

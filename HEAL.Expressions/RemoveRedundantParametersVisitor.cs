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
      if (IsParam(left, out var leftArrIdxExpr, out var leftIdx) && 
          IsParam(right, out var rightArrIdxExpr, out var rightIdx)) {
        // IsParam => left, right are BinaryExpressions (theta[.])
        switch (node.NodeType) {
          case ExpressionType.Add: {
            return CombineParam(leftArrIdxExpr, rightArrIdxExpr, (a, b) => a + b);
          }
          case ExpressionType.Subtract: {
            return CombineParam(leftArrIdxExpr, rightArrIdxExpr, (a, b) => a + b);
          }
          case ExpressionType.Multiply: {
            return CombineParam(leftArrIdxExpr, rightArrIdxExpr, (a, b) => a * b);
          }
          case ExpressionType.Divide: {
            return CombineParam(leftArrIdxExpr, rightArrIdxExpr, (a, b) => a / b);
          }
          default: throw new NotSupportedException(node.ToString());
        }
      }

      return node.Update(left, null, right);
    }


    protected override Expression VisitUnary(UnaryExpression node) {
      var operand = Visit(node.Operand);
      if (IsParam(operand, out var arrIdxExpr, out var idx)) {
        switch (node.NodeType) {
          case ExpressionType.Negate: {
            // negate parameter value
            var newValue = thetaValues != null
              ? -thetaValues[idx]
              : 0.0;
            newThetaValues.Add(newValue);
            return arrIdxExpr;
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
      
      // all values are parameters -> call the method for the parameter values and return as new parameter
      if (args.All(arg => IsParam(arg, out _, out _))) {
        double newVal = 0.0;
        if (thetaValues != null) {
          var paramValues = args.Select(arg => {
            IsParam(arg, out _, out var idx); // get idx
            return (object)thetaValues[idx]; // and retrieve value for it
          }).ToArray();
          newVal = (double)node.Method.Invoke(node.Object, paramValues);
        }
        
        newThetaValues.Add(newVal);
        return Expression.ArrayIndex(thetaParam, Expression.Constant(newThetaValues.Count - 1));
      }
      return node.Update(node.Object, args);
    }
    
    
    private Expression CombineParam(BinaryExpression left, BinaryExpression right, Func<double, double, double> op) {
      var leftArrayIndex = left.Right as ConstantExpression;
      var rightArrayIndex = right.Right as ConstantExpression;
      var newValue = thetaValues != null
        ? op(thetaValues[(int)leftArrayIndex.Value], thetaValues[(int)rightArrayIndex.Value])
        : 0.0;
      newThetaValues.Add(newValue);

      left.Update(left.Left, null, Expression.Constant(newThetaValues.Count - 1));
      return left;
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

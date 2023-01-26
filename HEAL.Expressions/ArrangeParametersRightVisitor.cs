using System;
using System.Data;
using System.Linq;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  public class ArrangeParametersRightVisitor : ExpressionVisitor {
    private readonly ParameterExpression p;
    private readonly double[] pValues;

    private ArrangeParametersRightVisitor(ParameterExpression p, double[] pValues) {
      this.p = p;
      this.pValues = pValues;
    }

    // pValues is updated
    public static Expression<Expr.ParametricFunction> Execute(Expression<Expr.ParametricFunction> expr,
      ParameterExpression p, double[] pValues) {

      return (Expression<Expr.ParametricFunction>)(new ArrangeParametersRightVisitor(p, pValues)).Visit(expr);
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      var left = Visit(node.Left);
      var right = Visit(node.Right);
      var leftIsParam = IsParam(left, out var leftParam, out var leftIdx);
      var rightIsParam = IsParam(right, out var rightParam, out var rightIdx);
      BinaryExpression leftBinary = left as BinaryExpression;

      if (leftIsParam && rightIsParam) {
        // fold directly
        switch (node.NodeType) {
          case ExpressionType.Add: {
              pValues[leftIdx] += pValues[rightIdx]; return leftParam;
            }
          case ExpressionType.Subtract: {
              pValues[leftIdx] -= pValues[rightIdx]; return leftParam;
            }
          case ExpressionType.Multiply: {
              pValues[leftIdx] *= pValues[rightIdx]; return leftParam;
            }
          case ExpressionType.Divide: {
              pValues[leftIdx] /= pValues[rightIdx]; return leftParam;
            }
          default: throw new NotSupportedException($"{node}");
        }
      } else if (leftIsParam) {
        // switch left <-> right to move param to right
        switch (node.NodeType) {
          case ExpressionType.Add: {
              return node.Update(right, null, left);
            }
          case ExpressionType.Subtract: {
              return Expression.Add(Expression.Negate(right), left);
            }
          case ExpressionType.Multiply: {
              return node.Update(right, null, left);
            }
          case ExpressionType.Divide: {
              return Expression.Multiply(Expression.Divide(Expression.Constant(1.0), right), left);
            }
          default: throw new NotSupportedException($"{node}");
        }
      } else if (rightIsParam) {
        // if param is on the right, prefer add over sub and mul over div
        switch (node.NodeType) {
          case ExpressionType.Subtract: {
              pValues[rightIdx] *= -1;
              return Expression.Add(left, right);
            }
          case ExpressionType.Divide: {
              pValues[rightIdx] = 1.0 / pValues[rightIdx];
              return Expression.Multiply(left, right);
            }
          default: return node.Update(left, null, right);
        }
      } else if (leftBinary != null
                && leftBinary.NodeType == node.NodeType
                && IsParam(leftBinary.Right, out _, out _)) {
        // right of left is parameter --> move parameter up
        // (a ° p) ° b --> (a ° b) ° p
        var ab = leftBinary.Update(leftBinary.Left, null, right);
        return node.Update(ab, null, leftBinary.Right);
      } else {
        return node.Update(left, null, right);
      }
    }


    /*
    // fold unary operations on parameters
    protected override Expression VisitUnary(UnaryExpression node) {
      var operand = Visit(node.Operand);
      if (IsParam(operand, out var arrayIdxExpr, out var paramIdx)) {
        switch (node.NodeType) {
          case ExpressionType.Negate: {
              pValues[paramIdx] *= -1;
              return arrayIdxExpr;
            }
          default: throw new NotSupportedException($"node");
        }
      } else return node.Update(operand);
    }
    */
    protected override Expression VisitUnary(UnaryExpression node) {
      var operand = Visit(node.Operand);
      switch ((node, operand)) {
        case ( { NodeType: ExpressionType.Negate },
              BinaryExpression(ExpressionType.ArrayIndex, _, ConstantExpression(_) constExpr) binExpr)
        when binExpr.Left == p: {
            pValues[(int)constExpr.Value] *= -1;
            return binExpr;
          }
        case ( { NodeType: ExpressionType.UnaryPlus }, _): {
            return operand;
          }
        default: return node.Update(operand);
      }
    }


    // fold methods that only depend on one or multiple parameters
    protected override Expression VisitMethodCall(MethodCallExpression node) {
      var args = node.Arguments.Select(Visit).ToArray();

      // this should work as soon as we are only calling static methods and the method returns double
      if (!node.Method.IsStatic || node.Method.ReturnType != typeof(double))
        throw new NotSupportedException(node.Method.Name);

      // all values are parameters -> call the method for the parameter values to fold function and return as new parameter
      if (args.All(arg => IsParam(arg, out _, out _))) {
        double newVal = 0.0;
        if (p != null) {
          var paramValues = args.Select(arg => {
            IsParam(arg, out _, out var idx); // get idx
            return (object)pValues[idx]; // and retrieve value for it
          }).ToArray();
          newVal = (double)node.Method.Invoke(node.Object, paramValues);
        }

        IsParam(args[0], out var arrIdxExpr, out var paramIdx);
        pValues[paramIdx] = newVal; // update value of first parameter

        return arrIdxExpr; // return first parameter instead of methodCall
      } else {
        // non-linear function
        return node.Update(node.Object, args);
      }
    }

    private bool IsParam(Expression expr, out BinaryExpression arrayIdxExpr, out int paramIdx) {
      arrayIdxExpr = null;
      paramIdx = -1;
      if (expr.NodeType == ExpressionType.ArrayIndex) {
        arrayIdxExpr = (BinaryExpression)expr;
        if (arrayIdxExpr.Left == p) {
          paramIdx = (int)((ConstantExpression)arrayIdxExpr.Right).Value;
          return true;
        }

        arrayIdxExpr = null;
      }
      return false;
    }
  }
}

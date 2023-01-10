using System;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;

namespace HEAL.Expressions {
  // Mainly useful to simplify expressions after symbolic derivation
  // folds constants and handles x + 0, x * 0, x * 1, x/1 , 0/x...
  // TODO would be interesting to extend this to other numeric types and using zero / identity instead of (0.0 and 1.0)
  // TODO FoldConstants and FoldParameters should implement the same rules (combine!)

  public class FoldConstantsVisitor : ExpressionVisitor {
    private MethodInfo aq = typeof(Functions).GetMethod("AQ", new[] { typeof(double), typeof(double) });
    private MethodInfo pow = typeof(Math).GetMethod("Pow", new[] { typeof(double), typeof(double) });
    private MethodInfo exp = typeof(Math).GetMethod("Exp", new[] { typeof(double) });
    protected override Expression VisitBinary(BinaryExpression node) {
      var left = Visit(node.Left);
      var right = Visit(node.Right);
      var leftConst = left as ConstantExpression;
      var rightConst = right as ConstantExpression;
      switch (node.NodeType) {
        case ExpressionType.Add: {
            if (leftConst != null && rightConst != null) return Expression.Constant((double)leftConst.Value + (double)rightConst.Value);
            else if (leftConst != null && leftConst.Value.Equals(0.0)) return right;
            else if (rightConst != null && rightConst.Value.Equals(0.0)) return left;
            else return node.Update(left, null, right);
          }
        case ExpressionType.Subtract: {
            if (leftConst != null && rightConst != null) return Expression.Constant((double)leftConst.Value - (double)rightConst.Value);
            else if (leftConst != null && leftConst.Value.Equals(0.0)) return Expression.Negate(right);
            else if (rightConst != null && rightConst.Value.Equals(0.0)) return left;
            else return node.Update(left, null, right);
          }
        case ExpressionType.Multiply: {
            if (leftConst != null && rightConst != null) return Expression.Constant((double)leftConst.Value * (double)rightConst.Value);
            else if (leftConst != null && leftConst.Value.Equals(0.0)) return Expression.Constant(0.0);
            else if (leftConst != null && leftConst.Value.Equals(1.0)) return right;
            else if (leftConst != null && leftConst.Value.Equals(-1.0)) return Expression.Negate(right);
            else if (rightConst != null && rightConst.Value.Equals(0.0)) return Expression.Constant(0.0);
            else if (rightConst != null && rightConst.Value.Equals(1.0)) return left;
            else if (rightConst != null && rightConst.Value.Equals(-1.0)) return Expression.Negate(left);
            else if (left is BinaryExpression leftDivExpr && leftDivExpr.NodeType == ExpressionType.Divide) {
              // 1/lr * right --> right / lr
              if (leftDivExpr.Left is ConstantExpression leftConstExpr && (double)leftConstExpr.Value == 1.0) return leftDivExpr.Update(right, null, leftDivExpr.Right);
              else return node.Update(left, null, right);
            } else if (right is BinaryExpression rightDivExpr && rightDivExpr.NodeType == ExpressionType.Divide) {
              // left * 1/rr -> left / rr
              if (rightDivExpr.Left is ConstantExpression rightConstExpr && (double)rightConstExpr.Value == 1.0) return rightDivExpr.Update(left, null, rightDivExpr.Right);
              else return node.Update(left, null, right);
            } else return node.Update(left, null, right);
          }
        case ExpressionType.Divide: {
            if (leftConst != null && rightConst != null) return Expression.Constant((double)leftConst.Value / (double)rightConst.Value);
            else if (leftConst != null && leftConst.Value.Equals(0.0)) return Expression.Constant(0.0);
            else if (rightConst != null && rightConst.Value.Equals(0.0)) return Expression.Constant(double.NaN);
            else if (rightConst != null && rightConst.Value.Equals(1.0)) return left;
            else if (rightConst != null && rightConst.Value.Equals(-1.0)) return Expression.Negate(left);
            else return node.Update(left, null, right);
          }
          // extend by time as necessary.
      }
      return node.Update(left, null, right);
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      var x = Visit(node.Operand);
      if (node.NodeType == ExpressionType.Negate &&
         x is ConstantExpression xConst) return Expression.Constant(-1.0 * (double)xConst.Value);
      else if (node.NodeType == ExpressionType.UnaryPlus) {
        return x;
      }
      return node.Update(x);
    }

    protected override Expression VisitMethodCall(MethodCallExpression node) {
      var args = node.Arguments.Select(Visit).ToArray();

      // method is static and 
      // all arguments are constant doubles
      // -> call the method (we don't care which method it is
      if (node.Method.IsStatic &&
          args.All(arg => arg.NodeType == ExpressionType.Constant
                          && arg.Type == typeof(double))) {
        var values = args.Select(arg => ((ConstantExpression)arg).Value).ToArray();
        return Expression.Constant(node.Method.Invoke(node.Object, values));
      } else if (node.Method == aq && args[1].NodeType == ExpressionType.Constant) {
        // aq(x, c) = x / sqrt(1 + c²) = 1/sqrt(1+c²) * x
        var c = (double)((ConstantExpression)args[1]).Value;
        return Expression.Multiply(Expression.Constant(1.0 / Math.Sqrt(1.0 + c * c)), args[0]);
      } else if (node.Method == pow
        && args[1].NodeType == ExpressionType.Constant
        && args[0] is MethodCallExpression subFuncCall
        && subFuncCall.Method == exp) {
        // exp(x)^c = exp(c*x)
        var x = subFuncCall.Arguments[0];
        return subFuncCall.Update(subFuncCall.Object, new[] { Expression.Multiply(x, args[1]) });
      }


      return node.Update(node.Object, args);
    }
  }
}

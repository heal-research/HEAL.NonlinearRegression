using System.Linq;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  // TODO would be interesting to extend this to other numeric types and using zero / identity instead of (0.0 and 1.0)

  public class SimplifyVisitor : ExpressionVisitor {
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
          else if (leftConst!=null && leftConst.Value.Equals(0.0)) return Expression.Negate(right);
          else if(rightConst != null && rightConst.Value.Equals(0.0)) return left;
          else return node.Update(left, null, right);
        }
        case ExpressionType.Multiply: {
          if (leftConst != null && rightConst != null) return Expression.Constant((double)leftConst.Value * (double)rightConst.Value);
          else if (leftConst != null && leftConst.Value.Equals(0.0)) return Expression.Constant(0.0);
          else if (leftConst != null && leftConst.Value.Equals(1.0)) return right;
          else if (rightConst != null && rightConst.Value.Equals(0.0)) return Expression.Constant(0.0);
          else if (rightConst != null && rightConst.Value.Equals(1.0)) return left;
          else return node.Update(left, null, right);
        }
        case ExpressionType.Divide: {
          if (leftConst != null && rightConst != null) return Expression.Constant((double)leftConst.Value / (double)rightConst.Value);
          else if (leftConst != null && leftConst.Value.Equals(0.0)) return Expression.Constant(0.0);
          else if (rightConst != null && rightConst.Value.Equals(0.0)) return Expression.Constant(double.NaN);
          else if (rightConst != null && rightConst.Value.Equals(1.0)) return left;
          else return node.Update(left, null, right);
        }
        // extend by time as necessary.
      }
      return node.Update(left, null, right);
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      var x = Visit(node.Operand);
      if(node.NodeType == ExpressionType.Negate && 
         x is ConstantExpression xConst) return Expression.Constant(-1.0 * (double)xConst.Value);
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
      }

      return node.Update(node.Object, args);
    }
  }
}
using System;
using System.Linq;
using System.Linq.Expressions;

namespace HEAL.Expressions {

  // we assume parameters occur only as right arguments
  // we can also assume that we only add or multiply parameters
  // the ArrangeParameterRightVisitor should be called first
  // TODO always call the arrangeRightVisitor first
  public class FoldParametersVisitor : ExpressionVisitor {
    private readonly ParameterExpression thetaParam;
    private readonly double[] thetaValues;

    // TODO make thetaValues optional
    public FoldParametersVisitor(ParameterExpression theta, double[] thetaValues) {
      this.thetaParam = theta;
      this.thetaValues = thetaValues;
    }

    public double[] GetNewParameterValues => thetaValues.ToArray();

    protected override Expression VisitBinary(BinaryExpression node) {
      var left = Visit(node.Left);
      var right = Visit(node.Right);
      var rightIsParam = IsParam(node.Right, out var paramExpr, out var paramIdx);
      var leftBinary = left as BinaryExpression;
      if (rightIsParam && leftBinary != null) {
        //   left           node  right
        // (... (o) ... )  (+/*)    p

        if (IsParam(leftBinary.Right, out var innerParamExpr, out var innerParamIdx)) {
          // (... (o) innerP )  (+/*)    p
          switch (node.NodeType) {
            case ExpressionType.Add: {
                if (leftBinary.NodeType == ExpressionType.Add) {
                  // merge 
                  thetaValues[innerParamIdx] += thetaValues[paramIdx];
                  return left;
                } else if (leftBinary.NodeType == ExpressionType.Subtract) {
                  throw new NotSupportedException("should be handled in ArrangeParametersRightVisitor");
                } else {
                  // unchanged
                  return node.Update(left, null, right);
                }
              }
            case ExpressionType.Multiply: {
                // (... (o) innerP )  *   p
                if (leftBinary.NodeType == ExpressionType.Multiply) {
                  // merge 
                  thetaValues[innerParamIdx] *= thetaValues[paramIdx];
                  return left;
                } else if (leftBinary.NodeType == ExpressionType.Divide) {
                  throw new NotSupportedException("should be handled in ArrangeParametersRightVisitor");
                } else return node.Update(left, null, right); // return unchanged. we would need to propagate the parameter value down recursively here
              }
            default: throw new NotSupportedException($"{node}");
          }
        }
      }

      return node.Update(left, null, right);
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      if (node.NodeType == ExpressionType.UnaryPlus) {
        return Visit(node.Operand);
      } else if (node.NodeType == ExpressionType.Negate) {
        var opd = Visit(node.Operand);
        if (IsParam(opd, out var _, out var paramIdx)) {
          thetaValues[paramIdx] *= -1;
        }
        return opd;
      } else
        return base.Visit(node);
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

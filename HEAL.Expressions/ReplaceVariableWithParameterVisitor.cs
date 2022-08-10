using System;
using System.Collections.Generic;
using System.Linq.Expressions;

namespace HEAL.Expressions {
  public class ReplaceVariableWithParameterVisitor : ExpressionVisitor {
    private readonly double replVal;
    private readonly int varIdx;
    private readonly ParameterExpression x;
    private readonly ParameterExpression theta;
    private readonly List<double> newTheta;
    private readonly double[] thetaValues;

    public ReplaceVariableWithParameterVisitor(ParameterExpression theta, double[] thetaValues, ParameterExpression x, int varIdx, double replVal) {
      this.theta = theta;
      this.thetaValues = thetaValues;
      this.x = x;
      this.varIdx = varIdx;
      this.replVal = replVal;
      this.newTheta = new List<double>();
    }
    
    protected override Expression VisitBinary(BinaryExpression node) {
      if (node.NodeType == ExpressionType.ArrayIndex) {
        // is it a reference to x[varIdx]?
        if (node.Left == x && (int)((ConstantExpression)node.Right).Value == varIdx) {
          newTheta.Add(replVal);
          return node.Update(theta, null, Expression.Constant(newTheta.Count - 1));
        } else if(node.Left == theta) {
          // copy existing theta and update the index
          var idx = (int)((ConstantExpression)node.Right).Value;
          newTheta.Add(thetaValues[idx]);
          return node.Update(theta, null, Expression.Constant(newTheta.Count - 1));
        }
      } 
      return base.VisitBinary(node);
    }
  }
}

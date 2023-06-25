using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using static HEAL.Expressions.Expr;

namespace HEAL.Expressions {

  public class FoldParametersVisitor : ExpressionVisitor {
    private readonly ParameterExpression thetaParam;
    private readonly List<double> thetaValues;

    private readonly MethodInfo aq = typeof(Functions).GetMethod("AQ", new[] { typeof(double), typeof(double) });
    private readonly MethodInfo pow = typeof(Math).GetMethod("Pow", new[] { typeof(double), typeof(double) });
    private readonly MethodInfo exp = typeof(Math).GetMethod("Exp", new[] { typeof(double) });

    public static ParameterizedExpression FoldParameters(ParameterizedExpression expr) {
      var visitor = new FoldParametersVisitor(expr.p, expr.pValues);
      var newExpr = (Expression<ParametricFunction>)visitor.Visit(expr.expr);
      return new ParameterizedExpression(newExpr, expr.p, visitor.GetNewParameterValues);
    }

    public FoldParametersVisitor(ParameterExpression theta, double[] thetaValues) {
      this.thetaParam = theta;
      this.thetaValues = thetaValues.ToList();
    }

    public double[] GetNewParameterValues => thetaValues.ToArray();

    protected override Expression VisitBinary(BinaryExpression node) {
      var left = Visit(node.Left);
      var right = Visit(node.Right);
      var rightIsParam = IsParam(node.Right, out var paramExpr, out var paramIdx);
      if (rightIsParam && left is BinaryExpression leftBinary) {
        switch (node.NodeType) {
          case ExpressionType.Add: {
              var terms = CollectTermsVisitor.CollectTerms(leftBinary);
              var parameterSum = terms.Where(IsParameter).Sum(ParameterValue);
              var remainingTerms = terms.Where(t => !IsParameter(t)).Aggregate(Expression.Add);
              return Expression.Add(remainingTerms, NewParam(parameterSum + ParameterValue(right))); // merge right parameter into existing parameters
            }
          case ExpressionType.Multiply: {
              // first extract additive parameters
              var terms = CollectTermsVisitor.CollectTerms(leftBinary).ToArray();
              if (terms.All(HasScalingParameter)) { // TODO problematic with constants
                return terms.Select(t => ScaleTerm(t, thetaValues[paramIdx])).Aggregate(Expression.Add);
              }

              // not all terms have a scaling parameter
              return node.Update(left, null, right);
            }
          case ExpressionType.Subtract: {
              throw new NotSupportedException("should be handled in ArrangeParametersRightVisitor");
            }
          case ExpressionType.Divide: {
              throw new NotSupportedException("should be handled in ArrangeParametersRightVisitor");
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
          return NewParam(thetaValues[paramIdx] * -1);
        } else {
          return node.Update(opd);
        }
      } else
        return base.Visit(node);
    }

    protected override Expression VisitMethodCall(MethodCallExpression node) {
      var args = node.Arguments.Select(arg => Visit(arg)).ToArray();
      // if all arguments to this function are parameters then we can remove the function and use a parameter instead
      if (args.All(arg => IsParam(arg, out _, out _))) {
        var argValues = new object[args.Length];
        IsParam(args[0], out _, out var firstParamIdx);
        for (int i = 0; i < args.Length; i++) {
          IsParam(args[i], out _, out var paramIdx);
          argValues[i] = thetaValues[paramIdx];
        }
        return NewParam((double)node.Method.Invoke(node.Object, argValues)); // eval function and create new parameter
      } else if (node.Method == aq && IsParam(args[1], out var arrIdxExpr, out var paramIdx)) {
        // aq(x, p) = x / sqrt(1 + p²) = x * 1/sqrt(1+p²) = x * p'
        return Expression.Multiply(args[0], NewParam(1.0 / Math.Sqrt(1 + thetaValues[paramIdx] * thetaValues[paramIdx])));
      } else if (node.Method == pow
        && IsParam(args[1], out _, out _)
        && args[0] is MethodCallExpression expCall
        && expCall.Method == exp) {
        // exp(x)^p = exp(p*x)
        var x = expCall.Arguments[0];
        return expCall.Update(expCall.Object, new[] { Expression.Multiply(x, args[1]) });
      } // no rule for aq(a,b)^p because this would introduce a new parameter
      else {
        return node.Update(node.Object, args);
      }
    }
    private Expression NewParam(double val) {
      thetaValues.Add(val);
      return Expression.ArrayIndex(thetaParam, Expression.Constant(thetaValues.Count - 1));
    }
    public Expression ScaleTerm(Expression term, double scale) {
      // return expr;
      if (scale == 1.0) return term; // nothing to do
                                     // TODO handle x * 0.0?

      var factors = CollectFactorsVisitor.CollectFactors(term);
      // product of parameter values
      var currentScalingFactor = factors.Where(IsParameter).Aggregate(1.0, (agg, f) => agg * ParameterValue(f)); // fold product
      var otherFactors = factors.Where(f => !IsParameter(f));
      if (otherFactors.Count() == factors.Count()) throw new InvalidProgramException(); // we must have at least one scaling parameter
      if (otherFactors.Any())
        return Expression.Multiply(otherFactors.Aggregate(Expression.Multiply), NewParam(scale * currentScalingFactor));
      else
        return NewParam(scale * currentScalingFactor);
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


    private int ParameterIndex(Expression expr) {
      if (!IsParameter(expr)) throw new InvalidProgramException("internal error");
      var binExpr = (BinaryExpression)expr;
      return (int)((ConstantExpression)binExpr.Right).Value;
    }

    private double ParameterValue(Expression expr) {
      return thetaValues[ParameterIndex(expr)];
    }

    private bool IsParameter(Expression expr) {
      return expr.NodeType == ExpressionType.ArrayIndex &&
        ((BinaryExpression)expr).Left == thetaParam;
    }

    private bool HasScalingParameter(Expression expr) {
      var factors = CollectFactorsVisitor.CollectFactors(expr);
      return factors.Any(f => IsParam(f, out _, out _));
    }
  }
}

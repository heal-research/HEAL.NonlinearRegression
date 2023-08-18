using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using static HEAL.Expressions.Expr;
using static System.Formats.Asn1.AsnWriter;

namespace HEAL.Expressions {
  /// <summary>
  /// Extracts parameters out of non-linear functions as far as possible. 
  /// This is a precursory step for simplification to remove linearly dependent parameters.
  /// </summary>
  /// 
  public class LiftParametersVisitor : ExpressionVisitor {
    private readonly ParameterExpression theta;
    private readonly List<double> thetaValues;

    private LiftParametersVisitor(ParameterExpression theta, double[] thetaValues) {
      this.theta = theta;
      this.thetaValues = thetaValues.ToList();
    }

    public static ParameterizedExpression LiftParameters(ParameterizedExpression expr) {
      var v = new LiftParametersVisitor(expr.p, expr.pValues);
      var newExpr = (Expression<ParametricFunction>)v.Visit(expr.expr);
      var updatedThetaValues = v.thetaValues.ToArray();

      // NOTE: the following would also replace exising parameters that happen to be 1, -1, 0 and therefore would (incorrectly) reduce the dof of the expression.
      // replace parameters with the value 1.0, -1.0, 0.0 with constants
      // var selectedIdx = Enumerable.Range(0, updatedThetaValues.Length)
      //   .Where(idx => updatedThetaValues[idx] == 1.0
      //   || updatedThetaValues[idx] == -1.0
      //   || updatedThetaValues[idx] == 0.0)
      //   .ToArray();
      // 
      // var replV = new ReplaceParameterWithNumberVisitor(expr.p, updatedThetaValues, selectedIdx);
      // newExpr = (Expression<ParametricFunction>)replV.Visit(newExpr);

      // remove unused parameters
      var collectVisitor = new CollectParametersVisitor(expr.p, updatedThetaValues);
      newExpr = (Expression<ParametricFunction>)collectVisitor.Visit(newExpr);
      return new ParameterizedExpression(newExpr, expr.p, collectVisitor.GetNewParameterValues);
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      var left = Visit(node.Left);
      var right = Visit(node.Right);
      if (node.NodeType == ExpressionType.Divide) {
        var leftTerms = CollectTermsVisitor.CollectTerms(left);
        var leftScale = 1.0;
        if (leftTerms.All(HasScalingParameter)) {
          leftScale = ParameterValue(FindScalingParameter(leftTerms.First()));
          leftTerms = ScaleTerms(leftTerms.First(), leftTerms.Skip(1), leftScale);
        }
        var rightScale = 1.0;
        var rightTerms = CollectTermsVisitor.CollectTerms(right);
        if (rightTerms.All(HasScalingParameter)) {
          rightScale = ParameterValue(FindScalingParameter(rightTerms.First()));
          rightTerms = ScaleTerms(rightTerms.First(), rightTerms.Skip(1), rightScale);
        }
        if (leftScale != 1.0 || rightScale != 1.0) {
          return Expression.Multiply(
            Expression.Divide(
              leftTerms.Aggregate(Expression.Add),
              rightTerms.Aggregate(Expression.Add)),
            NewParam(leftScale / rightScale));
        } else {
          return node.Update(left, null, right);
        }
      }
      return node.Update(left, null, right);
    }


    protected override Expression VisitUnary(UnaryExpression node) {
      // lift parameters out of negation if possible
      var opd = Visit(node.Operand);
      if (node.NodeType == ExpressionType.UnaryPlus) {
        return node.Update(opd);
      } else if (node.NodeType == ExpressionType.Negate) {
        var terms = CollectTermsVisitor.CollectTerms(opd); // addition or subtraction
        if (terms.All(HasScalingParameter)) {
          var p0 = ParameterValue(FindScalingParameter(terms.Last()));
          terms = ScaleTerms(terms.Last(), terms.Take(terms.Count()-1), p0);
          return Expression.Multiply(terms.Aggregate(Expression.Add), NewParam(-p0));
        }
      }
      return node.Update(opd);
    }

    protected override Expression VisitMethodCall(MethodCallExpression node) {
      // sqrt(p1 f(x)) = sqrt(p1) * sqrt(f(x)), p1 f(x) >= 0, p1 > 0, f(x) > 0
      // cbrt %
      // (p1 f(x)) ^ e = p1^e * f(x)^e
      // log(p1 f(x)) = log(f(x)) + log(p1), p1 > 0, p1 * f(x) > 0
      // exp(f(x) + p1) = exp(p1) * exp(f(x)) 

      var args = node.Arguments.Select(Visit).ToArray();

      if (node.Method.Name == "Sqrt") {
        var terms = CollectTermsVisitor.CollectTerms(args[0]);
        if (terms.All(HasScalingParameter)) {
          var p0 = Math.Abs(ParameterValue(FindScalingParameter(terms.First())));
          terms = ScaleTerms(terms.First(), terms.Skip(1), p0);
          return Expression.Multiply(node.Update(node.Object, new[] { terms.Aggregate(Expression.Add) }), NewParam(Math.Sqrt(p0)));
        }
      } else if (node.Method.Name == "Abs") {
        var terms = CollectTermsVisitor.CollectTerms(args[0]);
        if (terms.All(HasScalingParameter)) {
          var p0 = Math.Abs(ParameterValue(FindScalingParameter(terms.First())));
          terms = ScaleTerms(terms.First(), terms.Skip(1), p0);
          return Expression.Multiply(node.Update(node.Object, new[] { terms.Aggregate(Expression.Add) }), NewParam(p0));
        }
      } else if (node.Method.Name == "Cbrt") {
        var terms = CollectTermsVisitor.CollectTerms(args[0]);
        if (terms.All(HasScalingParameter)) {
          var p0 = ParameterValue(FindScalingParameter(terms.First()));
          terms = ScaleTerms(terms.First(), terms.Skip(1), p0);
          return Expression.Multiply(node.Update(node.Object, new[] { terms.Aggregate(Expression.Add) }), NewParam(Functions.Cbrt(p0)));
        }
      } else if (node.Method.Name == "Pow") {
        if (args[1] is ConstantExpression || IsParameter(args[1])) {
          var terms = CollectTermsVisitor.CollectTerms(args[0]);
          double exponent;
          if (args[1] is ConstantExpression constExpr) {
            exponent = (double)constExpr.Value; // this has to be a double
          } else if (IsParameter(args[1])) {
            exponent = thetaValues[ParameterIndex(args[1])];
          } else throw new InvalidProgramException("not possible");

          if (terms.All(HasScalingParameter)) {
            var p0 = ParameterValue(FindScalingParameter(terms.First()));
            terms = ScaleTerms(terms.First(), terms.Skip(1), p0);

            return Expression.Multiply(node.Update(node.Object, new[] { terms.Aggregate(Expression.Add), args[1] }), NewParam(Math.Pow(p0, exponent)));
          }
        }
      } else if (node.Method.Name == "Log") {
        var terms = CollectTermsVisitor.CollectTerms(args[0]);

        if (terms.All(HasScalingParameter)) {
          var p0 = Math.Abs(ParameterValue(FindScalingParameter(terms.First())));
          terms = ScaleTerms(terms.First(), terms.Skip(1), p0);

          return Expression.Add(node.Update(node.Object, new[] { terms.Aggregate(Expression.Add) }), NewParam(Math.Log(p0)));
        } else return node.Update(node.Object, args);
      } else if (node.Method.Name == "Exp") {
        var terms = CollectTermsVisitor.CollectTerms(args[0]);
        var parameterTerms = terms.Where(IsParameter);
        var remainingTerms = terms.Where(t => !IsParameter(t));
        if (parameterTerms.Any()) {
          var offset = parameterTerms.Sum(t => Math.Exp(ParameterValue(t)));
          return Expression.Multiply(node.Update(node.Object, new[] { remainingTerms.Aggregate(Expression.Add) }), NewParam(offset));
        }
      } else if (node.Method.Name == "AQ") {
        // aq(x,y) = aq(1,y)*x
        return Expression.Multiply(node.Update(node.Object, new[] { Expression.Constant(1.0), args[1] }), args[0]);
      }

      // all others:
      return node.Update(node.Object, args);
    }

    private IEnumerable<Expression> ScaleTerms(Expression parameterlessTerm, IEnumerable<Expression> parametricTerms, double scale) {
      // TODO: remove scaling factor from first term
      return new[] { RemoveScale(parameterlessTerm, scale) }.Concat(parametricTerms.Select(t => ScaleTerm(t, 1.0 / scale)));
    }

    private Expression RemoveScale(Expression term, double scale) {
      var factors = CollectFactorsVisitor.CollectFactors(term);
      // product of parameter values
      var currentScalingFactor = factors.Where(IsParameter).Aggregate(1.0, (agg, f) => agg * ParameterValue(f)); // fold product
      var otherFactors = factors.Where(f => !IsParameter(f));
      if (otherFactors.Count() == factors.Count()) throw new InvalidProgramException(); // we must have at least one scaling parameter
      if (otherFactors.Any())
        return Expression.Multiply(otherFactors.Aggregate(Expression.Multiply), Expression.Constant(currentScalingFactor / scale));
      else
        return Expression.Constant(currentScalingFactor / scale);

    }
    private Expression ScaleTerm(Expression term, double scale) {
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

    private Expression NewParam(double val) {

      thetaValues.Add(val);
      return Expression.ArrayIndex(theta, Expression.Constant(thetaValues.Count - 1));
    }

    private Expression FindScalingParameter(Expression expr) {
      var factors = CollectFactorsVisitor.CollectFactors(expr);
      return factors.FirstOrDefault(IsParameter);
    }

    private bool HasScalingParameter(Expression expr) {
      return FindScalingParameter(expr) != null;
    }
    private double ParameterValue(Expression expr) {
      return thetaValues[ParameterIndex(expr)];
    }
    private int ParameterIndex(Expression expr) {
      if (!IsParameter(expr)) throw new InvalidProgramException("internal error");
      var binExpr = (BinaryExpression)expr;
      return (int)((ConstantExpression)binExpr.Right).Value;
    }

    private bool IsParameter(Expression expr) {
      return expr.NodeType == ExpressionType.ArrayIndex &&
        ((BinaryExpression)expr).Left == theta;
    }
  }
}

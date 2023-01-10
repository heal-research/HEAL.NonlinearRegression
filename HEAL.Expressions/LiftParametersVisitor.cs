using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using static HEAL.Expressions.Expr;

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

    public static Expression<ParametricFunction> LiftParameters(Expression<ParametricFunction> expr, ParameterExpression theta, double[] thetaValues, out double[] newThetaValues) {
      var v = new LiftParametersVisitor(theta, thetaValues);
      var newExpr = (Expression<ParametricFunction>)v.Visit(expr);
      var updatedThetaValues = v.thetaValues.ToArray();

      // replace parameters with the value 1.0, -1.0, 0.0 with constants
      var selectedIdx = Enumerable.Range(0, updatedThetaValues.Length)
        .Where(idx => updatedThetaValues[idx] == 1.0
        || updatedThetaValues[idx] == -1.0
        || updatedThetaValues[idx] == 0.0)
        .ToArray();

      var replV = new ReplaceParameterWithNumberVisitor(theta, updatedThetaValues, selectedIdx);
      newExpr = (Expression<ParametricFunction>)replV.Visit(newExpr);

      // remove unused parameters
      var collectVisitor = new CollectParametersVisitor(theta, updatedThetaValues);
      newExpr = (Expression<ParametricFunction>)collectVisitor.Visit(newExpr);
      newThetaValues = collectVisitor.GetNewParameterValues;

      return newExpr;
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      var left = Visit(node.Left);
      var right = Visit(node.Right);
      if (node.NodeType == ExpressionType.Divide) {
        var numeratorTerms = CollectTermsVisitor.CollectTerms(left);
        var numeratorFactor = 1.0;
        if (numeratorTerms.All(HasScalingParameter)) {
          var paramExpr = FindScalingParameter(numeratorTerms.First());
          numeratorFactor = ParameterValue(paramExpr);
          foreach (var t in numeratorTerms) {
            thetaValues[ParameterIndex(FindScalingParameter(t))] /= numeratorFactor;
          }
        }
        var denomFactor = 1.0;
        var denomTerms = CollectTermsVisitor.CollectTerms(right);
        if (denomTerms.All(HasScalingParameter)) {
          var paramExpr = FindScalingParameter(denomTerms.First());
          denomFactor = ParameterValue(paramExpr);
          foreach (var t in denomTerms) {
            thetaValues[ParameterIndex(FindScalingParameter(t))] /= denomFactor;
          }
        }
        if (numeratorFactor != 1.0 || denomFactor != 1.0) {
          return Expression.Multiply(node.Update(left, null, right), CreateParameter(numeratorFactor / denomFactor));
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
          var paramExpr = FindScalingParameter(terms.Last());
          var p0 = ParameterValue(paramExpr);
          foreach (var t in terms) {
            thetaValues[ParameterIndex(FindScalingParameter(t))] /= p0;
          }
          return Expression.Multiply(opd, CreateParameter(-p0));
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

      var args = node.Arguments.Select(arg => Visit(arg)).ToArray();

      if (node.Method.Name == "Sqrt") {
        var terms = CollectTermsVisitor.CollectTerms(args[0]);
        if (terms.All(HasScalingParameter)) {
          var paramExpr = FindScalingParameter(terms.First());
          var p0 = Math.Abs(ParameterValue(paramExpr));
          foreach (var t in terms) {
            thetaValues[ParameterIndex(FindScalingParameter(t))] /= p0;
          }

          return Expression.Multiply(node.Update(node.Object, args), CreateParameter(Math.Sqrt(p0)));
        }
        node.Update(node.Object, args);
      } else if (node.Method.Name == "Abs") {
        var terms = CollectTermsVisitor.CollectTerms(args[0]);
        if (terms.All(HasScalingParameter)) {
          var paramExpr = FindScalingParameter(terms.First());
          var p0 = Math.Abs(ParameterValue(paramExpr));
          foreach (var t in terms) {
            thetaValues[ParameterIndex(FindScalingParameter(t))] /= p0;
          }

          return Expression.Multiply(node.Update(node.Object, args), CreateParameter(p0));
        }
        node.Update(node.Object, args);
      } else if (node.Method.Name == "Cbrt") {
        var terms = CollectTermsVisitor.CollectTerms(args[0]);
        if (terms.All(HasScalingParameter)) {
          var paramExpr = FindScalingParameter(terms.First());
          var p0 = ParameterValue(paramExpr);
          foreach (var t in terms) {
            thetaValues[ParameterIndex(FindScalingParameter(t))] /= p0;
          }

          return Expression.Multiply(node.Update(node.Object, args), CreateParameter(Functions.Cbrt(p0)));
        }
        node.Update(node.Object, args);
      } else if (node.Method.Name == "Pow") {
        var terms = CollectTermsVisitor.CollectTerms(args[0]);
        double exponent;
        if (args[1] is ConstantExpression constExpr) {
          exponent = (double)constExpr.Value; // this has to be a constant
        } else if (IsParameter(args[1])) {
          exponent = thetaValues[ParameterIndex(args[1])];
        } else throw new NotSupportedException("Power can only have constant or parameter exponents.");

        if (terms.All(HasScalingParameter)) {
          var paramExpr = FindScalingParameter(terms.First());
          var p0 = ParameterValue(paramExpr);
          foreach (var t in terms) {
            thetaValues[ParameterIndex(FindScalingParameter(t))] /= p0;
          }

          return Expression.Multiply(node.Update(node.Object, args), CreateParameter(Math.Pow(p0, exponent)));
        }
        node.Update(node.Object, args);
      } else if (node.Method.Name == "Log" /* || node.Method.Name == "plog"*/) {
        var terms = CollectTermsVisitor.CollectTerms(args[0]);

        if (terms.All(HasScalingParameter)) {
          var paramExpr = FindScalingParameter(terms.First());
          var p0 = Math.Abs(ParameterValue(paramExpr));
          foreach (var t in terms) {
            thetaValues[ParameterIndex(FindScalingParameter(t))] /= p0;
          }

          return Expression.Add(node.Update(node.Object, args), CreateParameter(Math.Log(p0)));
        }
        node.Update(node.Object, args);

      } else if (node.Method.Name == "Exp") {
        var terms = CollectTermsVisitor.CollectTerms(args[0]);
        var offsetParams = terms.Where(IsParameter);
        if (offsetParams.Any()) {
          var f = 1.0;
          foreach (var offsetParam in offsetParams) {
            var val = ParameterValue(offsetParam);
            f *= Math.Exp(val);
            thetaValues[ParameterIndex(offsetParam)] = 0.0;
          }

          return Expression.Multiply(node.Update(node.Object, args), CreateParameter(f));
        }
        node.Update(node.Object, args);
      }
      // cannot extract parameters from: sin, cos, tan, tanh
      return node.Update(node.Object, args);
    }

    private Expression CreateParameter(double value) {
      var pExpr = Expression.ArrayIndex(theta, Expression.Constant(thetaValues.Count));
      thetaValues.Add(value);
      return pExpr;
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

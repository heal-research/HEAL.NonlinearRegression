using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using BinaryExpressionRule = System.ValueTuple<System.Func<System.Linq.Expressions.BinaryExpression, bool>, System.Func<System.Linq.Expressions.BinaryExpression, System.Linq.Expressions.Expression>>;
using UnaryExpressionRule = System.ValueTuple<System.Func<System.Linq.Expressions.UnaryExpression, bool>, System.Func<System.Linq.Expressions.UnaryExpression, System.Linq.Expressions.Expression>>;
using MethodCallExpressionRule = System.ValueTuple<System.Func<System.Linq.Expressions.MethodCallExpression, bool>, System.Func<System.Linq.Expressions.MethodCallExpression, System.Linq.Expressions.Expression>>;
using System.Reflection;

namespace HEAL.Expressions {



  public class RuleBasedSimplificationVisitor :ExpressionVisitor {
    private readonly List<(Func<BinaryExpression, bool> Match, Func<BinaryExpression, Expression> Apply)> binaryRules = new List<BinaryExpressionRule>();
    private readonly List<(Func<UnaryExpression, bool> Match, Func<UnaryExpression, Expression> Apply)> unaryRules = new List<UnaryExpressionRule>();
    private readonly List<(Func<MethodCallExpression, bool> Match, Func<MethodCallExpression, Expression> Apply)> callRules = new List<MethodCallExpressionRule>();
    private readonly ParameterExpression p;
    private readonly List<double> pValues;

    public RuleBasedSimplificationVisitor(ParameterExpression p, double[] pValues) {
      this.p = p;
      this.pValues = pValues.ToList();

      // NOTE: We must be careful because parameters might occur multiple times.
      // TODO: Some of the other visitors can be replaced by rules to simplify the code base.

      binaryRules.AddRange(new[] {
      // x / x => 1.0
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Divide && e.Left.ToString() == e.Right.ToString(),
        e => Expression.Constant(1.0)
        ),
      // x - x => 0.0
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Subtract && e.Left.ToString() == e.Right.ToString(),
        e => Expression.Constant(0.0)
        ),
      // x * x => pow(x, 2)
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Multiply && e.Left.ToString() == e.Right.ToString(),
        e => Visit(Expression.Call(pow, e.Left, Expression.Constant(2.0)))
        ),
      // x * pow(x, z) => pow(x, z+1)
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Multiply && e.Right is MethodCallExpression callExpr && callExpr.Method == pow
        && e.Left.ToString() == e.Right.ToString(),
        e => {
          var callExpr = (MethodCallExpression)e.Right;
          return Visit(callExpr.Update(callExpr.Object,
            new [] {callExpr.Arguments[0],
              Expression.Add(callExpr.Arguments[1], Expression.Constant(1.0)) }));
        }
        ),
      // pow(x, z) * x => pow(x, z+1)
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Multiply && e.Left is MethodCallExpression callExpr && callExpr.Method == pow
        && callExpr.Arguments[0].ToString() == e.Right.ToString(),
        e => {
          var callExpr = (MethodCallExpression)e.Left;
          return Visit(callExpr.Update(callExpr.Object,
            new [] {callExpr.Arguments[0],
              Expression.Add(callExpr.Arguments[1], Expression.Constant(1.0)) }));
        }
        ),
      // pow(x, z) / x => pow(x, z-1)
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Divide && e.Left is MethodCallExpression callExpr && callExpr.Method == pow
        && callExpr.Arguments[0].ToString() == e.Right.ToString(),
        e => {
          var callExpr = (MethodCallExpression)e.Left;
          return Visit(callExpr.Update(callExpr.Object,
            new [] {callExpr.Arguments[0],
              Expression.Subtract(callExpr.Arguments[1], Expression.Constant(1.0))}));
        }),
      // x / pow(x, z) => pow(x, 1 - z)
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Divide && e.Right is MethodCallExpression callExpr && callExpr.Method == pow
             && e.Left.ToString() == callExpr.Arguments[0].ToString(),
        e => {
          var callExpr = (MethodCallExpression)e.Right;
          return Visit(callExpr.Update(callExpr.Object,
            new [] {callExpr.Arguments[0],
              Expression.Subtract(Expression.Constant(1.0), callExpr.Arguments[1]) }));
        }
        ),
      // const ° const -> const
      new BinaryExpressionRule(
        e => IsConstant(e.Left) && IsConstant(e.Right),
        e => Expression.Constant(Apply(e.NodeType, GetConstantValue(e.Left), GetConstantValue(e.Right)))
        ),
      // param ° param -> param
      new BinaryExpressionRule(
        e => IsParameter(e.Left) && IsParameter(e.Right),
        e => NewParameter(Apply(e.NodeType, GetParameterValue(e.Left), GetParameterValue(e.Right)))
        ),
      // param ° const -> param
      new BinaryExpressionRule(
        e => IsParameter(e.Left) && IsConstant(e.Right),
        e => NewParameter(Apply(e.NodeType, GetParameterValue(e.Left), GetConstantValue(e.Right)))
        ),
      // const ° param -> param
      new BinaryExpressionRule(
        e => IsConstant(e.Left) && IsParameter(e.Right),
        e => NewParameter(Apply(e.NodeType, GetConstantValue(e.Left), GetParameterValue(e.Right)))
        ),
      // nest associative left a ° (b ° c) -> (a ° b) ° c
      new BinaryExpressionRule(
        e => IsAssociative(e) && e.Right is BinaryExpression rightBinExpr && rightBinExpr.NodeType == e.NodeType,
        e => {
          var rightBin = (BinaryExpression)e.Right;
          return Visit(rightBin.Update(e.Update(e.Left, null, rightBin.Left), null, rightBin.Right));
          }
        ),
       // move parameters right p ° x -> x ° p
       new BinaryExpressionRule(
        e => IsAssociative(e) && IsParameter(e.Left),
        e => e.Update(e.Right, null, e.Left)
        ),
       // 1/pow(x, y) -> pow(x, -y)
       new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Divide && e.Right is MethodCallExpression callExpr && callExpr.Method == pow,
        e => {
          var callExpr = (MethodCallExpression)e.Right;
          return Visit(callExpr.Update(callExpr.Object, new [] {callExpr.Arguments[0], Expression.Negate(callExpr.Arguments[1]) }));
            }
        ),

      });

      unaryRules.AddRange(new[] {
        // +x -> x
        new UnaryExpressionRule(
          e => e.NodeType == ExpressionType.UnaryPlus,
          e => e.Operand
          ),
        // -(p) -> p
        new UnaryExpressionRule(
          e => e.NodeType == ExpressionType.Negate && IsParameter(e.Operand),
          e => NewParameter(-GetParameterValue(e.Operand))
          ),
        // -(const) -> -const
        new UnaryExpressionRule(
          e => e.NodeType == ExpressionType.Negate && IsConstant(e.Operand),
          e => Expression.Constant(-GetConstantValue(e.Operand))
          ),
          /// -(a + b) -> -a + -b
        new UnaryExpressionRule(
          e => e.NodeType == ExpressionType.Negate && e.Operand.NodeType == ExpressionType.Add,
          e => {
            var binExpr = (BinaryExpression)e.Operand;
            return Visit(binExpr.Update(Expression.Negate(binExpr.Left), null, Expression.Negate(binExpr.Right)));
              }
          ),
          /// -(a - b) -> b - a
        new UnaryExpressionRule(
          e => e.NodeType == ExpressionType.Negate && e.Operand.NodeType == ExpressionType.Subtract,
          e => {
            var binExpr = (BinaryExpression)e.Operand;
            return Visit(binExpr.Update(binExpr.Right, null, binExpr.Left));
              }
          ),
          /// -(a * b) -> a * -b
        new UnaryExpressionRule(
          e => e.NodeType == ExpressionType.Negate && e.Operand.NodeType == ExpressionType.Multiply,
          e => {
            var binExpr = (BinaryExpression)e.Operand;
            return Visit(binExpr.Update(binExpr.Left, null, Expression.Negate(binExpr.Right)));
              }
          ),
          /// -(a / b) -> a / -b
        new UnaryExpressionRule(
          e => e.NodeType == ExpressionType.Negate && e.Operand.NodeType == ExpressionType.Divide,
          e => {
            var binExpr = (BinaryExpression)e.Operand;
            return Visit(binExpr.Update(binExpr.Left, null, Expression.Negate(binExpr.Right)));
              }
          ),
        // odd functions: -f(x) = f(-x)
        new UnaryExpressionRule(
          e => e.NodeType == ExpressionType.Negate && e.Operand is MethodCallExpression callExpr && IsOddFunction(callExpr),
          e => {
            var callExpr = (MethodCallExpression)e.Operand;
            return Visit(callExpr.Update(callExpr.Object, new [] {Expression.Negate(callExpr.Arguments[0]) }));
          }),
          /// -(-x) -> x
        new UnaryExpressionRule(
          e => e.NodeType == ExpressionType.Negate && e.Operand.NodeType == ExpressionType.Negate,
          e => ((UnaryExpression) e.Operand).Operand
          ),

      });

      callRules.AddRange(new[] {
      // fold constants
      new MethodCallExpressionRule(
        e => e.Arguments.All(IsConstant),
        e => Expression.Constant(e.Method.Invoke(e.Object, e.Arguments.Select(GetConstantValue).OfType<object>().ToArray()))
        ),
      // fold parameters
      new MethodCallExpressionRule(
        e => e.Arguments.All(IsParameter),
        e => NewParameter((double) e.Method.Invoke(e.Object, e.Arguments.Select(GetParameterValue).OfType<object>().ToArray()))
        ),
      // only parameters and constants (rule for folding constants should be applied first)
      new MethodCallExpressionRule(
        e => e.Arguments.All(e => IsParameter(e) || IsConstant(e)),
        e => NewParameter((double) e.Method.Invoke(e.Object, e.Arguments.Select(GetParameterOrConstantValue).OfType<object>().ToArray()))
        ),
      // abs(-(x)) -> abs(x)
      new MethodCallExpressionRule(
        e => e.Method == abs && e.Arguments[0].NodeType == ExpressionType.Negate,
        e => e.Update(e.Object, new [] {((UnaryExpression) e.Arguments[0]).Operand
})
        ),
      // pow(x, 0) -> 1
      new MethodCallExpressionRule(
        e => e.Method == pow && IsConstant(e.Arguments[1]) && GetConstantValue(e.Arguments[1]) == 0.0,
        e => Expression.Constant(1.0)
        ),
      // pow(x, 1) -> 1
      new MethodCallExpressionRule(
        e => e.Method == pow && IsConstant(e.Arguments[1]) && GetConstantValue(e.Arguments[1]) == 1.0,
        e => e.Arguments[0]
        ),
      // pow(1, x) -> 1
      new MethodCallExpressionRule(
        e => e.Method == pow && IsConstant(e.Arguments[0]) && GetConstantValue(e.Arguments[0]) == 1.0,
        e => Expression.Constant(1.0)
        ),
      // pow(x*y, z) -> pow(x,z) * pow(y,z) // BEWARE: duplicates parameters
      new MethodCallExpressionRule(
        e => e.Method == pow && e.Arguments[0].NodeType == ExpressionType.Multiply,
        e => {
          var binExpr = (BinaryExpression)e.Arguments[0];
          return Visit(Expression.Multiply(
            Expression.Call(pow, binExpr.Left, e.Arguments[1]),
            Expression.Call(pow, binExpr.Right, e.Arguments[1])));
        }),
      // pow(x / y, z) -> pow(x,z) / pow(y,z) // BEWARE: duplicates parameters
      new MethodCallExpressionRule(
        e => e.Method == pow && e.Arguments[0].NodeType == ExpressionType.Divide,
        e => {
          var div = (BinaryExpression)e.Arguments[0];
          return Visit(Expression.Divide(
            Expression.Call(pow, div.Left, e.Arguments[1]),
            Expression.Call(pow, div.Right, e.Arguments[1])));
        }
        )
      });
    }

    private bool IsOddFunction(MethodCallExpression callExpr) => callExpr.Method == sin || callExpr.Method == cbrt;

    private bool IsAssociative(BinaryExpression e) => e.NodeType == ExpressionType.Add || e.NodeType == ExpressionType.Multiply;

    private double Apply(ExpressionType nodeType, double v1, double v2) {
      switch (nodeType) {
        case ExpressionType.Add: return v1 + v2;
        case ExpressionType.Subtract: return v1 - v2;
        case ExpressionType.Multiply: return v1 * v2;
        case ExpressionType.Divide: return v1 / v2;
        case ExpressionType.Power: return Math.Pow(v1, v2);
        default: throw new NotSupportedException();
      }
    }

    private Expression NewParameter(double val) {
      pValues.Add(val);
      return Expression.ArrayIndex(p, Expression.Constant(pValues.Count - 1));
    }

    private double GetConstantValue(Expression constExpr) => (double)((ConstantExpression)constExpr).Value;
    private double GetParameterValue(Expression expr) => pValues[(int)((ConstantExpression)((BinaryExpression)expr).Right).Value];
    private double GetParameterOrConstantValue(Expression expr) {
      if (expr is ConstantExpression) return GetConstantValue(expr);
      else if (IsParameter(expr)) return GetParameterValue(expr);
      else throw new ArgumentException();
    }

    private bool IsConstant(Expression expr) => expr is ConstantExpression;
    private bool IsParameter(Expression expr) => expr is BinaryExpression binExpr && binExpr.Left == p;

    private bool HasParameters(Expression left) => CountParametersVisitor.Count(left, p) > 0;

    public static ParameterizedExpression Simplify(ParameterizedExpression expr) {
      var checkVisitor = CheckExprVisitor.CheckValid(expr.expr);

      var v = new RuleBasedSimplificationVisitor(expr.p, expr.pValues);
      var body = v.Visit(expr.expr.Body);
      return new ParameterizedExpression(expr.expr.Update(body, expr.expr.Parameters), expr.p, v.pValues.ToArray());
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      // simplify left and right first
      var left = Visit(node.Left);
      var right = Visit(node.Right);
      Expression result = node.Update(left, null, right);

      var r = binaryRules.FirstOrDefault(r => r.Match((BinaryExpression)result));
      while (r != default(BinaryExpressionRule)) {
        result = r.Apply((BinaryExpression)result);
        if (result is BinaryExpression binExpr) {
          r = binaryRules.FirstOrDefault(r => r.Match(binExpr));
        } else break;
      }
      return result;
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      var opd = Visit(node.Operand);
      Expression result = node.Update(opd);
      var r = unaryRules.FirstOrDefault(r => r.Match((UnaryExpression)result));
      while (r != default(UnaryExpressionRule)) {
        result = r.Apply((UnaryExpression)result);
        if (result is UnaryExpression unaryExpr) {
          r = unaryRules.FirstOrDefault(r => r.Match(unaryExpr));
        } else break;
      }
      return result;
    }

    protected override Expression VisitMethodCall(MethodCallExpression node) {
      Expression result = node.Update(node.Object, node.Arguments.Select(Visit));
      var r = callRules.FirstOrDefault(r => r.Match((MethodCallExpression)result));
      while (r != default(MethodCallExpressionRule)) {
        result = r.Apply((MethodCallExpression)result);
        if (result is MethodCallExpression callExpr) {
          r = callRules.FirstOrDefault(r => r.Match(callExpr));
        } else break;
      }
      return result;
    }


    private readonly MethodInfo abs = typeof(Math).GetMethod("Abs", new[] { typeof(double) });
    private readonly MethodInfo log = typeof(Math).GetMethod("Log", new[] { typeof(double) });
    private readonly MethodInfo exp = typeof(Math).GetMethod("Exp", new[] { typeof(double) });
    private readonly MethodInfo sqrt = typeof(Math).GetMethod("Sqrt", new[] { typeof(double) });
    private readonly MethodInfo sin = typeof(Math).GetMethod("Sin", new[] { typeof(double) });
    private readonly MethodInfo cos = typeof(Math).GetMethod("Cos", new[] { typeof(double) });
    private readonly MethodInfo tanh = typeof(Math).GetMethod("Tanh", new[] { typeof(double) });
    private readonly MethodInfo pow = typeof(Math).GetMethod("Pow", new[] { typeof(double), typeof(double) });
    private readonly MethodInfo cbrt = typeof(Functions).GetMethod("Cbrt", new[] { typeof(double) });
    private readonly MethodInfo sign = typeof(Functions).GetMethod("Sign", new[] { typeof(double) });
    private readonly MethodInfo aq = typeof(Functions).GetMethod("AQ", new[] { typeof(double), typeof(double) });
    private readonly MethodInfo logistic = typeof(Functions).GetMethod("Logistic", new[] { typeof(double) });
    private readonly MethodInfo invLogistic = typeof(Functions).GetMethod("InvLogistic", new[] { typeof(double) });
    private readonly MethodInfo logisticPrime = typeof(Functions).GetMethod("LogisticPrime", new[] { typeof(double) });
    private readonly MethodInfo invLogisticPrime = typeof(Functions).GetMethod("InvLogisticPrime", new[] { typeof(double) });

  }
}

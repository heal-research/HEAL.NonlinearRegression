using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using BinaryExpressionRule = System.ValueTuple<System.Func<System.Linq.Expressions.BinaryExpression, bool>, System.Func<System.Linq.Expressions.BinaryExpression, System.Linq.Expressions.Expression>>;
using UnaryExpressionRule = System.ValueTuple<System.Func<System.Linq.Expressions.UnaryExpression, bool>, System.Func<System.Linq.Expressions.UnaryExpression, System.Linq.Expressions.Expression>>;
using MethodCallExpressionRule = System.ValueTuple<System.Func<System.Linq.Expressions.MethodCallExpression, bool>, System.Func<System.Linq.Expressions.MethodCallExpression, System.Linq.Expressions.Expression>>;
using System.Reflection;

namespace HEAL.Expressions {



  public class RuleBasedSimplificationVisitor : ExpressionVisitor {
    private readonly List<(Func<BinaryExpression, bool> Match, Func<BinaryExpression, Expression> Apply)> binaryRules = new List<BinaryExpressionRule>();
    private readonly List<(Func<UnaryExpression, bool> Match, Func<UnaryExpression, Expression> Apply)> unaryRules = new List<UnaryExpressionRule>();
    private readonly List<(Func<MethodCallExpression, bool> Match, Func<MethodCallExpression, Expression> Apply)> callRules = new List<MethodCallExpressionRule>();
    private readonly ParameterExpression p;
    private readonly List<double> pValues;

    public RuleBasedSimplificationVisitor(ParameterExpression p, double[] pValues) {
      this.p = p;
      this.pValues = pValues.ToList();

      AddConstantFoldingRules();
      AddBasicRules();
      AddReparameterizationRules();

      // NOTE: We must be careful because parameters might occur multiple times.
      // TODO: Some of the other visitors can be replaced by rules to simplify the code base.
      // TODO: memoization of Visit methods
    }

    private void ClearRules() {
      binaryRules.Clear();
      unaryRules.Clear();
      callRules.Clear();
    }

    private void AddConstantFoldingRules() {
      binaryRules.AddRange(new[] {
                // x (+/-) 0.0
        new BinaryExpressionRule(
        e => (e.NodeType == ExpressionType.Add || e.NodeType == ExpressionType.Subtract)
        && e.Right is ConstantExpression constExpr && (double)constExpr.Value == 0.0,
        e => e.Left
        ),
      // 0.0 + x
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Add
        && e.Left is ConstantExpression constExpr && (double)constExpr.Value == 0.0,
        e => e.Right
        ),
      // 0.0 - x
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Subtract
        && e.Left is ConstantExpression constExpr && (double)constExpr.Value == 0.0,
        e => Visit(Expression.Negate(e.Right))
        ),
      // x * 0.0
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Multiply
        && e.Right is ConstantExpression constExpr && (double)constExpr.Value == 0.0,
        e => e.Right
        ),
      // x * 1.0
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Multiply
        && e.Right is ConstantExpression constExpr && (double)constExpr.Value == 1.0,
        e => e.Left
        ),
      // x * -1.0
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Multiply
        && e.Right is ConstantExpression constExpr && (double)constExpr.Value == -1.0,
        e => Visit(Expression.Negate(e.Left))
        ),
      // 0.0 * x
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Multiply
        && e.Left is ConstantExpression constExpr && (double)constExpr.Value == 0.0,
        e => e.Left
        ),
      // 1.0 * x
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Multiply
        && e.Left is ConstantExpression constExpr && (double)constExpr.Value == 1.0,
        e => e.Right
        ),
      // -1.0 * x
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Multiply
        && e.Left is ConstantExpression constExpr && (double)constExpr.Value == -1.0,
        e => Visit(Expression.Negate(e.Left))
        ),
      // x / 1.0
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Divide
        && e.Right is ConstantExpression constExpr && (double)constExpr.Value == 1.0,
        e => e.Left
        ),
      // x / 0.0
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Divide
        && e.Right is ConstantExpression constExpr && (double)constExpr.Value == 0.0,
        e => Expression.Constant(double.NaN)
        ),
      // x / -1.0
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Divide
        && e.Right is ConstantExpression constExpr && (double)constExpr.Value == -1.0,
        e => Visit(Expression.Negate(e.Left))
        ),
      // 0 / x (-> 0  .. this is not strictly true because 0/0 is undefined but such exceptions are not relevant to us in useful models)
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Divide
        && e.Left is ConstantExpression constExpr && (double)constExpr.Value == 0.0,
        e => e.Left // 0.0
        ),
        // const ° const -> const
        new BinaryExpressionRule(
        e => IsConstant(e.Left) && IsConstant(e.Right),
        e => Expression.Constant(Apply(e.NodeType, GetConstantValue(e.Left), GetConstantValue(e.Right)))
        ),
      // (a ° const) ° const
      new BinaryExpressionRule(
        e => IsAssociative(e) && e.Left.NodeType == e.NodeType && e.Left is BinaryExpression leftExpr
        && IsConstant(leftExpr.Right) && IsConstant(e.Right),
        e => {
          var leftExpr = (BinaryExpression)e.Left;
          return Visit(e.Update(leftExpr.Left, null,
            Expression.Constant(Apply(e.NodeType, GetConstantValue(leftExpr.Right), GetConstantValue(e.Right)))));
            }
        ),
      });

      unaryRules.AddRange(new[] {
        // +x -> x
        new UnaryExpressionRule(
          e => e.NodeType == ExpressionType.UnaryPlus,
          e => e.Operand
          ),
        // -(const) -> -const
        new UnaryExpressionRule(
          e => e.NodeType == ExpressionType.Negate && IsConstant(e.Operand),
          e => Expression.Constant(-GetConstantValue(e.Operand))
          ),
      });

      callRules.AddRange(new[] {
        // fold constants
        new MethodCallExpressionRule(
          e => e.Arguments.All(IsConstant),
          e => Expression.Constant(e.Method.Invoke(e.Object, e.Arguments.Select(GetConstantValue).OfType<object>().ToArray()))
          ),
        // pow(x, 0) -> 1
        new MethodCallExpressionRule(
          e => e.Method == pow && IsConstant(e.Arguments[1]) && GetConstantValue(e.Arguments[1]) == 0.0,
          e => Expression.Constant(1.0)
          ),
        // pow(0, x) -> 0
        // (this is not 100% correct because 0^0 is not defined. But for our purposes 0^0 is irrelevant
        new MethodCallExpressionRule(
          e => e.Method == pow && IsConstant(e.Arguments[0]) && GetConstantValue(e.Arguments[0]) == 0.0,
          e => Expression.Constant(0.0)
          ),
  
        // pow(x, 1) -> x
        new MethodCallExpressionRule(
          e => e.Method == pow && IsConstant(e.Arguments[1]) && GetConstantValue(e.Arguments[1]) == 1.0,
          e => e.Arguments[0]
          ),
        // pow(1, x) -> 1
        new MethodCallExpressionRule(
          e => e.Method == pow && IsConstant(e.Arguments[0]) && GetConstantValue(e.Arguments[0]) == 1.0,
          e => Expression.Constant(1.0)
          ),

      });
    }
    private void AddBasicRules() {
      binaryRules.AddRange(new[] {
      // x / x => 1.0
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Divide && e.Left.ToString() == e.Right.ToString(),
        e => Expression.Constant(1.0)
        ),
      // (a * x) / x => a
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Divide && e.Left is BinaryExpression left &&
        left.NodeType == ExpressionType.Multiply && left.Right.ToString() == e.Right.ToString(),
        e => ((BinaryExpression)e.Left).Left
        ),
      // (a / x) * x => a
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Multiply && e.Left is BinaryExpression left &&
        left.NodeType == ExpressionType.Divide && left.Right.ToString() == e.Right.ToString(),
        e => ((BinaryExpression)e.Left).Left
        ),

      // x - x => 0.0
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Subtract && e.Left.ToString() == e.Right.ToString(),
        e => Expression.Constant(0.0)
        ),
       // (a + x) - x -> a
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Subtract && e.Left.NodeType == ExpressionType.Add &&
             e.Left is BinaryExpression binExpr && binExpr.Right.ToString() == e.Right.ToString(),
        e => ((BinaryExpression)e.Left).Left
        ),
      // (a - x) + x -> a
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Add && e.Left.NodeType == ExpressionType.Subtract &&
             e.Left is BinaryExpression binExpr && binExpr.Right.ToString() == e.Right.ToString(),
        e => ((BinaryExpression)e.Left).Left
        ),

      // x + x => x * 2
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Add && e.Left.ToString() == e.Right.ToString(),
        e => Visit(Expression.Multiply(e.Left, Expression.Constant(2.0)))
        ),
      // (a + x) + x => a + x * 2
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Add
          && e.Left is BinaryExpression left && left.NodeType == ExpressionType.Add
          && left.Right.ToString() == e.Right.ToString(),
        e => {
          var left = (BinaryExpression)e.Left;
          return Visit(left.Update(left.Left, null, Expression.Multiply(left.Right, Expression.Constant(2.0))));
          }
        ),


      // x * a (+/-) x  -> x * (a (+/-) 1)
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Add && e.Left is BinaryExpression left
          && left.NodeType == ExpressionType.Multiply && left.Left.ToString() == e.Right.ToString(),
        e => {
          var left = (BinaryExpression)e.Left;
          return Visit(Expression.Multiply(left.Left, Expression.Add(left.Right, Expression.Constant(1.0))));
          }
        ),
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Subtract && e.Left is BinaryExpression left
          && left.NodeType == ExpressionType.Multiply && left.Left.ToString() == e.Right.ToString(),
        e => {
          var left = (BinaryExpression)e.Left;
          return Visit(Expression.Multiply(left.Left, Expression.Subtract(left.Right, Expression.Constant(1.0))));
          }
        ),
      // x * alpha (+/-) x * beta -> x * (alpha (+/-) beta)
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Add
          && e.Left is BinaryExpression left && e.Right is BinaryExpression right
          && left.NodeType == ExpressionType.Multiply && right.NodeType == ExpressionType.Multiply
          && left.Left.ToString() == right.Left.ToString(),
        e => {
          var left = (BinaryExpression)e.Left;
          var right = (BinaryExpression)e.Right;
          return Visit(Expression.Multiply(left.Left, Expression.Add(left.Right, right.Right)));
          }
        ),
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Subtract
          && e.Left is BinaryExpression left && e.Right is BinaryExpression right
          && left.NodeType == ExpressionType.Multiply && right.NodeType == ExpressionType.Multiply
          && left.Left.ToString() == right.Left.ToString(),
        e => {
          var left = (BinaryExpression)e.Left;
          var right = (BinaryExpression)e.Right;
          return Visit(Expression.Multiply(left.Left, Expression.Subtract(left.Right, right.Right)));
          }
        ),

      // x * x => pow(x, 2)
      // TODO: (a * x) * x
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Multiply && e.Left.ToString() == e.Right.ToString(),
        e => Visit(Expression.Call(pow, e.Left, Expression.Constant(2.0)))
        ),

      // (1/x) * y
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Multiply && e.Left.NodeType == ExpressionType.Divide && !IsParameter(e.Right),
        e => {
          var binExpr = (BinaryExpression)e.Left;
          return Visit(Expression.Divide(Expression.Multiply(binExpr.Left, e.Right), binExpr.Right));
          }
        ),
      // y * (1/x)
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Multiply && e.Right.NodeType == ExpressionType.Divide,
        e => {
          var binExpr = (BinaryExpression)e.Right;
          return Visit(Expression.Divide(Expression.Multiply(e.Left, binExpr.Left), binExpr.Right));
          }
        ),
      // x + (-y) -> x - y 
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Add && e.Right.NodeType == ExpressionType.Negate,
        e => Visit(Expression.Subtract(e.Left, ((UnaryExpression)e.Right).Operand))
        ),
      
      // x + -c -> x - c  (negative constant)
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Add && e.Right is ConstantExpression constExpr && (double)constExpr.Value < 0.0,
        e => Visit(Expression.Subtract(e.Left, Expression.Constant(- (double)((ConstantExpression)e.Right).Value)))
        ),

      // x + (-(x))
      // TODO: necessary?
      // new BinaryExpressionRule(
      //   e => e.NodeType == ExpressionType.Add && e.Right.NodeType == ExpressionType.Negate
      //     && e.Right is UnaryExpression negExpr && negExpr.Operand.ToString() == e.Left.ToString(),
      //   e => Expression.Constant(0.0)
      //   ),
      // (a + x) + (-(x))
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Add && e.Left.NodeType == ExpressionType.Add
             && e.Right.NodeType == ExpressionType.Negate
             && e.Left is BinaryExpression left
             && e.Right is UnaryExpression negExpr && negExpr.Operand.ToString() == left.Right.ToString(),
        e => {
          var leftBin = (BinaryExpression)e.Left;
          return leftBin.Left;
          }
        ),


      // a ° (b ° c) -> (a ° b) ° c
      // This implicitly handles (a ° b) ° (c ° d) -> ((a ° b) ° c) ° d
      new BinaryExpressionRule(
        e => IsAssociative(e) && e.NodeType == e.Right.NodeType,
        e => {
          var rightExpr = (BinaryExpression)e.Right;
          return Visit(rightExpr.Update(e.Update(e.Left, null, rightExpr.Left), null, rightExpr.Right));
        }
        ),
      
      // sort arguments in (+,*) operations
      // (a ° c) ° b -> (a ° b) ° c   (with compare(b,c) <= 0)
      new BinaryExpressionRule(
        e => IsCommutative(e) && IsAssociative(e) && e.Left.NodeType == e.NodeType &&
        e.Left is BinaryExpression left && Compare(left.Right, e.Right) > 0,
        e => {
          var left = (BinaryExpression)e.Left;
          return Visit(e.Update(left.Update(left.Left, null, e.Right), null, left.Right));
        }
        ),
      // c ° b -> b ° c   (with compare(b,c) <= 0)
      new BinaryExpressionRule(
        e => IsCommutative(e) && Compare(e.Left, e.Right) > 0,
        e => {
          return Visit(e.Update(e.Right, null, e.Left));
        }
        ),
      

      // (a / b) / (c / d) -->  (a * d) / (b * c)
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Divide
          && e.Left.NodeType == ExpressionType.Divide
          && e.Right.NodeType == ExpressionType.Divide,
        e => {
          var left = (BinaryExpression)e.Left;
          var right = (BinaryExpression)e.Right;
          return Expression.Divide(
            Expression.Multiply(left.Left, right.Right),
            Expression.Multiply(left.Right, right.Left));
        }
        ),
      // (a / b) / c -->  a / (b * c)
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Divide
          && e.Left.NodeType == ExpressionType.Divide,
        e => {
          var left = (BinaryExpression)e.Left;
          return Expression.Divide(
            left.Left,
            Expression.Multiply(left.Right, e.Right));
        }
        ),
      // a / (b / c) -->  (a * c) / b 
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Divide
          && e.Right.NodeType == ExpressionType.Divide,
        e => {
          var right = (BinaryExpression)e.Right;
          return Expression.Divide(
            Expression.Multiply(e.Left, right.Right),
            right.Left);
        }
        ),
      // x * pow(x, z) => pow(x, z+1)
      // TODO: (a * x) * pow(x, z)
      new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Multiply && e.Right is MethodCallExpression callExpr && callExpr.Method == pow
        && e.Left.ToString() == callExpr.Arguments[0].ToString(),
        e => {
          var callExpr = (MethodCallExpression)e.Right;
          return Visit(callExpr.Update(callExpr.Object,
            new [] {callExpr.Arguments[0],
              Expression.Add(callExpr.Arguments[1], Expression.Constant(1.0)) }));
        }
        ),
      // pow(x, z) * x => pow(x, z+1) // TODO: not possible because calls are ordered right of binary expressions
      
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
      // nest associative left a ° (b ° c) -> (a ° b) ° c
      new BinaryExpressionRule(
        e => IsAssociative(e) && e.Right is BinaryExpression rightBinExpr && rightBinExpr.NodeType == e.NodeType,
        e => {
          var rightBin = (BinaryExpression)e.Right;
          return Visit(rightBin.Update(e.Update(e.Left, null, rightBin.Left), null, rightBin.Right));
          }
        ),
       // p / x -> 1/x * p
       // TODO (a * p) / x
       new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Divide && IsParameter(e.Left) && !IsParameter(e.Right),
        e => Visit(Expression.Multiply(Expression.Divide(Expression.Constant(1.0), e.Right), e.Left))
        ),
       // p - x -> -x + p
       // TODO (a (+/-) p) - x
       new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Subtract && IsParameter(e.Left) && !IsParameterOrConstant(e.Right),
        e => Visit(Expression.Add(Expression.Negate(e.Right), e.Left))
        ),
       // 1/pow(x, y) -> pow(x, -y)
       new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Divide && e.Right is MethodCallExpression callExpr && callExpr.Method == pow,
        e => {
          var callExpr = (MethodCallExpression)e.Right;
          return Visit(Expression.Multiply(e.Left, callExpr.Update(callExpr.Object, new [] {callExpr.Arguments[0], Expression.Negate(callExpr.Arguments[1]) })));
            }
        ),

      });
      unaryRules.AddRange(new[] {
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
            // prefer negation for parameters or constants
            if(binExpr.Left is ConstantExpression || IsParameter(binExpr.Left)) {
              return Visit(binExpr.Update(Expression.Negate(binExpr.Left), null, binExpr.Right));
            } else return Visit(binExpr.Update(binExpr.Left, null, Expression.Negate(binExpr.Right)));
              }
          ),
          /// -(a / b) -> a / -b
        new UnaryExpressionRule(
          e => e.NodeType == ExpressionType.Negate && e.Operand.NodeType == ExpressionType.Divide,
          e => {
            var binExpr = (BinaryExpression)e.Operand;
            // prefer negation for parameters or constants
            if(binExpr.Left is ConstantExpression || IsParameter(binExpr.Left)) {
              return Visit(binExpr.Update(Expression.Negate(binExpr.Left), null, binExpr.Right));
            } else return Visit(binExpr.Update(binExpr.Left, null, Expression.Negate(binExpr.Right)));
          }),
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
        // abs(-(x)) -> abs(x)
        new MethodCallExpressionRule(
          e => e.Method == abs && e.Arguments[0].NodeType == ExpressionType.Negate,
          e => e.Update(e.Object, new [] {((UnaryExpression) e.Arguments[0]).Operand})
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
          ),
        // pow(pow(a, x), y) -> pow(a, x * y)
        new MethodCallExpressionRule(
          e => e.Method == pow && e.Arguments[0] is MethodCallExpression callExpr && callExpr.Method == pow,
          e => {
            var inner = (MethodCallExpression)e.Arguments[0];
            return Visit(e.Update(e.Object, new [] {inner.Arguments[0], Expression.Multiply(e.Arguments[1], inner.Arguments[1]) }));
              }
          ),
        // exp(x)^c = exp(c*x)
        // exp(x)^p = exp(p*x)
        new MethodCallExpressionRule(
          e => e.Method == pow && e.Arguments[0] is MethodCallExpression inner && inner.Method == exp,
          e => {
            var inner = (MethodCallExpression)e.Arguments[0];
            return Visit(inner.Update(inner.Object, new [] {Expression.Multiply(inner.Arguments[0], e.Arguments[1]) }));
          })
      });
    }

    private void AddReparameterizationRules() {
      binaryRules.AddRange(new[] {
      // param ° param -> param
      new BinaryExpressionRule(
        e => IsParameter(e.Left) && IsParameter(e.Right),
        e => NewParameter(Apply(e.NodeType, GetParameterValue(e.Left), GetParameterValue(e.Right)))
        ),
      // (a ° param) ° param
      new BinaryExpressionRule(
        e => IsAssociative(e) && e.Left.NodeType == e.NodeType && e.Left is BinaryExpression leftExpr
        && IsParameter(leftExpr.Right) && IsParameter(e.Right),
        e => {
          var leftExpr = (BinaryExpression)e.Left;
          return Visit(e.Update(leftExpr.Left, null,
            NewParameter(Apply(e.NodeType, GetParameterValue(leftExpr.Right), GetParameterValue(e.Right)))));
            }
        ),
      // param ° const -> param
      new BinaryExpressionRule(
        e => IsParameter(e.Left) && IsConstant(e.Right),
        e => NewParameter(Apply(e.NodeType, GetParameterValue(e.Left), GetConstantValue(e.Right)))
        ),
      // (a ° param) ° const (TODO: combine param / const rules)
      new BinaryExpressionRule(
        e => IsAssociative(e) && e.Left.NodeType == e.NodeType && e.Left is BinaryExpression leftExpr
        && IsParameter(leftExpr.Right) && IsConstant(e.Right),
        e => {
          var leftExpr = (BinaryExpression)e.Left;
          return Visit(e.Update(leftExpr.Left, null,
            NewParameter(Apply(e.NodeType, GetParameterValue(leftExpr.Right), GetConstantValue(e.Right)))));
            }
        ),
      // const ° param -> param
      new BinaryExpressionRule(
        e => IsConstant(e.Left) && IsParameter(e.Right),
        e => NewParameter(Apply(e.NodeType, GetConstantValue(e.Left), GetParameterValue(e.Right)))
        ),
      // (a ° const) ° param (TODO: combine rules param/const)
      new BinaryExpressionRule(
        e => IsAssociative(e) && e.Left.NodeType == e.NodeType && e.Left is BinaryExpression leftExpr
        && IsConstant(leftExpr.Right) && IsParameter(e.Right),
        e => {
          var leftExpr = (BinaryExpression)e.Left;
          return Visit(e.Update(leftExpr.Left, null,
            NewParameter(Apply(e.NodeType, GetConstantValue(leftExpr.Right), GetParameterValue(e.Right)))));
            }
        ),

       // x - p -> x + (-p)
       // TODO (a (+/-) x) - p
       new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Subtract && IsParameter(e.Right),
        e => Visit(Expression.Add(e.Left, NewParameter(-GetParameterValue(e.Right))))
        ),
       // x / p -> x * (1/p)
       // TODO (a * x) / p -> (a * x) * (1/p)
       new BinaryExpressionRule(
        e => e.NodeType == ExpressionType.Divide && IsParameter(e.Right),
        e => Visit(Expression.Multiply(e.Left, NewParameter(1.0 / GetParameterValue(e.Right))))
        ),
      });

      unaryRules.AddRange(new[] {
        // -(p) -> p
        new UnaryExpressionRule(
          e => e.NodeType == ExpressionType.Negate && IsParameter(e.Operand),
          e => NewParameter(-GetParameterValue(e.Operand))
          ),
      });

      callRules.AddRange(new[] {
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

      // TODO: delete FoldParametersVisitor
      });

    }



    public static ParameterizedExpression Simplify(ParameterizedExpression expr) {
      if (!CheckExprVisitor.CheckValid(expr.expr)) throw new NotSupportedException();

      var v = new RuleBasedSimplificationVisitor(expr.p, expr.pValues);
      var body = v.Visit(expr.expr.Body);
      return new ParameterizedExpression(expr.expr.Update(body, expr.expr.Parameters), expr.p, v.pValues.ToArray());
    }

    public static ParameterizedExpression SimplifyWithoutReparameterization(ParameterizedExpression expr) {
      // This is necessary for simplification where it is not allowed to re-parameterize the expression.
      // For example, when simplifying the derivative of an expression.
      if (!CheckExprVisitor.CheckValid(expr.expr)) throw new NotSupportedException();

      var v = new RuleBasedSimplificationVisitor(expr.p, expr.pValues);
      v.ClearRules();
      v.AddConstantFoldingRules();
      v.AddBasicRules();
      var body = v.Visit(expr.expr.Body);
      return new ParameterizedExpression(expr.expr.Update(body, expr.expr.Parameters), expr.p, v.pValues.ToArray());
    }

    public static ParameterizedExpression FoldConstants(ParameterizedExpression expr) {
      // This uses only a reduced set of rules to fold constants and optimize operations without effect (*0, *1, +/-0 ...)
      // This is necessary for simplification where it is not allowed to re-parameterize the expression.
      // For example, when simplifying the derivative of an expression.
      if (!CheckExprVisitor.CheckValid(expr.expr)) throw new NotSupportedException();

      var v = new RuleBasedSimplificationVisitor(expr.p, expr.pValues);
      v.ClearRules();
      v.AddConstantFoldingRules();
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


    private int Compare(Expression left, Expression right) {
      // left < right -> -1
      // left = right -> 0
      // left > right -> 1
      // order: terms < parameters < constants
      // terms are ordered by expression type (binary before unary before call).
      // within the same type expressions are ordered by size
      // order of parameters and constants is irrelevant
      var typeCmp = CompareType(left, right);
      if (typeCmp != 0) return typeCmp;
      else {
        // same type: compare by size (larger expressions first)
        return CountNodesVisitor.Count(right) - CountNodesVisitor.Count(left);
      }
    }

    private int CompareType(Expression left, Expression right) {

      var leftOrd = OrdinalNumber(left);
      var rightOrd = OrdinalNumber(right);
      return leftOrd.CompareTo(rightOrd);
    }

    private int OrdinalNumber(Expression e) {
      return IsConstant(e) ? 5
        : IsParameter(e) ? 4
        : e is MethodCallExpression ? 3
        : e is UnaryExpression ? 2
        : e is BinaryExpression ? 1
        : throw new NotSupportedException();
    }

    private bool IsCommutative(BinaryExpression e) => e.NodeType == ExpressionType.Add || e.NodeType == ExpressionType.Multiply;

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
    private bool IsParameterOrConstant(Expression expr) => IsConstant(expr) || IsParameter(expr);

    private bool HasParameters(Expression left) => CountParametersVisitor.Count(left, p) > 0;

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

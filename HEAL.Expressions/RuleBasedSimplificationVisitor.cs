﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using BinaryExpressionRule = System.ValueTuple<string, System.Func<System.Linq.Expressions.BinaryExpression, bool>, System.Func<System.Linq.Expressions.BinaryExpression, System.Linq.Expressions.Expression>>;
using UnaryExpressionRule = System.ValueTuple<string, System.Func<System.Linq.Expressions.UnaryExpression, bool>, System.Func<System.Linq.Expressions.UnaryExpression, System.Linq.Expressions.Expression>>;
using MethodCallExpressionRule = System.ValueTuple<string, System.Func<System.Linq.Expressions.MethodCallExpression, bool>, System.Func<System.Linq.Expressions.MethodCallExpression, System.Linq.Expressions.Expression>>;
using System.Reflection;

namespace HEAL.Expressions {
  public class RuleBasedSimplificationVisitor : ExpressionVisitor {
    private readonly List<(string Description, Func<BinaryExpression, bool> Match, Func<BinaryExpression, Expression> Apply)> binaryRules = new List<BinaryExpressionRule>();
    private readonly List<(string Description, Func<UnaryExpression, bool> Match, Func<UnaryExpression, Expression> Apply)> unaryRules = new List<UnaryExpressionRule>();
    private readonly List<(string Description, Func<MethodCallExpression, bool> Match, Func<MethodCallExpression, Expression> Apply)> callRules = new List<MethodCallExpressionRule>();
    private ParameterExpression p;
    private List<double> pValues;

    private readonly List<(string rule, Expression expr)> matchedRules = new List<(string rule, Expression expr)>(); // for debugging which rules are actually used

    private readonly Dictionary<string, Expression> visitCache = new Dictionary<string, Expression>(); // for memoization of Visit() methods

    private RuleBasedSimplificationVisitor(ParameterExpression p, double[] pValues, bool debugRules = false) {
      this.p = p;
      this.pValues = pValues.ToList();
      this.debugRules = debugRules;

      AddConstantFoldingRules();
      AddBasicRules();
      AddReparameterizationRules();
    }

    private void ClearRules() {
      binaryRules.Clear();
      unaryRules.Clear();
      callRules.Clear();
      visitCache.Clear();
    }

    private void AddConstantFoldingRules() {
      visitCache.Clear();

      binaryRules.AddRange(new[] {
      new BinaryExpressionRule(
        "const ° const -> const",
        e => IsConstant(e.Left) && IsConstant(e.Right),
        e => Expression.Constant(Apply(e.NodeType, GetConstantValue(e.Left), GetConstantValue(e.Right)))
        ),
      new BinaryExpressionRule(
        "(a ° const) ° const",
        e => IsAssociative(e) && e.Left.NodeType == e.NodeType && e.Left is BinaryExpression leftExpr
        && IsConstant(leftExpr.Right) && IsConstant(e.Right),
        e => {
          var leftExpr = (BinaryExpression)e.Left;
          return Visit(e.Update(leftExpr.Left, null,
            Expression.Constant(Apply(e.NodeType, GetConstantValue(leftExpr.Right), GetConstantValue(e.Right)))));
            }
        ),

      // move constant right for commutative expressions
      new BinaryExpressionRule(
        "c ° x -> x ° c | x is not constant",
        e => IsCommutative(e)
          && IsConstant(e.Left) && !IsConstant(e.Right),
        e => {
            return Visit(e.Update(e.Right, null, e.Left));
          }
        ),
      new BinaryExpressionRule(
       "x (+/-) 0.0",
        e => (e.NodeType == ExpressionType.Add || e.NodeType == ExpressionType.Subtract)
        && e.Right is ConstantExpression constExpr && (double)constExpr.Value == 0.0,
        e => e.Left
        ),
      new BinaryExpressionRule(
        "0.0 - x",
        e => e.NodeType == ExpressionType.Subtract
        && e.Left is ConstantExpression constExpr && (double)constExpr.Value == 0.0,
        e => Visit(Expression.Negate(e.Right))
        ),
      new BinaryExpressionRule(
        "x * 0.0",
        e => e.NodeType == ExpressionType.Multiply
        && e.Right is ConstantExpression constExpr && (double)constExpr.Value == 0.0,
        e => e.Right
        ),
      new BinaryExpressionRule(
        "x * 1.0",
        e => e.NodeType == ExpressionType.Multiply
        && e.Right is ConstantExpression constExpr && (double)constExpr.Value == 1.0,
        e => e.Left
        ),
      new BinaryExpressionRule(
        "x * -1.0",
        e => e.NodeType == ExpressionType.Multiply
        && e.Right is ConstantExpression constExpr && (double)constExpr.Value == -1.0,
        e => Visit(Expression.Negate(e.Left))
        ),
      new BinaryExpressionRule(
        "x / 1.0",
        e => e.NodeType == ExpressionType.Divide
        && e.Right is ConstantExpression constExpr && (double)constExpr.Value == 1.0,
        e => e.Left
        ),
      new BinaryExpressionRule(
        "x / 0.0",
        e => e.NodeType == ExpressionType.Divide
        && e.Right is ConstantExpression constExpr && (double)constExpr.Value == 0.0,
        e => Expression.Constant(double.NaN)
        ),
      new BinaryExpressionRule(
        "x / -1.0",
        e => e.NodeType == ExpressionType.Divide
        && e.Right is ConstantExpression constExpr && (double)constExpr.Value == -1.0,
        e => Visit(Expression.Negate(e.Left))
        ),
      // this is not strictly true because 0/0 is undefined but such exceptions are not relevant to us in useful models
      new BinaryExpressionRule(
        "0 / x -> 0",
        e => e.NodeType == ExpressionType.Divide
        && e.Left is ConstantExpression constExpr && (double)constExpr.Value == 0.0,
        e => e.Left // 0.0
        ),
      new BinaryExpressionRule(
        "x / c -> x * c",
        e => e.NodeType == ExpressionType.Divide && IsConstant(e.Right),
        e => Expression.Multiply(e.Left, Expression.Constant(1.0 / GetConstantValue(e.Right)))
        ),
      new BinaryExpressionRule(
        "(x * c) + x -> (x * (c+1))",
        e => e.NodeType == ExpressionType.Add
          && e.Left is BinaryExpression binExpr && e.Left.NodeType == ExpressionType.Multiply && IsConstant(binExpr.Right)
          && binExpr.Left.ToString() == e.Right.ToString(),
        e => {
          var binExpr = (BinaryExpression)e.Left;
          return Visit(binExpr.Update(binExpr.Left, null, Expression.Constant(GetConstantValue(binExpr.Right) + 1)));
        }),
      new BinaryExpressionRule(
        "(x * c) - x -> (x * (c-1))",
        e => e.NodeType == ExpressionType.Subtract
          && e.Left is BinaryExpression binExpr && IsConstant(binExpr.Right)
          && binExpr.Left.ToString() == e.Right.ToString(),
        e => {
          var binExpr = (BinaryExpression)e.Left;
          return Visit(binExpr.Update(binExpr.Left, null, Expression.Constant(GetConstantValue(binExpr.Right) - 1)));
        }),
      new BinaryExpressionRule(
        "(x * c) / x -> c",
        e => e.NodeType == ExpressionType.Divide
          && e.Left is BinaryExpression binExpr && IsConstant(binExpr.Right)
          && binExpr.Left.ToString() == e.Right.ToString(),
        e => {
          var binExpr = (BinaryExpression)e.Left;
          return binExpr.Right;
        }),
      new BinaryExpressionRule(
        "x / (x * c) -> c",
        e => e.NodeType == ExpressionType.Divide
          && e.Right is BinaryExpression binExpr && binExpr.NodeType  == ExpressionType.Multiply && IsConstant(binExpr.Right)
          && e.Left.ToString() == binExpr.Left.ToString(),
        e => {
          var binExpr = (BinaryExpression)e.Right;
          return Expression.Constant(1.0 / GetConstantValue(binExpr.Right));
        }),

      new BinaryExpressionRule(
        "c / (x * c) -> c / x",
        e => e.NodeType == ExpressionType.Divide && IsConstant(e.Left)
          && e.Right.NodeType == ExpressionType.Multiply && e.Right is BinaryExpression right
          && IsConstant(right.Right),
        e => {
          var right = (BinaryExpression)e.Right;
          return e.Update(Expression.Constant(GetConstantValue(e.Left) / GetConstantValue(right.Right)), null, right.Left);
        }),
            new BinaryExpressionRule(
        "z / (x * c) -> (z*c) / x",
        e => e.NodeType == ExpressionType.Divide
          && e.Right.NodeType == ExpressionType.Multiply && e.Right is BinaryExpression right
          && IsConstant(right.Right),
        e => {
          var right = (BinaryExpression)e.Right;
          return Visit(Expression.Divide(
            Expression.Multiply(e.Left, Expression.Constant(1.0 / GetConstantValue(right.Right))),
            right.Left
            ));
        }),

      new BinaryExpressionRule(
         "x - (y * c) -> x + (y * c)",
         e => e.NodeType == ExpressionType.Subtract
           && e.Right is BinaryExpression rightBin && rightBin.NodeType == ExpressionType.Multiply && IsConstant(rightBin.Right),
         e => {
           var rightBin = (BinaryExpression)e.Right;
           return Visit(Expression.Add(e.Left, Expression.Multiply(rightBin.Left, Expression.Constant(-GetConstantValue(rightBin.Right)))));
         }),

      // prefer - 1/x over + (-1/x)  (1/x == inv(x) with len==2)
      new BinaryExpressionRule(
        "a + (-1 / x) -> a - (1 / x)",
        e => e.NodeType == ExpressionType.Add
             && e.Right.NodeType == ExpressionType.Divide && e.Right is BinaryExpression binExpr
             && IsConstant(binExpr.Left) && GetConstantValue(binExpr.Left) == -1.0,
        e => {
          var right = (BinaryExpression)e.Right;
          return Visit(Expression.Subtract(e.Left, Expression.Divide(Expression.Constant(1.0), right.Right)));
          }
        ),
      new BinaryExpressionRule(
        "x + -c -> x - c  (negative constant)",
        e => e.NodeType == ExpressionType.Add && e.Right is ConstantExpression constExpr && (double)constExpr.Value < 0.0,
        e => Visit(Expression.Subtract(e.Left, Expression.Constant(- (double)((ConstantExpression)e.Right).Value)))
        ),
      new BinaryExpressionRule(
        "-a * c -> a * -c",
        e => e.NodeType == ExpressionType.Multiply && e.Left.NodeType == ExpressionType.Negate && IsConstant(e.Right),
        e => {
           var unary = (UnaryExpression)e.Left;
           return Visit(Expression.Multiply(unary.Operand, Expression.Constant(-GetConstantValue(e.Right))));
          }
        ),
      });

      unaryRules.AddRange(new[] {
        // 
        new UnaryExpressionRule(
          "+x -> x",
          e => e.NodeType == ExpressionType.UnaryPlus,
          e => e.Operand
          ),
        // 
        new UnaryExpressionRule(
          "-(const) -> -const",
          e => e.NodeType == ExpressionType.Negate && IsConstant(e.Operand),
          e => Expression.Constant(-GetConstantValue(e.Operand))
          ),
      });

      callRules.AddRange(new[] {
        // 
        new MethodCallExpressionRule(
          "fold constant arguments in method calls",
          e => e.Arguments.All(IsConstant),
          e => Expression.Constant(e.Method.Invoke(e.Object, e.Arguments.Select(GetConstantValue).OfType<object>().ToArray()))
          ),
        // 
        new MethodCallExpressionRule(
          "pow(x, 0) -> 1",
          e => IsPower(e.Method) && IsConstant(e.Arguments[1]) && GetConstantValue(e.Arguments[1]) == 0.0,
          e => Expression.Constant(1.0)
          ),
        // 
        // (this is not 100% correct because 0^0 is not defined. But for our purposes 0^0 is irrelevant
        new MethodCallExpressionRule(
          "pow(0, x) -> 0",
          e => IsPower(e.Method) && IsConstant(e.Arguments[0]) && GetConstantValue(e.Arguments[0]) == 0.0,
          e => Expression.Constant(0.0)
          ),
  
        // 
        new MethodCallExpressionRule(
          "pow(x, 1) -> x",
          e => IsPower(e.Method) && IsConstant(e.Arguments[1]) && GetConstantValue(e.Arguments[1]) == 1.0,
          e => e.Arguments[0]
          ),
        // 
        new MethodCallExpressionRule(
          "pow(1, x) -> 1",
          e => IsPower(e.Method) && IsConstant(e.Arguments[0]) && GetConstantValue(e.Arguments[0]) == 1.0,
          e => Expression.Constant(1.0)
          ),
        new MethodCallExpressionRule(
          "pow(x, -1) -> 1 / x",
          e => IsPower(e.Method) && IsConstant(e.Arguments[1]) && GetConstantValue(e.Arguments[1]) == -1.0,
          e => Visit(Expression.Divide(Expression.Constant(1.0), e.Arguments[0]))
          ),

        // 
        new MethodCallExpressionRule(
          "exp(x + c) -> exp(x) * c",
          e => e.Method == exp
             && e.Arguments[0] is BinaryExpression binExpr && binExpr.NodeType == ExpressionType.Add
             && IsConstant(binExpr.Right),
          e => {
            var binExpr = (BinaryExpression)e.Arguments[0];
            var offset = GetConstantValue(binExpr.Right);
            return Visit(Expression.Multiply(Expression.Call(exp, new [] { binExpr.Left}), Expression.Constant(Math.Exp(offset))));
          }),
      });
    }


    private void AddBasicRules() {
      visitCache.Clear();

      binaryRules.AddRange(new[] {
      new BinaryExpressionRule(
        "x / x -> 1.0",
        e => e.NodeType == ExpressionType.Divide && e.Left.ToString() == e.Right.ToString(),
        e => Expression.Constant(1.0)
        ),
      new BinaryExpressionRule(
        "(a * x) / x -> a",
        e => e.NodeType == ExpressionType.Divide && e.Left is BinaryExpression left &&
        left.NodeType == ExpressionType.Multiply && left.Right.ToString() == e.Right.ToString(),
        e => ((BinaryExpression)e.Left).Left
        ),
      new BinaryExpressionRule(
        "(a / x) * x -> a",
        e => e.NodeType == ExpressionType.Multiply && e.Left is BinaryExpression left &&
        left.NodeType == ExpressionType.Divide && left.Right.ToString() == e.Right.ToString(),
        e => ((BinaryExpression)e.Left).Left
        ),

      new BinaryExpressionRule(
        "x - x -> 0.0",
        e => e.NodeType == ExpressionType.Subtract && e.Left.ToString() == e.Right.ToString(),
        e => Expression.Constant(0.0)
        ),
      new BinaryExpressionRule(
        "(a + x) - x -> a",
        e => e.NodeType == ExpressionType.Subtract && e.Left.NodeType == ExpressionType.Add &&
             e.Left is BinaryExpression binExpr && binExpr.Right.ToString() == e.Right.ToString(),
        e => ((BinaryExpression)e.Left).Left
        ),
      new BinaryExpressionRule(
        "(a - x) + x -> a",
        e => e.NodeType == ExpressionType.Add && e.Left.NodeType == ExpressionType.Subtract &&
             e.Left is BinaryExpression binExpr && binExpr.Right.ToString() == e.Right.ToString(),
        e => ((BinaryExpression)e.Left).Left
        ),

      new BinaryExpressionRule(
        "x + x -> x * 2",
        e => e.NodeType == ExpressionType.Add && e.Left.ToString() == e.Right.ToString(),
        e => Visit(Expression.Multiply(e.Left, Expression.Constant(2.0)))
        ),
      new BinaryExpressionRule(
        "(a + x) + x -> a + x * 2",
        e => e.NodeType == ExpressionType.Add
          && e.Left is BinaryExpression left && left.NodeType == ExpressionType.Add
          && left.Right.ToString() == e.Right.ToString(),
        e => {
          var left = (BinaryExpression)e.Left;
          return Visit(left.Update(left.Left, null, Expression.Multiply(left.Right, Expression.Constant(2.0))));
          }
        ),
      new BinaryExpressionRule(
        "(a - x) - x -> a - x * 2",
        e => e.NodeType == ExpressionType.Subtract
          && e.Left is BinaryExpression left && left.NodeType == ExpressionType.Subtract
          && left.Right.ToString() == e.Right.ToString(),
        e => {
          var left = (BinaryExpression)e.Left;
          return Visit(left.Update(left.Left, null, Expression.Multiply(left.Right, Expression.Constant(2.0))));
          }
        ),

      new BinaryExpressionRule(
        "x * a + x  -> x * (a + 1)",
        e => e.NodeType == ExpressionType.Add && e.Left is BinaryExpression left
          && left.NodeType == ExpressionType.Multiply && left.Left.ToString() == e.Right.ToString(),
        e => {
          var left = (BinaryExpression)e.Left;
          return Visit(Expression.Multiply(left.Left, Expression.Add(left.Right, Expression.Constant(1.0))));
          }
        ),
      new BinaryExpressionRule(
        "x * a - x  -> x * (a - 1)",
        e => e.NodeType == ExpressionType.Subtract && e.Left is BinaryExpression left
          && left.NodeType == ExpressionType.Multiply && left.Left.ToString() == e.Right.ToString(),
        e => {
          var left = (BinaryExpression)e.Left;
          return Visit(Expression.Multiply(left.Left, Expression.Subtract(left.Right, Expression.Constant(1.0))));
          }
        ),

      new BinaryExpressionRule(
        "(x * a) / x  -> a",
        e => e.NodeType == ExpressionType.Divide && e.Left is BinaryExpression left
          && left.NodeType == ExpressionType.Multiply && left.Left.ToString() == e.Right.ToString(),
        e => {
          var left = (BinaryExpression)e.Left;
          return left.Right;
          }
        ),



      new BinaryExpressionRule(
        "x * alpha + x * beta -> x * (alpha + beta)",
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
        "x * alpha - x * beta -> x * (alpha - beta)",
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

      // 
      new BinaryExpressionRule(
        "x * x -> pow(x, 2)",
        e => e.NodeType == ExpressionType.Multiply && e.Left.ToString() == e.Right.ToString(),
        e => Visit(Expression.Call(pow, e.Left, Expression.Constant(2.0)))
        ),
      new BinaryExpressionRule(
        "(a * x) * x -> a * pow(x, 2)",
        e => e.NodeType == ExpressionType.Multiply && e.Left is BinaryExpression left
          && left.NodeType == ExpressionType.Multiply && left.Right.ToString() == e.Right.ToString(),
        e => {
          var left = (BinaryExpression)e.Left;
          return Visit(Expression.Multiply(left.Left, Expression.Call(pow, e.Right, Expression.Constant(2.0))));
          }
        ),

      // 
      new BinaryExpressionRule(
        "(x/y) * z -> (x*z) / y | z is no parameter",
        e => e.NodeType == ExpressionType.Multiply && e.Left.NodeType == ExpressionType.Divide && !IsParameter(e.Right),
        e => {
          var binExpr = (BinaryExpression)e.Left;
          return Visit(Expression.Divide(Expression.Multiply(binExpr.Left, e.Right), binExpr.Right));
          }
        ),
      //
      new BinaryExpressionRule(
        "x * (y/z) -> (x*y) / z",
        e => e.NodeType == ExpressionType.Multiply && e.Right.NodeType == ExpressionType.Divide,
        e => {
          var binExpr = (BinaryExpression)e.Right;
          return Visit(Expression.Divide(Expression.Multiply(e.Left, binExpr.Left), binExpr.Right));
          }
        ),
      // 
      new BinaryExpressionRule(
        "x + (-y) -> x - y ",
        e => e.NodeType == ExpressionType.Add && e.Right.NodeType == ExpressionType.Negate,
        e => Visit(Expression.Subtract(e.Left, ((UnaryExpression)e.Right).Operand))
        ),
      new BinaryExpressionRule(
          "(-x - y) * p -> (x + y) * p'",
          e => e.NodeType == ExpressionType.Multiply && IsParameterOrConstant(e.Right)
            && e.Left is BinaryExpression left && left.NodeType == ExpressionType.Subtract
            && left.Left.NodeType == ExpressionType.Negate,
          e => {
              var left = (BinaryExpression)e.Left;
              var unary = (UnaryExpression)left.Left;
              return Visit(Expression.Multiply(
                    Expression.Add(unary.Operand, left.Right),
                    Expression.Multiply(e.Right, Expression.Constant(-1.0))
                  ));
          }
        ),

      new BinaryExpressionRule(
        "x - (-y) -> x + y",
        e => e.NodeType == ExpressionType.Subtract && e.Right.NodeType == ExpressionType.Negate,
        e => Expression.Add(e.Left, ((UnaryExpression)e.Right).Operand)
      ),      
      // 
      new BinaryExpressionRule(
        "(a + x) + (-(x))",
        e => e.NodeType == ExpressionType.Add && e.Left.NodeType == ExpressionType.Add
             && e.Right.NodeType == ExpressionType.Negate
             && e.Left is BinaryExpression left
             && e.Right is UnaryExpression negExpr && negExpr.Operand.ToString() == left.Right.ToString(),
        e => {
          var leftBin = (BinaryExpression)e.Left;
          return leftBin.Left;
          }
        ),


      // 
      // This implicitly handles (a ° b) ° (c ° d) -> ((a ° b) ° c) ° d
      new BinaryExpressionRule(
        "a ° (b ° c) -> (a ° b) ° c",
        e => IsAssociative(e) && e.NodeType == e.Right.NodeType,
        e => {
          var rightExpr = (BinaryExpression)e.Right;
          return Visit(rightExpr.Update(e.Update(e.Left, null, rightExpr.Left), null, rightExpr.Right));
        }
        ),
      
      // sort arguments in (+,*) operations
      // 
      new BinaryExpressionRule(
        "(a ° c) ° b -> (a ° b) ° c   (with compare(b,c) <= 0)",
        e => IsCommutative(e) && IsAssociative(e) && e.Left.NodeType == e.NodeType &&
        e.Left is BinaryExpression left && Compare(left.Right, e.Right) > 0,
        e => {
          var left = (BinaryExpression)e.Left;
          return Visit(e.Update(left.Update(left.Left, null, e.Right), null, left.Right));
        }
        ),
      // 
      new BinaryExpressionRule(
        "c ° b -> b ° c   (with compare(b,c) <= 0)",
        e => IsCommutative(e) && Compare(e.Left, e.Right) > 0,
        e => {
          return Visit(e.Update(e.Right, null, e.Left));
        }
        ),
      

      // 
      new BinaryExpressionRule(
        "(a / b) / (c / d) -->  (a * d) / (b * c)",
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
      // 
      new BinaryExpressionRule(
        "(a / b) / c -->  a / (b * c)",
        e => e.NodeType == ExpressionType.Divide
          && e.Left.NodeType == ExpressionType.Divide,
        e => {
          var left = (BinaryExpression)e.Left;
          return Expression.Divide(
            left.Left,
            Expression.Multiply(left.Right, e.Right));
        }
        ),
      // 
      new BinaryExpressionRule(
        "a / (b / c) -->  (a * c) / b",
        e => e.NodeType == ExpressionType.Divide
          && e.Right.NodeType == ExpressionType.Divide,
        e => {
          var right = (BinaryExpression)e.Right;
          return Expression.Divide(
            Expression.Multiply(e.Left, right.Right),
            right.Left);
        }
        ),
      // 
      new BinaryExpressionRule(
        "x * pow(x, z) -> pow(x, z+1)",
        e => e.NodeType == ExpressionType.Multiply && e.Right is MethodCallExpression callExpr && callExpr.Method == pow
        && e.Left.ToString() == callExpr.Arguments[0].ToString(),
        e => {
          var callExpr = (MethodCallExpression)e.Right;
          return Visit(callExpr.Update(callExpr.Object,
            new [] {callExpr.Arguments[0],
              Expression.Add(callExpr.Arguments[1], Expression.Constant(1.0)) }));
        }
        ),
      new BinaryExpressionRule(
        "(a * x) * pow(x, z) -> a * pow(x, z+1)",
        e => e.NodeType == ExpressionType.Multiply && e.Right is MethodCallExpression callExpr && callExpr.Method == pow
           && e.Left.NodeType == ExpressionType.Multiply && e.Left is BinaryExpression left
           && left.Right.ToString() == callExpr.Arguments[0].ToString(),
        e => {
          var callExpr = (MethodCallExpression)e.Right;
          var left = (BinaryExpression)e.Left;
          return Visit(Expression.Multiply(left.Left, callExpr.Update(callExpr.Object,
            new [] {callExpr.Arguments[0],
              Expression.Add(callExpr.Arguments[1], Expression.Constant(1.0)) })));
        }
        ),
      new BinaryExpressionRule(
        "pow(x, z) * x -> pow(x, z+1)",
        e => e.NodeType == ExpressionType.Multiply && e.Left is MethodCallExpression callExpr && callExpr.Method == pow
        && callExpr.Arguments[0].ToString() == e.Right.ToString(),
        e => {
          var callExpr = (MethodCallExpression)e.Left;
          return Visit(callExpr.Update(callExpr.Object,
            new [] {callExpr.Arguments[0],
              Expression.Add(callExpr.Arguments[1], Expression.Constant(1.0)) }));
        }
        ),
      // 
      new BinaryExpressionRule(
        "pow(x, z) / x -> pow(x, z-1)",
        e => e.NodeType == ExpressionType.Divide && e.Left is MethodCallExpression callExpr && callExpr.Method == pow
        && callExpr.Arguments[0].ToString() == e.Right.ToString(),
        e => {
          var callExpr = (MethodCallExpression)e.Left;
          return Visit(callExpr.Update(callExpr.Object,
            new [] {callExpr.Arguments[0],
              Expression.Subtract(callExpr.Arguments[1], Expression.Constant(1.0))}));
        }),
      // 
      new BinaryExpressionRule(
        "x / pow(x, z) -> pow(x, 1 - z)",
        e => e.NodeType == ExpressionType.Divide && e.Right is MethodCallExpression callExpr && callExpr.Method == pow
             && e.Left.ToString() == callExpr.Arguments[0].ToString(),
        e => {
          var callExpr = (MethodCallExpression)e.Right;
          return Visit(callExpr.Update(callExpr.Object,
            new [] {callExpr.Arguments[0],
              Expression.Subtract(Expression.Constant(1.0), callExpr.Arguments[1]) }));
        }
        ),
      // nest associative left 
      // DUPLICATE
      // new BinaryExpressionRule(
      //   "a ° (b ° c) -> (a ° b) ° c",
      //   e => IsAssociative(e) && e.Right is BinaryExpression rightBinExpr && rightBinExpr.NodeType == e.NodeType,
      //   e => {
      //     var rightBin = (BinaryExpression)e.Right;
      //     return Visit(rightBin.Update(e.Update(e.Left, null, rightBin.Left), null, rightBin.Right));
      //     }
      //   ),
       // 
       //new BinaryExpressionRule(
       //  "p / x -> 1/x * p", // this makes the expression longer but is necessary together with other rules
       // e => e.NodeType == ExpressionType.Divide && IsParameter(e.Left) && !IsParameter(e.Right),
       // e => Visit(Expression.Multiply(Expression.Divide(Expression.Constant(1.0), e.Right), e.Left))
       // ),
       new BinaryExpressionRule(
         "1/pow(x,y) -> pow(1/x, y)",
         e => e.NodeType == ExpressionType.Divide
           && e.Left is ConstantExpression constExpr && GetConstantValue(e.Left) == 1.0
           && e.Right is MethodCallExpression callExpr && IsPower(callExpr.Method),
           // && IsParameterOrConstant(callExpr.Arguments[0]),
         e => {
           var callExpr = (MethodCallExpression)e.Right;
            return Visit(callExpr.Update(callExpr.Object, new [] { Expression.Divide(Expression.Constant(1.0), callExpr.Arguments[0]), callExpr.Arguments[1] }));
           }
         ),
       new BinaryExpressionRule(
         "pow(x, -y) * z -> z / pow(x, y)",
         e => e.NodeType == ExpressionType.Multiply && e.Left is MethodCallExpression callExpr && IsPower(callExpr.Method)
           && callExpr.Arguments[1].NodeType == ExpressionType.Negate,
         e => {
           var callExpr = (MethodCallExpression)e.Left;
           var exponent = (UnaryExpression)callExpr.Arguments[1];
           return Expression.Divide(e.Right, callExpr.Update(callExpr.Object, new [] { callExpr.Arguments[0], exponent.Operand }));
             }
         ),
       new BinaryExpressionRule(
         "z/pow(x, a) -> z * pow(x, -a)) where a is parameter or constant",
        e => e.NodeType == ExpressionType.Divide && e.Right is MethodCallExpression callExpr && IsPower(callExpr.Method) && IsParameterOrConstant(callExpr.Arguments[1]),
        e => {
          var callExpr = (MethodCallExpression)e.Right;
          return Visit(Expression.Multiply(e.Left, callExpr.Update(callExpr.Object, new [] {callExpr.Arguments[0], Expression.Negate(callExpr.Arguments[1]) })));
            }
        ),
       // 
       new BinaryExpressionRule(
         "exp(x) * exp(y) -> exp(x + y)",
        e => e.NodeType == ExpressionType.Multiply
           && e.Left is MethodCallExpression leftCall && leftCall.Method == exp
           && e.Right is MethodCallExpression rightCall && rightCall.Method == exp,
        e => {
          var leftCall = (MethodCallExpression)e.Left;
          var rightCall = (MethodCallExpression)e.Right;
          return Visit(leftCall.Update(leftCall.Object, new [] { Expression.Add(leftCall.Arguments[0], rightCall.Arguments[0]) }));
        }),
       // 
       new BinaryExpressionRule(
         "exp(x) / exp(y) -> exp(x - y)",
        e => e.NodeType == ExpressionType.Divide
           && e.Left is MethodCallExpression leftCall && leftCall.Method == exp
           && e.Right is MethodCallExpression rightCall && rightCall.Method == exp,
        e => {
          var leftCall = (MethodCallExpression)e.Left;
          var rightCall = (MethodCallExpression)e.Right;
          return Visit(leftCall.Update(leftCall.Object, new [] { Expression.Subtract(leftCall.Arguments[0], rightCall.Arguments[0]) }));
        }),
      });

      unaryRules.AddRange(new[] {
          /// 
        new UnaryExpressionRule(
          "-(a + b) -> -a + -b",
          e => e.NodeType == ExpressionType.Negate && e.Operand.NodeType == ExpressionType.Add,
          e => {
            var binExpr = (BinaryExpression)e.Operand;
            return Visit(binExpr.Update(Expression.Negate(binExpr.Left), null, Expression.Negate(binExpr.Right)));
              }
          ),
          /// 
        new UnaryExpressionRule(
          "-(a - b) -> b - a",
          e => e.NodeType == ExpressionType.Negate && e.Operand.NodeType == ExpressionType.Subtract,
          e => {
            var binExpr = (BinaryExpression)e.Operand;
            return Visit(binExpr.Update(binExpr.Right, null, binExpr.Left));
              }
          ),
          /// 
        new UnaryExpressionRule(
          "-(a * b) -> a * -b",
          e => e.NodeType == ExpressionType.Negate && e.Operand.NodeType == ExpressionType.Multiply,
          e => {
            var binExpr = (BinaryExpression)e.Operand;
            // prefer negation for parameters or constants
            if(binExpr.Left is ConstantExpression || IsParameter(binExpr.Left)) {
              return Visit(binExpr.Update(Expression.Negate(binExpr.Left), null, binExpr.Right));
            } else return Visit(binExpr.Update(binExpr.Left, null, Expression.Negate(binExpr.Right)));
              }
          ),
          /// 
        new UnaryExpressionRule(
          "-(a / b) -> a / -b",
          e => e.NodeType == ExpressionType.Negate && e.Operand.NodeType == ExpressionType.Divide,
          e => {
            var binExpr = (BinaryExpression)e.Operand;
            // prefer negation for parameters or constants
            if(binExpr.Left is ConstantExpression || IsParameter(binExpr.Left)) {
              return Visit(binExpr.Update(Expression.Negate(binExpr.Left), null, binExpr.Right));
            } else return Visit(binExpr.Update(binExpr.Left, null, Expression.Negate(binExpr.Right)));
          }),
        // 
        new UnaryExpressionRule(
          "-f(x) = f(-x) | f is odd",
          e => e.NodeType == ExpressionType.Negate && e.Operand is MethodCallExpression callExpr && IsOddFunction(callExpr),
          e => {
            var callExpr = (MethodCallExpression)e.Operand;
            return Visit(callExpr.Update(callExpr.Object, new [] {Expression.Negate(callExpr.Arguments[0]) }));
          }),
        /// 
        new UnaryExpressionRule(
          "-(-x) -> x",
          e => e.NodeType == ExpressionType.Negate && e.Operand.NodeType == ExpressionType.Negate,
          e => ((UnaryExpression) e.Operand).Operand
          ),

      });

      callRules.AddRange(new[] {
        // 
        new MethodCallExpressionRule(
          "abs(-(x)) -> abs(x)",
          e => e.Method == abs && e.Arguments[0].NodeType == ExpressionType.Negate,
          e => e.Update(e.Object, new [] {((UnaryExpression) e.Arguments[0]).Operand})
          ),
        new MethodCallExpressionRule(
          "pow(pow(a, x), y) -> pow(a, x * y)",
          e => e.Method == pow && e.Arguments[0] is MethodCallExpression callExpr && callExpr.Method == pow,
          e => {
            var inner = (MethodCallExpression)e.Arguments[0];
            return Visit(e.Update(e.Object, new [] {inner.Arguments[0], Expression.Multiply(e.Arguments[1], inner.Arguments[1]) }));
              }
          ),
        new MethodCallExpressionRule(
         "pow(1/x, y) -> pow(x, -y) | len(simplify(-y)) <= len(y)",
         e => IsPower(e.Method)
           && e.Arguments[0].NodeType == ExpressionType.Divide
           && e.Arguments[0] is BinaryExpression binaryExpression && IsConstant(binaryExpression.Left) && GetConstantValue(binaryExpression.Left) == 1.0,
           // && Length(e.Arguments[1]) >= Length(Visit(Expression.Negate(e.Arguments[1]))),
         e => {
           var div = (BinaryExpression)e.Arguments[0];
           var c = GetConstantValue(div.Left);
           var exponent = e.Arguments[1];
           return Visit(e.Update(e.Object, new [] { div.Right, Expression.Negate(exponent)}));
         }
        ),
        
         // recursive?!
        // new MethodCallExpressionRule(
        //   "pow(x, -y) -> 1 / pow(x, y)",
        //   e => IsPower(e.Method) && e.Arguments[1].NodeType == ExpressionType.Negate,
        //   e => {
        //       var exponent = (UnaryExpression)e.Arguments[1];
        //       return Visit(Expression.Divide(Expression.Constant(1.0), e.Update(e.Object, new [] { e.Arguments[0], exponent.Operand })));
        //     }
        //   ),

        // 
        new MethodCallExpressionRule(
          "exp(x)^y = exp(x*y)",
          e => IsPower(e.Method) && e.Arguments[0] is MethodCallExpression inner && inner.Method == exp,
          e => {
            var inner = (MethodCallExpression)e.Arguments[0];
            return Visit(inner.Update(inner.Object, new [] {Expression.Multiply(inner.Arguments[0], e.Arguments[1]) }));
          }),
        new MethodCallExpressionRule(
          "sqrt(x²) = abs(x)",
          e => e.Method == sqrt && e.Arguments[0] is MethodCallExpression inner && IsPower(inner.Method)
               && IsConstant(inner.Arguments[1]) && GetConstantValue(inner.Arguments[1])==2,
          e => {
            var inner = (MethodCallExpression)e.Arguments[0];
            return Visit(Expression.Call(abs, inner));
          }),
        new MethodCallExpressionRule(
          "sqrt(x)² = x",
          e => IsPower(e.Method) && IsConstant(e.Arguments[1]) && GetConstantValue(e.Arguments[1])==2
              && e.Arguments[0] is MethodCallExpression inner && inner.Method == sqrt,
          e => {
            var inner = (MethodCallExpression)e.Arguments[0];
            return Visit(inner.Arguments[0]);
          }),

        new MethodCallExpressionRule(
          "cbrt(x³) = x",
          e => e.Method == cbrt && e.Arguments[0] is MethodCallExpression inner && IsPower(inner.Method)
               && IsConstant(inner.Arguments[1]) && GetConstantValue(inner.Arguments[1])==3,
          e => {
            var inner = (MethodCallExpression)e.Arguments[0];
            return Visit(inner);
          }),
        new MethodCallExpressionRule(
          "cbrt(x)³ = x",
          e => IsPower(e.Method) && IsConstant(e.Arguments[1]) && GetConstantValue(e.Arguments[1])==3
              && e.Arguments[0] is MethodCallExpression inner && inner.Method == cbrt,
          e => {
            var inner = (MethodCallExpression)e.Arguments[0];
            return Visit(inner.Arguments[0]);
          }),
      });
    }

    private void AddReparameterizationRules() {
      visitCache.Clear();

      binaryRules.AddRange(new[] {
      // 
      new BinaryExpressionRule(
        "param ° param -> param | ° is associative",
        e => IsParameter(e.Left) && IsParameter(e.Right),
        e => NewParameter(Apply(e.NodeType, GetParameterValue(e.Left), GetParameterValue(e.Right)))
        ),
      // 
      new BinaryExpressionRule(
        "(a ° param) ° param -> a ° param  | ° is associative",
        e => IsAssociative(e) && e.Left.NodeType == e.NodeType && e.Left is BinaryExpression leftExpr
        && IsParameter(leftExpr.Right) && IsParameter(e.Right),
        e => {
          var leftExpr = (BinaryExpression)e.Left;
          return Visit(e.Update(leftExpr.Left, null,
            NewParameter(Apply(e.NodeType, GetParameterValue(leftExpr.Right), GetParameterValue(e.Right)))));
            }
        ),
      // 
      new BinaryExpressionRule(
        "param ° const -> param | ° is associative",
        e => IsParameter(e.Left) && IsConstant(e.Right),
        e => NewParameter(Apply(e.NodeType, GetParameterValue(e.Left), GetConstantValue(e.Right)))
        ),
      new BinaryExpressionRule(
        "(a ° param) ° const -> a ° param | ° is associative",
        e => IsAssociative(e) && e.Left.NodeType == e.NodeType && e.Left is BinaryExpression leftExpr
        && IsParameter(leftExpr.Right) && IsConstant(e.Right),
        e => {
          var leftExpr = (BinaryExpression)e.Left;
          return Visit(e.Update(leftExpr.Left, null,
            NewParameter(Apply(e.NodeType, GetParameterValue(leftExpr.Right), GetConstantValue(e.Right)))));
            }
        ),
      // 
      new BinaryExpressionRule(
        "const ° param -> param | ° is associative",
        e => IsConstant(e.Left) && IsParameter(e.Right),
        e => NewParameter(Apply(e.NodeType, GetConstantValue(e.Left), GetParameterValue(e.Right)))
        ),
      new BinaryExpressionRule(
        "(a ° const) ° param -> a ° param | ° is associative",
        e => IsAssociative(e) && e.Left.NodeType == e.NodeType && e.Left is BinaryExpression leftExpr
        && IsConstant(leftExpr.Right) && IsParameter(e.Right),
        e => {
          var leftExpr = (BinaryExpression)e.Left;
          return Visit(e.Update(leftExpr.Left, null,
            NewParameter(Apply(e.NodeType, GetConstantValue(leftExpr.Right), GetParameterValue(e.Right)))));
            }
        ),
      new BinaryExpressionRule(
        "-a * p -> a * p",
        e => e.NodeType == ExpressionType.Multiply && e.Left.NodeType == ExpressionType.Negate && IsParameter(e.Right),
        e => {
           var unary = (UnaryExpression)e.Left;
           return Visit(Expression.Multiply(unary.Operand, NewParameter(-GetParameterValue(e.Right))));
          }
        ),
      new BinaryExpressionRule(
        "((a * p) + p) * p -> (a * p) + p",
        e => e.NodeType == ExpressionType.Multiply && IsParameter(e.Right)
          && e.Left is BinaryExpression left && left.NodeType == ExpressionType.Add && IsParameter(left.Right)
          && left.Left is BinaryExpression leftLeft && leftLeft.NodeType == ExpressionType.Multiply && IsParameter(leftLeft.Right),
        e => {
            var left = (BinaryExpression)e.Left;
            var leftLeft = (BinaryExpression)left.Left;
            var scale = GetParameterValue(e.Right);
            var offset = GetParameterValue(left.Right);
            var innerScale = GetParameterValue(leftLeft.Right);
            return Visit(Expression.Add(Expression.Multiply(leftLeft.Left, NewParameter(innerScale*scale)), NewParameter(offset * scale)));
          }
        ),
       new BinaryExpressionRule(
        "x / (x * p) -> p",
        e => e.NodeType == ExpressionType.Divide
          && e.Right.NodeType == ExpressionType.Multiply && e.Right is BinaryExpression right
          && right.Left.ToString() == e.Left.ToString()
          && IsParameterOrConstant(right.Right),
        e => {
          var right = (BinaryExpression)e.Right;
          return NewParameter(1.0 / GetParameterValue(right.Right));
        }),
      new BinaryExpressionRule(
        "z / (x * p) -> (z*p') / x",
        e => e.NodeType == ExpressionType.Divide
          && e.Right.NodeType == ExpressionType.Multiply && e.Right is BinaryExpression right
          && IsParameter(right.Right),
        e => {
          var right = (BinaryExpression)e.Right;
          return Visit(Expression.Divide(
            Expression.Multiply(e.Left, NewParameter(1.0 / GetParameterValue(right.Right))),
            right.Left
            ));
        }),

      new BinaryExpressionRule(
        "sum -> sum' | where sum' contains merged terms",
        e =>  {
            if(!(e.NodeType == ExpressionType.Add || e.NodeType == ExpressionType.Subtract)) return false;
            var terms = CollectTermsVisitor.CollectTerms(e).ToArray();
            var newTerms = FoldTerms(terms).ToArray();
            return (terms.Length > newTerms.Length);
          },
        e => {
            return Visit(FoldTerms(CollectTermsVisitor.CollectTerms(e)).Aggregate(Expression.Add));
          }
        ),
       // 
       new BinaryExpressionRule(
         "x - p -> x + (-p)",
         e => e.NodeType == ExpressionType.Subtract && IsParameter(e.Right),
         e => Visit(Expression.Add(e.Left, NewParameter(-GetParameterValue(e.Right))))
        ),
       // 
       new BinaryExpressionRule(
         "x / p -> x * (1/p)",
         e => e.NodeType == ExpressionType.Divide && IsParameter(e.Right),
         e => Visit(Expression.Multiply(e.Left, NewParameter(1.0 / GetParameterValue(e.Right))))
        ),
       new BinaryExpressionRule(
         "x - (y * p) -> x + (y * p)",
         e => e.NodeType == ExpressionType.Subtract
           && e.Right is BinaryExpression rightBin && rightBin.NodeType == ExpressionType.Multiply && IsParameter(rightBin.Right),
         e => {
           var rightBin = (BinaryExpression)e.Right;
           return Visit(Expression.Add(e.Left, Expression.Multiply(rightBin.Left, NewParameter(-GetParameterValue(rightBin.Right)))));
         }),
       new BinaryExpressionRule(
         "x - (p / y) -> x + (p / y)",
         e => e.NodeType == ExpressionType.Subtract
           && e.Right is BinaryExpression rightBin && rightBin.NodeType == ExpressionType.Divide && IsParameter(rightBin.Left),
         e => {
           var rightBin = (BinaryExpression)e.Right;
           return Visit(Expression.Add(e.Left, Expression.Divide(NewParameter(-GetParameterValue(rightBin.Left)), rightBin.Right)));
         }),

       // extract scaling parameter out of affine form a 
       // 
       new BinaryExpressionRule(
         "a / b -> (a'/ b) * p  | (a is affine) or (b is affine)",
         e => e.NodeType == ExpressionType.Divide 
           // && !(IsConstant(e.Left) && GetConstantValue(e.Left) == 1.0)
           && HasParameters(e.Left) && HasParameters(e.Right)
           && (IsAffine(e.Left) || IsAffine(e.Right))
           && !(ExtractScaleFromAffine(e.Left).scale == 1.0 && ExtractScaleFromAffine(e.Right).scale == 1.0),
         e => {
           (var scaledLeft, var scaleLeft) = ExtractScaleFromAffine(e.Left);
           (var scaledRight, var scaleRight) = ExtractScaleFromAffine(e.Right);          
           // TODO: if both scales are 1.0 this rule leads to an infinite recursion
           return Visit(Expression.Multiply(Expression.Divide(scaledLeft, scaledRight), NewParameter(scaleLeft / scaleRight)));
           }),

       // new BinaryExpressionRule(
       //   "pow(x, y) * p -> (1 / pow(x, -y)) * p | len(simplify(-y)) <= len(y)",
       //   e => e.NodeType == ExpressionType.Multiply && IsParameterOrConstant(e.Right) 
       //    && e.Left is MethodCallExpression callExpr && IsPower(callExpr.Method) 
       //    && Length(Visit(Expression.Negate(callExpr.Arguments[1]))) <= Length(callExpr.Arguments[1]),
       //   e => {
       //     var callExpr = (MethodCallExpression)e.Left;
       //     return Visit(Expression.Multiply(
       //         Expression.Divide(
       //           Expression.Constant(1.0),
       //           callExpr.Update(callExpr.Object, new [] { callExpr.Arguments[0], Expression.Negate(callExpr.Arguments[1]) }))
       //         ,e.Right));
       //   }),
      });

      unaryRules.AddRange(new[] {
        // 
        new UnaryExpressionRule(
          "-(p) -> p",
          e => e.NodeType == ExpressionType.Negate && IsParameter(e.Operand),
          e => NewParameter(-GetParameterValue(e.Operand))
          ),
        // 
        new UnaryExpressionRule(
          "-(a) -> a' * p | a is affine",
          e => e.NodeType == ExpressionType.Negate && IsAffine(e.Operand),
          e => {
            (var scaledAffine, var scale) = ExtractScaleFromAffine(e.Operand);
              return Visit(Expression.Multiply(scaledAffine, NewParameter(-scale)));
            }),
      });

      callRules.AddRange(new[] {
      // 
      new MethodCallExpressionRule(
        "fold parameters in method calls",
        e => e.Arguments.All(IsParameter),
        e => NewParameter((double) e.Method.Invoke(e.Object, e.Arguments.Select(GetParameterValue).OfType<object>().ToArray()))
        ),
      // only parameters and constants (rule for folding constants should be applied first)
      new MethodCallExpressionRule(
        "fold parameters and constants in method calls",
        e => e.Arguments.All(arg => IsParameter(arg) || IsConstant(arg)),
        e => NewParameter((double) e.Method.Invoke(e.Object, e.Arguments.Select(GetParameterOrConstantValue).OfType<object>().ToArray()))
        ),
      // 
      new MethodCallExpressionRule(
        "aq(x, p) = x / sqrt(1 + p²) = x * 1/sqrt(1+p²) = x * p'",
        e => e.Method == aq && IsParameterOrConstant(e.Arguments[1]),
        e => {
          if(IsParameter(e.Arguments[1])) {
            return Visit(Expression.Multiply(e.Arguments[0], NewParameter(1.0 / Math.Sqrt(1.0 + GetParameterValue(e.Arguments[1])))));
          } else {
            // is constant
            return Visit(Expression.Multiply(e.Arguments[0], Expression.Constant(1.0 / Math.Sqrt(1.0 + GetConstantValue(e.Arguments[1])))));
          }
        }),  

      // 
      new MethodCallExpressionRule(
        "aq(x, y) -> aq(1,y) * x",
        e => e.Method == aq,
        e => Visit(Expression.Multiply(e.Update(e.Object, new [] { Expression.Constant(1), e.Arguments[1] }), e.Arguments[0]))
        ),

      // 
      new MethodCallExpressionRule(
        "sqrt(a) -> sqrt(a') * p | a is affine, p is positive",
        e => e.Method == sqrt && IsAffine(e.Arguments[0]),
        e => {
          (var scaledAffine, var scale) = ExtractPositiveScaleFromAffine(e.Arguments[0]);
          return Visit(Expression.Multiply(e.Update(e.Object, new [] { scaledAffine }), NewParameter(Math.Sqrt(scale))));
        }),
      // 
      new MethodCallExpressionRule(
        "cbrt(a) -> cbrt(a') * p | a is affine",
        e => e.Method == cbrt && IsAffine(e.Arguments[0]),
        e => {
          (var scaledAffine, var scale) = ExtractScaleFromAffine(e.Arguments[0]);
          return Visit(Expression.Multiply(e.Update(e.Object, new [] { scaledAffine }), NewParameter(Functions.Cbrt(scale))));
        }),
      // 
      new MethodCallExpressionRule(
        "abs(a) -> abs(a') * p | a is affine, p is positive",
        e => e.Method == abs && IsAffine(e.Arguments[0]),
        e => {
          (var scaledAffine, var scale) = ExtractPositiveScaleFromAffine(e.Arguments[0]);
          return Visit(Expression.Multiply(e.Update(e.Object, new [] { scaledAffine }), NewParameter(scale)));
        }),
      // 
      new MethodCallExpressionRule(
        "log(a) -> log(a') + p | a is affine, p is positive",
        e => e.Method == log && IsAffine(e.Arguments[0]),
        e => {
          (var scaledAffine, var scale) = ExtractPositiveScaleFromAffine(e.Arguments[0]);
          return Visit(Expression.Add(e.Update(e.Object, new [] { scaledAffine }), NewParameter(Math.Log(scale))));
        }),
      // 
      new MethodCallExpressionRule(
        "exp(x + p) -> exp(x) * p",
        e => e.Method == exp
           && e.Arguments[0] is BinaryExpression binExpr && binExpr.NodeType == ExpressionType.Add
           && IsParameter(binExpr.Right),
        e => {
          var binExpr = (BinaryExpression)e.Arguments[0];
          var offset = GetParameterValue(binExpr.Right);
          return Visit(Expression.Multiply(Expression.Call(exp, new [] { binExpr.Left}), NewParameter(Math.Exp(offset))));
        }),
      new MethodCallExpressionRule(
        "pow(a, p) -> pow(a',p') | a is affine, p is param or const",
        e => e.Method == pow && IsAffine(e.Arguments[0]) && IsParameter(e.Arguments[1]),
        e => {
          (var scaledAffine, var scale) = ExtractScaleFromAffine(e.Arguments[0]);
          var exponent = GetParameterValue(e.Arguments[1]);
          return Visit(Expression.Multiply(e.Update(e.Object, new [] { scaledAffine, e.Arguments[1] }), NewParameter(Math.Pow(scale, exponent))));
        }),
      new MethodCallExpressionRule(
        "pow(pow(x, a), b) -> pow(x,a*b)",
        e => IsPower(e.Method)
          && e.Arguments[0] is MethodCallExpression inner && IsPower(inner.Method),
        e => {
          var inner = (MethodCallExpression)e.Arguments[0];
          return Visit(e.Update(e.Object, new [] { inner.Arguments[0], Expression.Multiply(inner.Arguments[1], e.Arguments[1]) }));
        }),

    });
    }
    public void AddCleanupRules() {
      // a ruleset that pushes multiplicative parameters down into affine expressions without introducing new parameters
      binaryRules.AddRange(new[] {
        new BinaryExpressionRule(
          "(z + p) * p -> (z * p) + p",
          e => e.NodeType == ExpressionType.Multiply && IsParameter(e.Right)
            && e.Left is BinaryExpression left && left.NodeType == ExpressionType.Add && IsParameter(left.Right),
          e => {
            var left =(BinaryExpression)e.Left;
            return Visit(Expression.Add(
              Expression.Multiply(left.Left, e.Right),
              NewParameter(GetParameterValue(left.Right) * GetParameterValue(e.Right))
              ));
          }),
        new BinaryExpressionRule(
          "(x / y) * z -> (x * z) / y",
          e => e.NodeType == ExpressionType.Multiply && e.Left.NodeType == ExpressionType.Divide,
          e => {
            var left = (BinaryExpression)e.Left;
            return Visit(Expression.Divide(Expression.Multiply(left.Left, e.Right), left.Right));
          }),
        new BinaryExpressionRule(
          "(p - x) * p -> p*x + p",
          e => e.NodeType == ExpressionType.Multiply && IsParameter(e.Right) &&
            e.Left is BinaryExpression left && IsParameter(left.Left) && left.NodeType == ExpressionType.Subtract,
          e => {
            var left = (BinaryExpression)e.Left;
            return Visit(Expression.Add(
              Expression.Multiply(left.Right, NewParameter(-GetParameterValue(e.Right))),
              NewParameter(GetParameterValue(left.Left) * GetParameterValue(e.Right))
              ));
          }),
        // lift negation out of division
        new BinaryExpressionRule(
          "-1 / x -> -(1/x)",
          e => e.NodeType == ExpressionType.Divide && IsConstant(e.Left) && GetConstantValue(e.Left) == -1.0,
          e => Expression.Negate(Expression.Divide(Expression.Constant(1.0), e.Right))
          ),
        new BinaryExpressionRule(
          "x / -y -> -(x/y)",
          e => e.NodeType == ExpressionType.Divide && e.Right.NodeType == ExpressionType.Negate,
          e => Expression.Negate(Expression.Divide(e.Left, ((UnaryExpression)e.Right).Operand))
          ),
        // lift negation out of subtraction
        new BinaryExpressionRule(
          "(-x - y) -> -(x + y)",
          e => e.NodeType == ExpressionType.Subtract && e.Left.NodeType == ExpressionType.Negate,
          e => Expression.Negate(Expression.Add(((UnaryExpression)e.Left).Operand, e.Right))
          ),
        new BinaryExpressionRule(
          "-x * p -> x * p",
          e => e.NodeType == ExpressionType.Multiply && e.Left.NodeType == ExpressionType.Negate && IsParameter(e.Right),
          e => Expression.Multiply(((UnaryExpression)e.Left).Operand, NewParameter(-GetParameterValue(e.Right)))
          ),
        new BinaryExpressionRule(
          "-x * p -> x * p",
          e => e.NodeType == ExpressionType.Multiply && e.Left.NodeType == ExpressionType.Negate && IsConstant(e.Right),
          e => Expression.Multiply(((UnaryExpression)e.Left).Operand, Expression.Constant(-GetConstantValue(e.Right)))
          ),
        new BinaryExpressionRule(
          "(-1/x) + z -> z - 1/x", // prefer (1/x == inv(x))
          e => e.NodeType == ExpressionType.Add
            && e.Left is BinaryExpression left && left.NodeType == ExpressionType.Divide
            && IsConstant(left.Left) && GetConstantValue(left.Left) == -1.0,
          e => {
            var left = (BinaryExpression)e.Left;
            return Expression.Subtract(e.Right, Expression.Divide(Expression.Constant(1.0), left.Right));
              }
          ),
        new BinaryExpressionRule(
          "z + (-1/x) -> z - 1/x", // prefer (1/x == inv(x))
          e => e.NodeType == ExpressionType.Add
            && e.Right is BinaryExpression right && right.NodeType == ExpressionType.Divide
            && IsConstant(right.Left) && GetConstantValue(right.Left) == -1.0,
          e => {
            var right = (BinaryExpression)e.Right;
            return Expression.Subtract(e.Left, Expression.Divide(Expression.Constant(1.0), right.Right));
              }
          ),
         new BinaryExpressionRule(
           "((-1/x) - z) * p -> (1/x + z) * p", // prefer (1/x == inv(x))
           e => e.NodeType == ExpressionType.Multiply && IsParameter(e.Right)
             && e.Left is BinaryExpression left && left.NodeType == ExpressionType.Subtract
             && left.Left is BinaryExpression leftLeft && leftLeft.NodeType == ExpressionType.Divide
             && IsConstant(leftLeft.Left) && GetConstantValue(leftLeft.Left) == -1.0,
           e => {
             var left = (BinaryExpression)e.Left;
             var leftLeft =(BinaryExpression)left.Left;
             return e.Update(
                 Expression.Add(Expression.Divide(Expression.Constant(1.0), leftLeft.Right), left.Right),
                 null,
                 NewParameter(-GetParameterValue(e.Right)));
               }
           ),
         new BinaryExpressionRule(
           "-x + y -> y - x",
           e => e.NodeType == ExpressionType.Add && e.Left.NodeType == ExpressionType.Negate,
           e => Expression.Subtract(e.Right, ((UnaryExpression)e.Left).Operand)
           ),
         new BinaryExpressionRule(
           "x/-y + z -> z - x/y",
           e => e.NodeType == ExpressionType.Add
             && e.Left.NodeType == ExpressionType.Divide && e.Left is BinaryExpression left
             && left.Right.NodeType == ExpressionType.Negate,
           e => {
             var left = (BinaryExpression)e.Left;
             return Expression.Subtract(e.Right, left.Update(left.Left, null, ((UnaryExpression)left.Right).Operand));
           }),
         new BinaryExpressionRule(
           "x*(-y) + z -> z - x*y",
           e => e.NodeType == ExpressionType.Add
             && e.Left.NodeType == ExpressionType.Multiply && e.Left is BinaryExpression left
             && left.Right.NodeType == ExpressionType.Negate,
           e => {
             var left = (BinaryExpression)e.Left;
             return Expression.Subtract(e.Right, left.Update(left.Left, null, ((UnaryExpression)left.Right).Operand));
           }),

        new BinaryExpressionRule(
          "sum * p -> sum'",
          e => e.NodeType == ExpressionType.Multiply && IsParameter(e.Right)
            && CollectTermsVisitor.CollectTerms(e.Left).Where(HasScalingParameter).Count() >= 1,
          e => {
            var terms = CollectTermsVisitor.CollectTerms(e.Left).ToArray();
            var termsWithScale = terms.Where(HasScalingParameter).ToArray();

            var termsWithoutScale = terms.Where(t => !HasScalingParameter(t)).DefaultIfEmpty(Expression.Constant(0.0)).ToArray();
            var scale = GetParameterValue(e.Right);
            return Visit(Expression.Add(
                termsWithScale.Select(t => ScaleTerm(t, scale)).Aggregate(Expression.Add),
                Expression.Multiply(termsWithoutScale.Aggregate(Expression.Add), e.Right)
              ));
          }
          ),
        // not exact 0/0 | ∞/∞ | ∞ - ∞ | 0^0 | 1^∞ | ∞^0 are all NaN
        new BinaryExpressionRule(
          "x ° c -> c | c is Inf or NaN",
          e => e.NodeType != ExpressionType.ArrayIndex && IsConstant(e.Right)
            && (double.IsInfinity(GetConstantValue(e.Right)) || double.IsNaN(GetConstantValue(e.Right))),
          e => Expression.Constant(GetConstantValue(e.Right))
        ),
        new BinaryExpressionRule(
          "c ° x -> c | c is Inf or NaN",
          e => e.NodeType != ExpressionType.ArrayIndex && IsConstant(e.Left)
            && (double.IsInfinity(GetConstantValue(e.Left)) || double.IsNaN(GetConstantValue(e.Left))),
          e => Expression.Constant(GetConstantValue(e.Left))
        )
      });
      unaryRules.AddRange(new[] {
        new UnaryExpressionRule(
          "+/- NaN -> NaN",
          e => IsConstant(e.Operand) && (double.IsNaN(GetConstantValue(e.Operand)) || double.IsInfinity(GetConstantValue(e.Operand))),
          e => Expression.Constant(Apply(e.NodeType, GetConstantValue(e.Operand)))
          )
      });

      callRules.AddRange(new[] {
        // lift negation out of exponent
        new MethodCallExpressionRule(
          "pow(x, -y) => 1/pow(x, y)",
          e => IsPower(e.Method) && e.Arguments[1].NodeType == ExpressionType.Negate,
          e => Expression.Divide(Expression.Constant(1.0), e.Update(e.Object, new [] { e.Arguments[0], ((UnaryExpression)e.Arguments[1]).Operand }))
          ),
        // not exact but we do not care
        new MethodCallExpressionRule(
          "f(c, ...) | c is NaN or Infinity",
          e => e.Arguments.Where(IsConstant).Any(arg => double.IsNaN(GetConstantValue(arg)) || double.IsInfinity(GetConstantValue(arg))),
          e => Expression.Constant(double.NaN)
          )
      });
    }


    public static ParameterizedExpression Simplify(ParameterizedExpression expr, bool debugRules = false) {
      if (!CheckExprVisitor.CheckValid(expr.expr)) throw new NotSupportedException();

      var v = new RuleBasedSimplificationVisitor(expr.p, expr.pValues, debugRules);
      var body = v.Visit(expr.expr.Body);
      if (v.debugRules) {
        Console.WriteLine("Used rules:");
        foreach (var tup in v.matchedRules) {
          Console.WriteLine($"=> {tup.expr} ° {tup.rule}");
        }
      }
      return new ParameterizedExpression(expr.expr.Update(body, expr.expr.Parameters), expr.p, v.pValues.ToArray());

    }

    public static ParameterizedExpression SimplifyWithoutReparameterization(ParameterizedExpression expr, bool debugRules = false) {
      // This is necessary for simplification where it is not allowed to re-parameterize the expression.
      // For example, when simplifying the derivative of an expression.
      if (!CheckExprVisitor.CheckValid(expr.expr)) throw new NotSupportedException();

      var v = new RuleBasedSimplificationVisitor(expr.p, expr.pValues, debugRules);
      v.ClearRules();
      v.AddConstantFoldingRules();
      v.AddBasicRules();
      var body = v.Visit(expr.expr.Body);
      return new ParameterizedExpression(expr.expr.Update(body, expr.expr.Parameters), expr.p, v.pValues.ToArray());
    }

    // For simplification a canonical representation is used that works best for this.
    // Afterwards we want to prevent dependencies between parameters as best as possible which requires a different ruleset.
    public static ParameterizedExpression Cleanup(ParameterizedExpression expr, bool debugRules = false) {
      var v = new RuleBasedSimplificationVisitor(expr.p, expr.pValues, debugRules);
      v.ClearRules();
      v.AddConstantFoldingRules();
      v.AddCleanupRules();

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
      if (visitCache.TryGetValue(node.ToString(), out var result)) return result;

      // simplify left and right first
      var left = Visit(node.Left);
      var right = Visit(node.Right);
      result = node.Update(left, null, right);

      var r = binaryRules.FirstOrDefault(ru => ru.Match((BinaryExpression)result));
      while (r != default(BinaryExpressionRule)) {
        MarkUsage(r.Description, result);
        result = r.Apply((BinaryExpression)result);
        if (result is BinaryExpression binExpr) {
          r = binaryRules.FirstOrDefault(ru => ru.Match(binExpr));
        } else break;
      }

      visitCache.Add(node.ToString(), result);
      return result;
    }


    protected override Expression VisitUnary(UnaryExpression node) {
      if (visitCache.TryGetValue(node.ToString(), out var result)) return result;

      var opd = Visit(node.Operand);
      result = node.Update(opd);
      var r = unaryRules.FirstOrDefault(ru => ru.Match((UnaryExpression)result));
      while (r != default(UnaryExpressionRule)) {
        MarkUsage(r.Description, result);
        result = r.Apply((UnaryExpression)result);
        if (result is UnaryExpression unaryExpr) {
          r = unaryRules.FirstOrDefault(ru => ru.Match(unaryExpr));
        } else break;
      }

      visitCache.Add(node.ToString(), result);
      return result;
    }

    protected override Expression VisitMethodCall(MethodCallExpression node) {
      if (visitCache.TryGetValue(node.ToString(), out var result)) return result;

      result = node.Update(node.Object, node.Arguments.Select(Visit));
      var r = callRules.FirstOrDefault(ru => ru.Match((MethodCallExpression)result));
      while (r != default(MethodCallExpressionRule)) {
        MarkUsage(r.Description, result);
        result = r.Apply((MethodCallExpression)result);
        if (result is MethodCallExpression callExpr) {
          r = callRules.FirstOrDefault(ru => ru.Match(callExpr));
        } else break;
      }

      visitCache.Add(node.ToString(), result);
      return result;
    }


    // for debugging rules
    private void MarkUsage(string description, Expression expr) {
      matchedRules.Add(ValueTuple.Create(description, expr));
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
      else if (IsParameter(left) && IsParameter(right) || IsVariable(left) && IsVariable(right)) {
        // order parameters and variables by index
        return ArrayIndex(left) - ArrayIndex(right);
      } else {
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
      return IsConstant(e) ? 6
        : IsParameter(e) ? 5
        : IsVariable(e) ? 4
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

    private double Apply(ExpressionType nodeType, double v) {
      switch (nodeType) {
        case ExpressionType.UnaryPlus: return v;
        case ExpressionType.Negate: return -v;
        default: throw new NotSupportedException();
      }
    }

    private Expression NewParameter(double val) {
      pValues.Add(val);
      return Expression.ArrayIndex(p, Expression.Constant(pValues.Count - 1));
    }

    private double GetConstantValue(Expression constExpr) => (double)((ConstantExpression)constExpr).Value;
    private int ArrayIndex(Expression expr) => (int)((ConstantExpression)((BinaryExpression)expr).Right).Value;
    private double GetParameterValue(Expression expr) => pValues[ArrayIndex(expr)];
    private double GetParameterOrConstantValue(Expression expr) {
      if (expr is ConstantExpression) return GetConstantValue(expr);
      else if (IsParameter(expr)) return GetParameterValue(expr);
      else throw new ArgumentException();
    }

    private bool IsConstant(Expression expr) => expr is ConstantExpression;
    private bool IsParameter(Expression expr) => expr is BinaryExpression binExpr && binExpr.Left == p;
    private bool IsVariable(Expression expr) => expr is BinaryExpression binExpr && expr.NodeType == ExpressionType.ArrayIndex && binExpr.Left != p;
    private bool IsParameterOrConstant(Expression expr) => IsConstant(expr) || IsParameter(expr);

    private bool HasParameters(Expression left) => CountParametersVisitor.Count(left, p) > 0;

    private bool IsAffine(Expression expr) => CollectTermsVisitor.CollectTerms(expr).All(HasScalingParameter); // constants as well
    private int Length(Expression expression) => CountNodesVisitor.Count(expression); // number of nodes


    private bool HasScalingParameter(Expression arg) {
      return IsParameter(arg) ||
        arg is BinaryExpression binExpr && binExpr.NodeType == ExpressionType.Multiply && (IsParameter(binExpr.Left) || IsParameter(binExpr.Right)); // constants as well
    }
    private bool IsPower(MethodInfo method) => method == pow || method == powabs;

    private IEnumerable<Expression> FoldTerms(IEnumerable<Expression> terms) {
      var exprStr2scale = new Dictionary<string, Expression>();
      var exprStr2expr = new Dictionary<string, Expression>();
      foreach (var t in terms) {
        (var scaledTerm, var scale) = ExtractScaleExprFromTerm(t);
        var scaledTermStr = scaledTerm.ToString();
        if (exprStr2scale.TryGetValue(scaledTermStr, out var curScale)) {
          exprStr2scale[scaledTermStr] = Expression.Add(curScale, scale);
        } else {
          exprStr2scale[scaledTermStr] = scale;
          exprStr2expr[scaledTermStr] = scaledTerm;
        }
      }

      foreach (var kvp in exprStr2scale) {
        var termStr = kvp.Key;
        var scale = kvp.Value;
        yield return Expression.Multiply(exprStr2expr[termStr], scale);
      }
    }


    private Expression ScaleTerm(Expression t, double scale) {
      var factors = CollectFactorsVisitor.CollectFactors(t);
      var nonParamFactors = factors.Where(f => !IsParameter(f)).ToArray();
      var curScale = factors.Where(IsParameter).Select(GetParameterValue).DefaultIfEmpty(1.0).Aggregate((a, b) => a * b);
      if (!nonParamFactors.Any()) {
        return NewParameter(scale * curScale);
      } else {
        return Expression.Multiply(
          nonParamFactors.Aggregate(Expression.Multiply),
          NewParameter(scale * curScale));
      }
    }

    private (Expression scaledAffine, double scale) ExtractScaleFromAffine(Expression affine) {
      if (!IsAffine(affine)) return (affine, 1.0);
      // p1 x + p2 x + ... + pk -> p1 * (x + p2' x + ... + pk')
      // Only parameters are allowed for p1 .. pk.
      // It would be possible to allow constants as well, as long as there is one parameter that can be extracted.
      // However, in this case we would need to make sure that constants are replaced by constants and parameters by parameters

      var terms = CollectTermsVisitor.CollectTerms(affine).ToArray();
      // try to find first term with non-zero scale (because we divide by scale below)
      Expression scaledAffine = null;
      double scale = 1.0;
      int scaleTermIndex;
      for (scaleTermIndex = 0; scaleTermIndex < terms.Length; scaleTermIndex++) {
        (scaledAffine, scale) = ExtractScaleFromTerm(terms[scaleTermIndex]);
        if (scale != 0.0) {
          break;
        }
      }
      for (int i = 0; i < terms.Length; i++) {
        if (i == scaleTermIndex) continue;
        (var scaledT, var s) = ExtractScaleFromTerm(terms[i]);
        scaledAffine = Expression.Add(scaledAffine, Expression.Multiply(scaledT, NewParameter(s / scale)));
      }
      return (scaledAffine, scale);
    }


    private (Expression scaledAffine, double scale) ExtractPositiveScaleFromAffine(Expression affine) {
      // same as above but returns only positive scale.
      // This is required for extracting parameters out of sqrt and log
      if (!IsAffine(affine)) return (affine, 1.0);

      var terms = CollectTermsVisitor.CollectTerms(affine);
      var firstTerm = terms.First();
      (var scaledAffine, var scale) = ExtractScaleFromTerm(firstTerm);
      if (scale < 0) {
        scaledAffine = Expression.Negate(scaledAffine);
        scale *= -1.0;
      }
      foreach (var t in terms.Skip(1)) {
        (var scaledT, var s) = ExtractScaleFromTerm(t);
        scaledAffine = Expression.Add(scaledAffine, Expression.Multiply(scaledT, NewParameter(s / scale)));
      }
      return (scaledAffine, scale);
    }

    private (Expression scaledTerm, double scale) ExtractScaleFromTerm(Expression term) {
      if (term.NodeType == ExpressionType.Negate) {
        (var scaledTerm, var scale) = ExtractScaleFromTerm(((UnaryExpression)term).Operand);
        return (scaledTerm, -scale);
      } else {
        // t = f1 * .. * fk, we can assume that one of the factors is a parameter
        var factors = CollectFactorsVisitor.CollectFactors(term).ToArray();
        var scaleIdx = Array.FindIndex(factors, IsParameter); // constants as well?
        if (scaleIdx < 0 || factors.Length == 1) return (term, 1.0);
        else return (factors.Take(scaleIdx).Concat(factors.Skip(scaleIdx + 1)).Aggregate(Expression.Multiply), GetParameterValue(factors[scaleIdx]));
      }
    }

    private (Expression scaledTerm, Expression scale) ExtractScaleExprFromTerm(Expression term) {
      if (term.NodeType == ExpressionType.Negate) {
        (var scaledTerm, var scale) = ExtractScaleExprFromTerm(((UnaryExpression)term).Operand);
        return (scaledTerm, Expression.Negate(scale));
      } else {
        var factors = CollectFactorsVisitor.CollectFactors(term).ToArray();
        var scaleIdx = Array.FindIndex(factors, IsParameterOrConstant); // constants as well
        if (scaleIdx < 0) return (term, Expression.Constant(1.0));
        else if (factors.Length == 1) return (Expression.Constant(1.0), factors[scaleIdx]);
        else return (factors.Take(scaleIdx).Concat(factors.Skip(scaleIdx + 1)).Aggregate(Expression.Multiply), factors[scaleIdx]);
      }
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
    private readonly MethodInfo powabs = typeof(Functions).GetMethod("PowAbs", new[] { typeof(double), typeof(double) });
    private readonly bool debugRules;
  }
}

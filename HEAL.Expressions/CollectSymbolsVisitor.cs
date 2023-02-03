using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using static HEAL.Expressions.Expr;

namespace HEAL.Expressions {
  // Collects symbols occuring in the expression.
  // Different variables are counted as different symbols.
  // Parameters are counted as the same symbol.
  // Constants are counted as the same symbol.
  public class CollectSymbolsVisitor : ExpressionVisitor {
    private readonly List<string> Symbols = new List<string>();
    private readonly ParameterExpression p;

    private CollectSymbolsVisitor(ParameterExpression p) : base() {
      this.p = p;
    }

    public static string[] CollectSymbols(Expression<ParametricFunction> expr, ParameterExpression p) {
      var v = new CollectSymbolsVisitor(p);
      v.Visit(expr);
      return v.Symbols.ToArray();
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      // do not recurse into array index expressions (do not count array indices)
      if (node.NodeType == ExpressionType.ArrayIndex)
        if (node.Right == p) {
          Symbols.Add("param");
          return node;
        } else {
          Symbols.Add("var_" + (int)((ConstantExpression)node.Right).Value);
          return node;
        }
      else {
        switch (node.NodeType) {
          case ExpressionType.Add: {
              Symbols.Add("+");
              base.Visit(node.Left);
              base.Visit(node.Right);
              return node;
            }
          case ExpressionType.Subtract: {
              Symbols.Add("-");
              base.Visit(node.Left);
              base.Visit(node.Right);
              return node;
            }
          case ExpressionType.Divide: {
              Symbols.Add("/");
              base.Visit(node.Left);
              base.Visit(node.Right);
              return node;
            }
          case ExpressionType.Multiply: {
              Symbols.Add("*");
              base.Visit(node.Left);
              base.Visit(node.Right);
              return node;
            }
          default: throw new NotSupportedException($"Operator: {node.NodeType}");
        }
      }
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      switch (node.NodeType) {
        case ExpressionType.UnaryPlus: {
            Symbols.Add("+");
            base.Visit(node.Operand);
            return node;
          }
        case ExpressionType.Negate: {
            Symbols.Add("-");
            base.Visit(node.Operand);
            return node;
          }
        default: throw new NotSupportedException($"Operator: {node.NodeType}");
      }
    }

    protected override Expression VisitMethodCall(MethodCallExpression node) {
      Symbols.Add(node.Method.DeclaringType.Name + "." + node.Method.Name + "()");
      foreach (var arg in node.Arguments) {
        Visit(arg);
      }
      return node;
    }

    protected override Expression VisitConstant(ConstantExpression node) {
      Symbols.Add("const");
      return node;
    }
  }
}

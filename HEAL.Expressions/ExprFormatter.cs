﻿using System;
using System.Globalization;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Text;

namespace HEAL.Expressions {
  public class ExprFormatter : ExpressionVisitor {
    private readonly string[] varNames;
    private readonly double[] p;
    private readonly ParameterExpression pParam;
    private readonly ParameterExpression varParam;
    private readonly StringBuilder sb;

    private ExprFormatter(string[] varNames, double[] p, ParameterExpression pParam, ParameterExpression varParam) {
      this.varNames = varNames;
      this.p = p;
      this.pParam = pParam;
      this.varParam = varParam;
      this.sb = new StringBuilder();
    }

    public static string ToString(Expression<Expr.ParametricFunction> expr, string[] varNames, double[] p) {
      var f = new ExprFormatter(varNames, p, expr.Parameters[0], expr.Parameters[1]);
      f.Visit(expr.Body);
      return f.sb.ToString();
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      if (node.NodeType == ExpressionType.ArrayIndex) {
        if (node.Left == pParam) {
          sb.AppendFormat(CultureInfo.InvariantCulture, "{0:g8}", p[ExprFormatter.GetIndex(node)]);
        } else if (node.Left == varParam) {
          sb.Append(ExprFormatter.EscapeVarName(varNames[ExprFormatter.GetIndex(node)]));
        } else throw new ArgumentException();
      } else {
        FormatLeftChildExpr(node.NodeType, node.Left);
        sb.Append($" {ExprFormatter.OpSymbol(node.NodeType)} ");
        if (!IsBinaryOp(node.Right)) {
          Visit(node.Right);
        } else if (node.NodeType == ExpressionType.Subtract || node.NodeType == ExpressionType.Divide) {
          sb.Append('(');
          Visit(node.Right);
          sb.Append(')');
        } else FormatLeftChildExpr(node.NodeType, node.Right); // leftChild on purpose because we want to ignore order of evaluation (a + b) + c = a + (b + c) for our purposes
      }
      return node;
    }


    protected override Expression VisitConstant(ConstantExpression node) {
      sb.AppendFormat(CultureInfo.InvariantCulture, "{0}f", node.Value); // f stands for fixed (see ExprParser)
      return node;
    }

    protected override Expression VisitMethodCall(MethodCallExpression node) {
      if (node.Method == pow) return VisitPow(node);
      sb.Append(node.Method.Name.ToString().ToLower()).Append('(');
      Visit(node.Arguments[0]);
      foreach (var arg in node.Arguments.Skip(1)) {
        sb.Append(", ");
        Visit(arg);
      }
      sb.Append(')');
      return node;
    }

    private Expression VisitPow(MethodCallExpression node) {
      var b = node.Arguments[0];
      var e = node.Arguments[1];

      FormatLeftChildExpr(node.NodeType, b);
      sb.Append(" ** ");
      FormatRightChildExpr(node.NodeType, e);
      return node;
    }



    protected override Expression VisitUnary(UnaryExpression node) {
      if (node.NodeType == ExpressionType.UnaryPlus) Visit(node.Operand);
      else {
        if (!IsBinaryOp(node.Operand) && ExprFormatter.Priority(node.NodeType) >= ExprFormatter.Priority(node.Operand.NodeType)) {
          sb.Append("-(");
          Visit(node.Operand);
          sb.Append(')');
        } else {
          sb.Append('-');
          Visit(node.Operand);
        }
      }
      return node;
    }


    // childExpr is the left argument for the left-associative parent expression
    private void FormatLeftChildExpr(ExpressionType parentType, Expression childExpr) {
      if (IsBinaryOp(childExpr) && Priority(childExpr.NodeType) < Priority(parentType)) {
        sb.Append('('); Visit(childExpr); sb.Append(')');
      } else {
        Visit(childExpr);
      }
    }

    private bool IsBinaryOp(Expression expr) =>
      expr.NodeType == ExpressionType.Add || expr.NodeType == ExpressionType.Subtract
      || expr.NodeType == ExpressionType.Multiply || expr.NodeType == ExpressionType.Divide;

    // childExpr is the right argument for the left-associative parent expression
    private void FormatRightChildExpr(ExpressionType parentType, Expression childExpr) {
      if (IsBinaryOp(childExpr) && Priority(parentType) >= Priority(childExpr.NodeType)) {
        sb.Append('('); Visit(childExpr); sb.Append(')');
      } else {
        Visit(childExpr);
      }
    }

    private static int Priority(ExpressionType nodeType) {
      switch(nodeType) {
        case ExpressionType.Constant: return 13;
        case ExpressionType.Call:
        case ExpressionType.ArrayIndex: return 12;
        case ExpressionType.UnaryPlus:
        case ExpressionType.Negate: return 10;
        case ExpressionType.Multiply:
        case ExpressionType.Divide: return 9;
        case ExpressionType.Add:
        case ExpressionType.Subtract: return 7;
        default: throw new ArgumentException();
      }
    }

    private static string OpSymbol(ExpressionType nodeType) {
      switch(nodeType) {
        case ExpressionType.Add: return "+";
        case ExpressionType.Subtract: return "-";
        case ExpressionType.Multiply: return "*";
        case ExpressionType.Divide: return "/";
        default: throw new ArgumentException();
      }
    }


    private static int GetIndex(BinaryExpression node) {
      return (int)((ConstantExpression)node.Right).Value;
    }



    private static string EscapeVarName(string v) {
      if (v.Contains(' ')) return $"'{v}'";
      else return v;
    }

    private readonly MethodInfo pow = typeof(Math).GetMethod("Pow", new[] { typeof(double), typeof(double) });

  }
}

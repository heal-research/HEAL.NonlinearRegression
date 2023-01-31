using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Text;

namespace HEAL.Expressions {
  public class GraphvizVisitor : ExpressionVisitor {
    private readonly StringBuilder sb;
    private readonly ParameterExpression pParam;
    private readonly ParameterExpression xParam;
    private readonly string[] varNames;
    private readonly double[] paramValues;
    private readonly Dictionary<Expression, double> saturation;
    private readonly double minSat;
    private readonly double maxSat;


    private GraphvizVisitor(StringBuilder stringBuilder, ParameterExpression pParam, double[] pValues,
      ParameterExpression xParam, string[] varNames,
      Dictionary<Expression, double> saturation) {
      this.sb = stringBuilder;
      this.pParam = pParam;
      this.xParam = xParam;
      this.paramValues = pValues;
      this.varNames = varNames;
      this.saturation = saturation;
      if (saturation != null) {
        this.minSat = saturation.Values.Min();
        this.maxSat = saturation.Values.Max();
      }
    }

    public static string Execute(Expression<Expr.ParametricFunction> expression, double[] paramValues = null, string[] varNames = null, Dictionary<Expression, double> saturation = null) {
      var sb = new StringBuilder();
      sb.AppendLine("strict graph {");
      var v = new GraphvizVisitor(sb, expression.Parameters[0], paramValues,
        expression.Parameters[1], varNames, saturation);
      v.Visit(expression.Body);
      sb.AppendLine("}");
      return sb.ToString();
    }

    public override Expression Visit(Expression node) {
      if (node != null) {
        // called for all visited nodes
        if (saturation != null && saturation.TryGetValue(node, out var sat)) {
          // h=0 is red
          // h=0.5 is green
          sat = (sat - minSat) / (maxSat - minSat);
          sb.AppendLine($"n{node.GetHashCode()} [label=\"{Label(node)}\", style=\"filled\", color=\"0,{sat:f2},1\"];"); // color=h,s,v hue,saturation,brightness
        } else {
          sb.AppendLine($"n{node.GetHashCode()} [label=\"{Label(node)}\", style=\"filled\", color=\"0,0,1\"];"); // this draws completely white nodes. remove fill and color if necessary.
        }
      }
      return base.Visit(node);
    }

    private string Label(Expression node) {
      switch (node.NodeType) {
        case ExpressionType.Add: return "+";
        case ExpressionType.Subtract: return "-";
        case ExpressionType.Multiply: {
            // special case: combine p[0] * x[0]
            if (node is BinaryExpression binNode &&
                ((IsVar(binNode.Left) && (IsParam(binNode.Right) || binNode.Right.NodeType == ExpressionType.Constant)) ||
                 (IsVar(binNode.Right) && (IsParam(binNode.Left) || binNode.Left.NodeType == ExpressionType.Constant)))) {
              return $"{Label(binNode.Left)} {Label(binNode.Right)}";
            }
            return "*";
          }
        case ExpressionType.Divide: return "/";
        case ExpressionType.ArrayIndex: {
            var binExpr = node as BinaryExpression;
            var param = binExpr.Left;
            var idx = (int)((ConstantExpression)binExpr.Right).Value;
            if (param == pParam) {
              if (paramValues != null) return paramValues[idx].ToString("e5");
              else return $"p_{idx}";
            } else if (param == xParam) {
              if (varNames != null) return varNames[idx];
              else return $"x_{idx}";
            } else return node.ToString();
          };
        case ExpressionType.Call: return ((MethodCallExpression)node).Method.Name;
        case ExpressionType.Constant: return ((double)((ConstantExpression)node).Value).ToString("g3");
        default: return node.NodeType.ToString();
      }
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      if (node.NodeType == ExpressionType.ArrayIndex) {
        // no separate nodes for arrayIndex
        return node;
      } else if (node.NodeType == ExpressionType.Multiply &&
        ((IsVar(node.Left) && (IsParam(node.Right) || node.Right.NodeType == ExpressionType.Constant)) ||
         (IsVar(node.Right) && (IsParam(node.Left) || node.Left.NodeType == ExpressionType.Constant)))) {
        // no separate nodes for "param * var" or "const * var" or similar
        return node;
      } else {
        sb.AppendLine($"n{node.GetHashCode()} -- n{node.Left.GetHashCode()};");
        sb.AppendLine($"n{node.GetHashCode()} -- n{node.Right.GetHashCode()};");
        return base.VisitBinary(node);
      }
    }

    private bool IsVar(Expression expr) {
      return (expr is BinaryExpression binExpr) && binExpr.NodeType == ExpressionType.ArrayIndex && binExpr.Left == xParam;
    }

    private bool IsParam(Expression expr) {
      return (expr is BinaryExpression binExpr) && binExpr.NodeType == ExpressionType.ArrayIndex && binExpr.Left == pParam;
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      sb.AppendLine($"n{node.GetHashCode()} -- n{node.Operand.GetHashCode()};");
      return base.VisitUnary(node);
    }

    protected override Expression VisitMethodCall(MethodCallExpression node) {
      foreach (var arg in node.Arguments) {
        sb.AppendLine($"n{node.GetHashCode()} -- n{arg.GetHashCode()};");
      }

      return base.VisitMethodCall(node);
    }
  }
}

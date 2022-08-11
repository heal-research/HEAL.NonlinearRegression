using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Security.Cryptography;
using System.Text;

namespace HEAL.Expressions {
  public class GraphvizVisitor : ExpressionVisitor {
    private readonly StringBuilder sb;
    private readonly ParameterExpression pParam;
    private readonly ParameterExpression xParam;
    private readonly string[] varNames;
    private readonly double[] pValues;
    private readonly Dictionary<Expression,double> saturation;
    private readonly double minSat;
    private readonly double maxSat;


    private GraphvizVisitor(StringBuilder stringBuilder, ParameterExpression pParam, double[] pValues,
      ParameterExpression xParam, string[] varNames,
      Dictionary<Expression, double> saturation) {
      this.sb = stringBuilder;
      this.pParam = pParam;
      this.xParam = xParam;
      this.pValues = pValues;
      this.varNames = varNames;
      this.saturation = saturation;
      if (saturation != null) {
        this.minSat = saturation.Values.Min();
        this.maxSat = saturation.Values.Max();
      }
    }

    public static string Execute(Expression<Expr.ParametricFunction> expression, double[] pValues = null, string[] varNames = null, Dictionary<Expression, double> saturation = null) {
      var sb = new StringBuilder();
      sb.AppendLine("strict graph");
      var v = new GraphvizVisitor(sb, expression.Parameters[0],  pValues, 
        expression.Parameters[1], varNames, saturation);
      v.Visit(expression.Body);
      return sb.ToString();
    }

    public override Expression Visit(Expression node) {
      if (node != null) {
        // called for all visited nodes
        if (saturation != null && saturation.TryGetValue(node, out var brightness)) {
          // h=0 is red
          // h=0.5 is green
          brightness = (brightness - minSat)/ (maxSat - minSat);
          sb.AppendLine($"n{node.GetHashCode()} [label=\"{Label(node)}\", color=\"0,1,{brightness:f2}\"];"); // color=h,s,v hue,saturation,brigthness
        } else {
          sb.AppendLine($"n{node.GetHashCode()} [label=\"{Label(node)}\"];");
        }
      }
      return base.Visit(node);
    }

    private string Label(Expression node) {
      switch (node.NodeType) {
        case ExpressionType.Add: return "+";
        case ExpressionType.Subtract: return "-";
        case ExpressionType.Multiply: return "*";
        case ExpressionType.Divide: return "/";
        case ExpressionType.ArrayIndex: {
          var binExpr = node as BinaryExpression;
          var param = binExpr.Left;
          var idx = (int)((ConstantExpression)binExpr.Right).Value;

          if (param == pParam) {
            if (pValues != null) return pValues[idx].ToString("e5");
            else return $"p_{idx}";
          } else if (param == xParam) {
            if (varNames != null) return varNames[idx];
            else return $"x_{idx}";
          } else return node.ToString();
        };
        case ExpressionType.Call: return ((MethodCallExpression)node).Method.Name;
        case ExpressionType.Constant: return node.ToString();
        default: return node.NodeType.ToString();
      }
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      if (node.NodeType == ExpressionType.ArrayIndex) {
        // no separate nodes for arrayIndex
        return node;
      } else {
        sb.AppendLine($"n{node.GetHashCode()} -- n{node.Left.GetHashCode()};");
        sb.AppendLine($"n{node.GetHashCode()} -- n{node.Right.GetHashCode()};");
        return base.VisitBinary(node);
      }
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

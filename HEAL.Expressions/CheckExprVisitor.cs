using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;

namespace HEAL.Expressions {
  /// <summary>
  /// Checks an expression for a ParametricFunction whether it is supported by static methods in Expr.
  /// </summary>
  // NOTE: All visitors should always be made consistent!
  // TODO: derive other visitors from this class?
  public class CheckExprVisitor :ExpressionVisitor {
    private readonly ParameterExpression x;
    private readonly Dictionary<int, int> parameterOccurances;
    private readonly ParameterExpression theta;

    private static readonly MethodInfo[] SupportedMethods = new[] {
      typeof(Math).GetMethod("Abs", new [] {typeof(double)}),
      typeof(Math).GetMethod("Log", new[] {typeof(double)}),
      typeof(Math).GetMethod("Log10", new[] {typeof(double)}),
      typeof(Math).GetMethod("Exp", new[] {typeof(double)}),
      typeof(Math).GetMethod("Sqrt", new[] {typeof(double)}),
      typeof(Math).GetMethod("Sin", new[] {typeof(double)}),
      typeof(Math).GetMethod("Cos", new[] {typeof(double)}),
      typeof(Math).GetMethod("Cosh", new[] {typeof(double)}),
      typeof(Math).GetMethod("Tanh", new[] {typeof(double)}),
      typeof(Math).GetMethod("Pow", new[] {typeof(double), typeof(double)}),
      typeof(Functions).GetMethod("PowAbs", new[] {typeof(double), typeof(double)}),
      typeof(Functions).GetMethod("Cbrt", new[] {typeof(double)}),
      typeof(Functions).GetMethod("Sign", new[] {typeof(double)}),
      typeof(Functions).GetMethod("AQ", new[] {typeof(double), typeof(double) }),
      typeof(Functions).GetMethod("Logistic", new [] { typeof(double) }),
      typeof(Functions).GetMethod("InvLogistic", new [] { typeof(double) }),
      typeof(Functions).GetMethod("LogisticPrime", new [] { typeof(double) }),
      typeof(Functions).GetMethod("LogisticPrimePrime", new [] { typeof(double) }),
      typeof(Functions).GetMethod("InvLogisticPrime", new [] { typeof(double) }),
      typeof(Functions).GetMethod("InvLogisticPrimePrime", new [] { typeof(double) }),
    };

    private CheckExprVisitor(ParameterExpression theta, ParameterExpression x) {
      this.theta = theta;
      this.x = x;
      this.parameterOccurances = new Dictionary<int, int>(); // index -> freq
    }

    public static bool CheckValid(Expression<Expr.ParametricFunction> expr) {
      var visitor = new CheckExprVisitor(expr.Parameters[0], expr.Parameters[1]);
      visitor.Visit(expr.Body);

      return true; // otherwise Visit throws an Exception
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      if (node.NodeType != ExpressionType.Add &&
           node.NodeType != ExpressionType.Subtract &&
           node.NodeType != ExpressionType.Multiply &&
           node.NodeType != ExpressionType.Divide &&
           (node.NodeType != ExpressionType.ArrayIndex || !ArrayIndexValid(node))
          ) throw new NotSupportedException(node.ToString());

      if (node.NodeType == ExpressionType.ArrayIndex && node.Left == theta) {
        var paramIdx = (int)((ConstantExpression)node.Right).Value;
        if (parameterOccurances.ContainsKey(paramIdx))
          parameterOccurances[paramIdx]++;
        else
          parameterOccurances.Add(paramIdx, 1);
      }
      return base.VisitBinary(node);
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      if (node.NodeType != ExpressionType.Negate && node.NodeType != ExpressionType.UnaryPlus) {
        throw new NotSupportedException(node.ToString());
      }
      return base.VisitUnary(node);
    }

    protected override Expression VisitMethodCall(MethodCallExpression node) {
      if (!SupportedMethods.Contains(node.Method)) throw new NotSupportedException($"unsupported method {node.Method}");
      return base.VisitMethodCall(node);
    }

    protected override Expression VisitParameter(ParameterExpression node) {
      if (node != theta && node != x) new NotSupportedException(node.ToString());
      return base.VisitParameter(node);
    }

    private bool ArrayIndexValid(BinaryExpression node) {
      var left = node.Left;
      // only theta and x allowed
      if (left != theta && left != x) throw new NotSupportedException($"other variables than theta and x are not allowed {left}");
      if (!(node.Right is ConstantExpression)) throw new NotSupportedException($"array index must be constant {node.Right}");
      return true;
    }

    #region unsupported
    protected override Expression VisitBlock(BlockExpression node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override Expression VisitConditional(ConditionalExpression node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override Expression VisitDynamic(DynamicExpression node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override Expression VisitExtension(Expression node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override Expression VisitDefault(DefaultExpression node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override Expression VisitGoto(GotoExpression node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override Expression VisitIndex(IndexExpression node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override Expression VisitInvocation(InvocationExpression node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override Expression VisitLabel(LabelExpression node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override Expression VisitLambda<T>(Expression<T> node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override Expression VisitLoop(LoopExpression node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override Expression VisitMember(MemberExpression node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override Expression VisitNew(NewExpression node) {
      throw new NotSupportedException(node.ToString());
    }


    protected override Expression VisitSwitch(SwitchExpression node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override Expression VisitTry(TryExpression node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override CatchBlock VisitCatchBlock(CatchBlock node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override ElementInit VisitElementInit(ElementInit node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override LabelTarget VisitLabelTarget(LabelTarget node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override Expression VisitListInit(ListInitExpression node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override MemberAssignment VisitMemberAssignment(MemberAssignment node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override MemberBinding VisitMemberBinding(MemberBinding node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override Expression VisitMemberInit(MemberInitExpression node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override Expression VisitNewArray(NewArrayExpression node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override Expression VisitRuntimeVariables(RuntimeVariablesExpression node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override SwitchCase VisitSwitchCase(SwitchCase node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override Expression VisitTypeBinary(TypeBinaryExpression node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override MemberListBinding VisitMemberListBinding(MemberListBinding node) {
      throw new NotSupportedException(node.ToString());
    }

    protected override MemberMemberBinding VisitMemberMemberBinding(MemberMemberBinding node) {
      throw new NotSupportedException(node.ToString());
    }
    #endregion
  }
}

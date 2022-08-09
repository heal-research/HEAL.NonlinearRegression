using System;
using System.Drawing;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Xml;
using Type = System.Type;

namespace HEAL.Expressions {
  public static class Expressions {
    public class ParameterBindingVisitor : ExpressionVisitor {
      private ParameterExpression oldParam;
      private ParameterExpression newParam;

      public Expression BindParameter(Expression<Func<double[], double>> expr, ParameterExpression oldParam,
        ParameterExpression newParam) {
        this.oldParam = oldParam;
        this.newParam = newParam;
        return Visit(expr.Body);
      }

      protected override Expression VisitParameter(ParameterExpression node) {
        if (node == oldParam) return newParam;
        else return base.VisitParameter(node);
      }
    }

    public static Expression<Func<double[,], double[]>> Broadcast(Expression<Func<double[], double>> fx) {
      var fx_param = fx.Parameters.First();
      var endLoop = Expression.Label("endloop");
      var endLambda = Expression.Label("endLambda");
      var iVar = Expression.Variable(typeof(int), "i"); // loop counter
      var nVar = Expression.Variable(typeof(int), "n");
      var dVar = Expression.Variable(typeof(int), "d");
      var resVar = Expression.Variable(typeof(double[]), "res");
      var xParam = Expression.Parameter(typeof(double[,]), "x");
      var vVar = Expression.Variable(typeof(double[]), "v"); // temp buffer
      var getLength = typeof(double[,]).GetMethod("GetLength",
        new Type[] { typeof(int) });
      var blockCopy = typeof(Buffer).GetMethod("BlockCopy",
        new Type[] { typeof(Array), typeof(int), typeof(Array), typeof(int), typeof(int) });

      // for debugging
      var writeLn = typeof(Console).GetMethod("WriteLine", new Type[] { typeof(string) });
      var toString = typeof(object).GetMethod("ToString");

      return Expression.Lambda<Func<double[,], double[]>>(
        Expression.Block(
          new[] { nVar, iVar, dVar, vVar, resVar },
          new Expression[] {
            Expression.Assign(nVar, Expression.Call(xParam, getLength, Expression.Constant(0))),
            // Expression.Call(writeLn, Expression.Call(nVar, toString)),
            Expression.Assign(dVar, Expression.Call(xParam, getLength, Expression.Constant(1))),
            Expression.Assign(resVar, Expression.NewArrayBounds(typeof(double), nVar)),
            Expression.Assign(vVar, Expression.NewArrayBounds(typeof(double), dVar)),
            Expression.Assign(iVar, Expression.Constant(0)),
            Expression.Loop(Expression.Block(
                Expression.IfThen(Expression.Equal(iVar, nVar),
                  Expression.Break(endLoop)),
                Expression.Call(blockCopy, xParam,
                  Expression.Multiply(iVar, Expression.Multiply(dVar, Expression.Constant(sizeof(double)))),
                  vVar, Expression.Constant(0),
                  Expression.Multiply(dVar, Expression.Constant(sizeof(double)))),
                Expression.Assign(Expression.ArrayAccess(resVar, iVar),
                  (new ParameterBindingVisitor()).BindParameter(fx, fx_param, vVar)),
                Expression.PostIncrementAssign(iVar)
              )
              , endLoop),
            resVar
          }), xParam);
    }

    public static Expression<Func<double[], double>> Derive(Expression<Func<double[], double>> expr, int dxIdx) {
      var deriveVisitor = new DeriveVisitor(expr.Parameters.First(), dxIdx);
      var simplifyVisitor = new SimplifyVisitor();
      return (Expression<Func<double[], double>>)simplifyVisitor.Visit(deriveVisitor.Visit(expr));
    }

    public static Expression Simplify(Expression expr) {
      var simplifyVisitor = new SimplifyVisitor();
      return simplifyVisitor.Visit(expr);
    }

    /*
    public class ReplaceParamVisitor : ExpressionVisitor {
      private string[] paramNames;

      public Expression Replace(string[] paramNames, Expression<Func<double, double, double>> expr) {
        this.paramNames = paramNames;
        return Visit(expr.Body);
      }

      protected override Expression VisitParameter(ParameterExpression node) {
        if (node.Name == "a") return Expression.Parameter(typeof(double), paramNames[0]);
        else if (node.Name == "b") return Expression.Parameter(typeof(double), paramNames[1]);
        else return base.VisitParameter(node);
      }
    }*/
 }

  public class SimplifyVisitor : ExpressionVisitor {
    protected override Expression VisitBinary(BinaryExpression node) {
      var left = Visit(node.Left);
      var right = Visit(node.Right);
      var leftConst = left as ConstantExpression;
      var rightConst = right as ConstantExpression;
      switch (node.NodeType) {
        case ExpressionType.Add: {
          if (leftConst != null && rightConst != null) return Expression.Constant((double)leftConst.Value + (double)rightConst.Value);
          else if (leftConst != null && leftConst.Value.Equals(0.0)) return right;
          else if (rightConst != null && rightConst.Value.Equals(0.0)) return left;
          else return node.Update(left, null, right);
        }
        case ExpressionType.Subtract: {
          if (leftConst != null && rightConst != null) return Expression.Constant((double)leftConst.Value - (double)rightConst.Value);
          else if (leftConst!=null && leftConst.Value.Equals(0.0)) return Expression.Negate(right);
          else if(rightConst != null && rightConst.Value.Equals(0.0)) return left;
          else return node.Update(left, null, right);
        }
        case ExpressionType.Multiply: {
          if (leftConst != null && rightConst != null) return Expression.Constant((double)leftConst.Value * (double)rightConst.Value);
          else if (leftConst != null && leftConst.Value.Equals(0.0)) return Expression.Constant(0.0);
          else if (leftConst != null && leftConst.Value.Equals(1.0)) return right;
          else if (rightConst != null && rightConst.Value.Equals(0.0)) return Expression.Constant(0.0);
          else if (rightConst != null && rightConst.Value.Equals(1.0)) return left;
          else return node.Update(left, null, right);
        }
      }
      return node.Update(left, null, right);
    }

    protected override Expression VisitUnary(UnaryExpression node) {
      var x = Visit(node.Operand);
      if(node.NodeType == ExpressionType.Negate && 
         x is ConstantExpression xConst) return Expression.Constant(-1.0 * (double)xConst.Value);
      return node.Update(x);
    }

    protected override Expression VisitMethodCall(MethodCallExpression node) {
      var args = node.Arguments.Select(Visit).ToArray();

      // method is static and 
      // all arguments are constant doubles
      // -> call the method (we don't care which method it is
      if (node.Method.IsStatic &&
        args.All(arg => arg.NodeType == ExpressionType.Constant 
                          && arg.Type == typeof(double))) {
        var values = args.Select(arg => ((ConstantExpression)arg).Value).ToArray();
        return Expression.Constant(node.Method.Invoke(node.Object, values));
      }

      return node.Update(node.Object, args);
    }
  }

  public class DeriveVisitor : ExpressionVisitor {
    private readonly int dxIdx;
    private readonly ParameterExpression param;

    public DeriveVisitor(ParameterExpression param, int dxIdx) {
      this.param = param;
      this.dxIdx = dxIdx;
    }

    public Expression<Func<double[], double>> Visit(Expression<Func<double[], double>> expr) {
      return Expression.Lambda<Func<double[], double>>(Visit(expr.Body), expr.Parameters);
    }

    protected override Expression VisitConstant(ConstantExpression node) {
      return Expression.Constant(0.0);
    }
    
    protected override Expression VisitBinary(BinaryExpression node) {
      Console.WriteLine(node);
      switch (node.NodeType) {
        case ExpressionType.Add: {
          return Expression.Add(Visit(node.Left), Visit(node.Right));
        }
        case ExpressionType.Subtract: {
          return Expression.Subtract(Visit(node.Left), Visit( node.Right));
        }
        case ExpressionType.Multiply: {
          return Expression.Add(
            Expression.Multiply(node.Left, Visit(node.Right)),
            Expression.Multiply(node.Right, Visit(node.Left)));
        }
        case ExpressionType.ArrayIndex: {
          if (node.Left == param) {
            var firstIndex = node.Right;
            if (firstIndex.NodeType == ExpressionType.Constant) {
              var idx = (int)((ConstantExpression)firstIndex).Value;
              if (idx == dxIdx) return Expression.Constant(1.0);
              else return Expression.Constant(0.0);
            } else throw new NotSupportedException("only constant indices for parameter are allowed");
          } else return base.VisitBinary(node); // array index for other variables or parameters ok
        }
        default: throw new NotSupportedException(node.ToString());
      }

      return base.VisitBinary(node);
    }

    private MethodInfo sin = typeof(Math).GetMethod("Sin", new[] { typeof(double) });
    private MethodInfo cos = typeof(Math).GetMethod("Sin", new[] { typeof(double) });

    protected override Expression VisitMethodCall(MethodCallExpression node) {
      if (node.Method == sin) {
        var x = node.Arguments.First();
        return Expression.Multiply(Expression.Call(cos, x), Visit(x));
      } else if (node.Method == cos) {
        var x = node.Arguments.First();
        return Expression.Negate(Expression.Multiply(Expression.Call(sin, x), Visit(x)));
      }

      return base.VisitMethodCall(node);
    }

    
  }
}

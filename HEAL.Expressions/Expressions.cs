using System;
using System.Drawing;
using System.Linq;
using System.Linq.Expressions;
using System.Xml;
using Type = System.Type;

namespace HEAL.Expressions {
  public static class Expressions {

    /// <summary>
    /// Broadcasts an expression to work on an array. 
    /// </summary>
    /// <param name="fx">An expression that takes a double[] and returns a double.</param>
    /// <returns>A new expression that takes and double[,] input and returns a double[].</returns>
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

    
    /// <summary>
    /// Takes an expression and generates it's partial derivative.
    /// </summary>
    /// <param name="expr">An expression that takes a double[] parameter and returns a double. </param>
    /// <param name="dxIdx">The parameter index for which to calculate generate the partial derivative</param>
    /// <returns>A new expression that calculates the partial derivative of d expr(x) / d x_i</returns>
    public static Expression<Func<double[], double>> Derive(Expression<Func<double[], double>> expr, int dxIdx) {
      var deriveVisitor = new DeriveVisitor(expr.Parameters.First(), dxIdx);
      var simplifyVisitor = new SimplifyVisitor();
      return (Expression<Func<double[], double>>)simplifyVisitor.Visit(deriveVisitor.Visit(expr));
    }

    /// <summary>
    /// Takes an expression and folds double constants. 
    /// </summary>
    /// <param name="expr">The expression to simplify</param>
    /// <returns>A new expression with folded double constants.</returns>
    public static Expression Simplify(Expression expr) {
      var simplifyVisitor = new SimplifyVisitor();
      return simplifyVisitor.Visit(expr);
    }
  }
}

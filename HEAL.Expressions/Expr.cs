using System;
using System.Drawing;
using System.Linq;
using System.Linq.Expressions;
using System.Xml;
using Type = System.Type;
//using ParametricFunction=System.Func<double[], double[], double>;
//using ParametricExpression=System.Linq.Expressions.Expression<System.Func<double[], double[], double>>;
//using ParametricVectorFunction = System.Func<double[], double[,], double[]>;
//using ParametricVectorExpression = System.Linq.Expressions.Expression<System.Func<double[], double[,], double[]>>;
//using ParametricGradientFunction = System.Func<double[], double[], System.Tuple<double, double[]>>;
//using ParametricGradientExpression = System.Linq.Expressions.Expression<System.Func<double[], double[], System.Tuple<double, double[]>>>;
//using ParametricJacobianFunction = System.Func<double[], double[,], System.Tuple<double[], double[,]>>;
//using ParametricJacobianExpression = System.Linq.Expressions.Expression<System.Func<double[], double[,], System.Tuple<double[], double[,]>>>;

namespace HEAL.Expressions {
  public static class Expr {
    public delegate double ParametricFunction(double[] theta, double[] x);

    public delegate void ParametricVectorFunction(double[] theta, double[,] X, double[] f);

    public delegate double ParametricGradientFunction(double[] theta, double[] x, double[] grad);

    public delegate void ParametricJacobianFunction(double[] theta, double[,] X, double[] f, double[,] Jac);
    
    
    // TODO method to check validity of model expression
    // - can only contain variables theta and x
    // - theta and x are always indexed and always indexed with constants. The model cannot determine x.length or
    //   call methods for x or theta. LINQ Extension methods e.g. theta.Select() or theta.Zip(x, ...) are not supported
    // - Parameters must be used only once in the expression. This is not strictly necessary, but the code assumes
    //   this for now.
    // - Only static methods of double parameters returning double (and one of a small list of supported methods, Log, Exp, Sin, Cos, ...)

    /// <summary>
    /// Broadcasts an expression to work on an array. 
    /// </summary>
    /// <param name="fx">An expression that takes a double[] and returns a double.</param>
    /// <returns>A new expression that takes and double[,] input and returns a double[].</returns>
    public static Expression<ParametricVectorFunction> Broadcast(Expression<ParametricFunction> fx) {
      // original parameters
      var thetaParam = fx.Parameters[0];
      var xVecParam = fx.Parameters[1];
      
      var endLoop = Expression.Label("endloop");
      
      // local variables
      var iVar = Expression.Variable(typeof(int), "i"); // loop counter
      var nVar = Expression.Variable(typeof(int), "n");
      var dVar = Expression.Variable(typeof(int), "d");
      var vVar = Expression.Variable(typeof(double[]), "v"); // temp buffer

      // new parameters
      var fVar = Expression.Parameter(typeof(double[]), "f");
      var xMatrixParam = Expression.Parameter(typeof(double[,]), "x");
      
      // necessary methods
      var getLength = typeof(double[,]).GetMethod("GetLength",
        new Type[] { typeof(int) });
      var blockCopy = typeof(Buffer).GetMethod("BlockCopy",
        new Type[] { typeof(Array), typeof(int), typeof(Array), typeof(int), typeof(int) });

      // for debugging
      var writeLn = typeof(Console).GetMethod("WriteLine", new Type[] { typeof(string) });
      var toString = typeof(object).GetMethod("ToString");

      var adjExpr = (new ParameterReplacementVisitor()).ReplaceParameter(fx.Body, xVecParam, vVar);
      // Console.WriteLine(adjExpr);
      
      return Expression.Lambda<ParametricVectorFunction>(
        Expression.Block(
          new[] { nVar, iVar, dVar, vVar },
          new Expression[] {
            Expression.Assign(nVar, Expression.Call(xMatrixParam, getLength, Expression.Constant(0))),
            // Expression.Call(writeLn, Expression.Call(nVar, toString)),
            Expression.Assign(dVar, Expression.Call(xMatrixParam, getLength, Expression.Constant(1))),
            Expression.Assign(vVar, Expression.NewArrayBounds(typeof(double), dVar)),
            Expression.Assign(iVar, Expression.Constant(0)),
            Expression.Loop(Expression.Block(
                Expression.IfThen(Expression.Equal(iVar, nVar),
                  Expression.Break(endLoop)),
                Expression.Call(blockCopy, xMatrixParam,
                  Expression.Multiply(iVar, Expression.Multiply(dVar, Expression.Constant(sizeof(double)))),
                  vVar, Expression.Constant(0),
                  Expression.Multiply(dVar, Expression.Constant(sizeof(double)))),
                Expression.Assign(Expression.ArrayAccess(fVar, iVar), adjExpr),
                Expression.PostIncrementAssign(iVar)
              )
              , endLoop),
          }), thetaParam, xMatrixParam, fVar);
    }

    public static Expression<ParametricJacobianFunction> Broadcast(Expression<ParametricGradientFunction> fx) {
      // original parameters
      var thetaParam = fx.Parameters[0];
      var xVecParam = fx.Parameters[1];
      var gVecParam = fx.Parameters[2];
      
      var endLoop = Expression.Label("endloop");
      
      // local variables
      var iVar = Expression.Variable(typeof(int), "i"); // loop counter
      var nVar = Expression.Variable(typeof(int), "n"); // nrows(X)
      var dVar = Expression.Variable(typeof(int), "d"); // len(x)
      var kVar = Expression.Variable(typeof(int), "k"); // len(theta)
      var vVar = Expression.Variable(typeof(double[]), "v"); // temp buffer
      var gVar = Expression.Variable(typeof(double[]), "g"); // temp buffer

      // new parameters
      var fVar = Expression.Parameter(typeof(double[]), "f");
      var xMatrixParam = Expression.Parameter(typeof(double[,]), "x");
      var jacParam = Expression.Parameter(typeof(double[,]), "Jac");
      
      // necessary methods
      var getLength = typeof(double[,]).GetMethod("GetLength",
        new Type[] { typeof(int) });
      var blockCopy = typeof(Buffer).GetMethod("BlockCopy",
        new Type[] { typeof(Array), typeof(int), typeof(Array), typeof(int), typeof(int) });

      // for debugging
      // var writeLn = typeof(Console).GetMethod("WriteLine", new Type[] { typeof(string) });
      // var toString = typeof(object).GetMethod("ToString");

      return Expression.Lambda<ParametricJacobianFunction>(
        Expression.Block(
          new[] { nVar, iVar, dVar, kVar, vVar, gVar },
          new Expression[] {
            Expression.Assign(nVar, Expression.Call(xMatrixParam, getLength, Expression.Constant(0))),
            // Expression.Call(writeLn, Expression.Call(nVar, toString)),
            Expression.Assign(dVar, Expression.Call(xMatrixParam, getLength, Expression.Constant(1))),
            Expression.Assign(kVar, Expression.ArrayLength(thetaParam)),
            Expression.Assign(vVar, Expression.NewArrayBounds(typeof(double), dVar)),
            Expression.Assign(gVar, Expression.NewArrayBounds(typeof(double), kVar)),
            Expression.Assign(iVar, Expression.Constant(0)),
            Expression.Loop(Expression.Block(
                Expression.IfThen(Expression.Equal(iVar, nVar),
                  Expression.Break(endLoop)),
                // copy row i from X to v
                Expression.Call(blockCopy, xMatrixParam,
                  Expression.Multiply(iVar, Expression.Multiply(dVar, Expression.Constant(sizeof(double)))),
                  vVar, Expression.Constant(0),
                  Expression.Multiply(dVar, Expression.Constant(sizeof(double)))),
                // call gradient function (v, g)
                Expression.Assign(Expression.ArrayAccess(fVar, iVar),
                  (new ParameterReplacementVisitor()).ReplaceParameters(fx.Body, new [] {xVecParam, gVecParam}, 
                    new [] {vVar, gVar})),
                // copy g to Jac row i
                Expression.Call(blockCopy, gVar, 
                  Expression.Constant(0), 
                  jacParam, Expression.Multiply(iVar, Expression.Multiply(kVar, Expression.Constant(sizeof(double)))),
                  Expression.Multiply(kVar, Expression.Constant(sizeof(double)))
                  ),
                Expression.PostIncrementAssign(iVar)
              )
              , endLoop),
          }), thetaParam, xMatrixParam, fVar, jacParam);
    }

    
    /// <summary>
    /// Takes an expression and generates it's partial derivative.
    /// </summary>
    /// <param name="expr">An expression that takes a double[] parameter and returns a double. </param>
    /// <param name="dxIdx">The parameter index for which to calculate generate the partial derivative</param>
    /// <returns>A new expression that calculates the partial derivative of d expr(x) / d x_i</returns>
    public static Expression<ParametricFunction> Derive(Expression<ParametricFunction> expr, int dxIdx) {
      if(!CheckExprVisitor.CheckValid(expr)) throw new NotSupportedException(expr.ToString());
      var deriveVisitor = new DeriveVisitor(expr.Parameters.First(), dxIdx);
      return (Expression<ParametricFunction>)Simplify(deriveVisitor.Visit(expr));
    }

    // returns function that also returns the gradient 
    public static Expression<ParametricGradientFunction> Gradient(Expression<ParametricFunction> expr, int numParam) {
      var gParam = Expression.Parameter(typeof(double[]), "g");
      var fVar = Expression.Variable(typeof(double), "f");
      var df_dx = new Expression<ParametricFunction>[numParam];
      for (int i = 0; i < numParam; i++) {
        df_dx[i] = Derive(expr, i);
        // Console.WriteLine(df_dx[i]);
      }

      var expressions = new Expression[numParam + 2];
      // Console.WriteLine(expr.Body);
      
      expressions[0] = Expression.Assign(fVar, expr.Body); // f(x)
      // Console.WriteLine(expressions[0]);
      for (int i = 0; i < numParam; i++) {
        expressions[i + 1] = Expression.Assign(Expression.ArrayAccess(gParam, Expression.Constant(i)), df_dx[i].Body);
        // Console.WriteLine(expressions[i + 1]);
      }

      expressions[expressions.Length - 1] = fVar; // result of the block is the result of the last expression
      
      

      var res = Expression.Lambda<ParametricGradientFunction>(
        Expression.Block(
            new [] {fVar}, // block local variables
            expressions
          ), 
        expr.Parameters.Concat(new  [] {gParam}) // lambda parameters
        );
      // Console.WriteLine(res);
      return res;
    }

    public static Expression<ParametricJacobianFunction> Jacobian(Expression<ParametricFunction> expr, int numParam) {
      return Broadcast(Gradient(expr, numParam));
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

    public static Expression<ParametricFunction> RemoveRedundantParameters(Expression<ParametricFunction> expr, double[] parameterValues, out double[] newParameterValues) {
      var theta = expr.Parameters[0];
      var x = expr.Parameters[1];
      var visitor = new RemoveRedundantParametersVisitor(theta, parameterValues);
      var newExpr = visitor.Visit(expr);
      newParameterValues = visitor.GetNewParameterValues;
      return (Expression<ParametricFunction>)newExpr;
    }

    public static Expression<ParametricFunction> ReplaceVariableWithParameter(Expression<ParametricFunction> expr,
      double[] thetaValues, int varIdx, double replVal) {
      var theta = expr.Parameters[0];
      var x = expr.Parameters[1];
      var visitor = new ReplaceVariableWithParameterVisitor(theta, thetaValues,x, varIdx, replVal);
      return (Expression<ParametricFunction>)visitor.Visit(expr);
    }
    
    // TODO: method to take an expression and extract all double constants as parameters
    // ( so that we can directly copy operon models into the code)
  }
}

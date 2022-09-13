using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using Type = System.Type;

// TODO:
// - refactor visitors (move into folder?)
// - fold constants visitor

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
      // var writeLn = typeof(Console).GetMethod("WriteLine", new Type[] { typeof(string) });
      // var toString = typeof(object).GetMethod("ToString");

      var adjExpr = (new SubstituteParameterVisitor()).ReplaceParameter(fx.Body, xVecParam, vVar);
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


    public static Expression<T> ReplaceParameterWithValues<T>(LambdaExpression expr,
      ParameterExpression parameter,
      double[] parameterValues) {
      var v = new ReplaceParameterWithNumberVisitor(parameter, parameterValues);
      var newExpr = v.Visit(expr.Body);
      return Expression.Lambda<T>(newExpr, expr.Parameters.Except(new[] { parameter }));
    }

    /// <summary>
    /// Returns a new expression where all double values are replaced by a reference to a new double[] parameter theta[.]. 
    /// The values in expr are collected into thetaValues.
    /// </summary>
    /// <param name="expr">The expr which contains double constants.</param>
    /// <param name="thetaValues">Array of parameter values occuring in expr</param>
    /// <returns></returns>
    public static LambdaExpression ReplaceNumbersWithParameter(LambdaExpression expr, out double[] thetaValues) {
      var theta = Expression.Parameter(typeof(double[]), "theta");
      var v = new ReplaceNumberWithParameterVisitor(theta);
      var newExpr = v.Visit(expr.Body);
      thetaValues = v.ParameterValues;
      return Expression.Lambda(newExpr, new[] { theta }.Concat(expr.Parameters));
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
                  (new SubstituteParameterVisitor()).ReplaceParameters(fx.Body, new [] {xVecParam, gVecParam},
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
    /// Takes an expression and generates its partial derivative.
    /// </summary>
    /// <param name="expr">A parametric expression. </param>
    /// <param name="dxIdx">The parameter index for which to calculate generate the partial derivative</param>
    /// <returns>A new expression that calculates the partial derivative of d expr(x) / d x_i</returns>
    public static Expression<ParametricFunction> Derive(Expression<ParametricFunction> expr, int dxIdx) {
      if (!CheckExprVisitor.CheckValid(expr)) throw new NotSupportedException(expr.ToString());
      var deriveVisitor = new DeriveVisitor(expr.Parameters.First(), dxIdx);
      return Simplify((Expression<ParametricFunction>)deriveVisitor.Visit(expr));
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
            new[] { fVar }, // block local variables
            expressions
          ),
        expr.Parameters.Concat(new[] { gParam }) // lambda parameters
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
    public static Expression<ParametricFunction> Simplify(Expression<ParametricFunction> expr) {
      var theta = expr.Parameters[0];
      var simplifyVisitor = new SimplifyVisitor();
      return (Expression<ParametricFunction>)simplifyVisitor.Visit(expr);
    }


    /// <summary>
    /// As simplify but also reduces the length of the parameter vector to include only the parameters
    /// that are still referenced in the simplified expression. 
    /// </summary>
    /// <param name="expr"></param>
    /// <param name="thetaValues"></param>
    /// <param name="newThetaValues"></param>
    /// <returns></returns>
    public static Expression<ParametricFunction> SimplifyAndRemoveParameters(Expression<ParametricFunction> expr, double[] thetaValues, out double[] newThetaValues) {
      var theta = expr.Parameters[0];
      var simplifyVisitor = new SimplifyVisitor();
      var simplifiedExpr = (Expression<ParametricFunction>)simplifyVisitor.Visit(expr);
      var collectParamVisitor = new CollectParametersVisitor(theta, thetaValues);
      simplifiedExpr = (Expression<ParametricFunction>)collectParamVisitor.Visit(simplifiedExpr);
      newThetaValues = collectParamVisitor.GetNewParameterValues;
      return simplifiedExpr;
    }

    public static Expression<ParametricFunction> FoldParameters(Expression<ParametricFunction> expr,
      double[] parameterValues, out double[] newParameterValues) {
      var theta = expr.Parameters[0];
      var x = expr.Parameters[1];
      var rotateVisitor = new RotateBinaryExpressionsVisitor();
      expr = (Expression<ParametricFunction>)rotateVisitor.Visit(expr);
      // Console.WriteLine($"Rotated: {expr}");
      expr = ArrangeParametersRightVisitor.Execute(expr, theta, parameterValues);
      // Console.WriteLine($"Rearranged: {expr}");

      var visitor = new FoldParametersVisitor(theta, parameterValues);
      var newExpr = (Expression<ParametricFunction>)visitor.Visit(expr);
      // Console.WriteLine($"Folded: {newExpr}");

      newParameterValues = visitor.GetNewParameterValues;
      //Console.WriteLine($"Folded parameters: {newExpr}");

      var collectVisitor = new CollectParametersVisitor(theta, newParameterValues);
      newExpr = (Expression<ParametricFunction>)collectVisitor.Visit(newExpr);
      newParameterValues = collectVisitor.GetNewParameterValues;
      //Console.WriteLine($"Removed unused parameters: {newExpr}");
      return (Expression<ParametricFunction>)newExpr;
    }

    public static Expression<ParametricFunction> ReplaceVariableWithParameter(Expression<ParametricFunction> expr,
      double[] thetaValues, int varIdx, double replVal, out double[] newThetaValues) {
      var theta = expr.Parameters[0];
      var x = expr.Parameters[1];
      // Console.WriteLine($"Original: {expr}:");
      var visitor = new ReplaceVariableWithParameterVisitor(theta, thetaValues, x, varIdx, replVal);
      var newExpr = (Expression<ParametricFunction>)visitor.Visit(expr);
      // Console.WriteLine($"x{varIdx} replaced: {newExpr}:");
      newThetaValues = visitor.NewThetaValues;
      return newExpr;
    }

    public static string ToGraphViz(Expression<ParametricFunction> expr,
      double[] paramValues = null, string[] varNames = null, Dictionary<Expression, double> saturation = null) {
      return GraphvizVisitor.Execute(expr, paramValues, varNames, saturation);
    }

    public static string ToString(Expression<ParametricFunction> expr, string[] varNames, double[] p) {
      // for the output
      var parameterizedExpression = Expr.ReplaceParameterWithValues<Func<double[], double>>(expr, expr.Parameters[0], p);

      var exprBody = parameterizedExpression.Body.ToString();
      for (int i = 0; i < varNames.Length; i++) {
        exprBody = exprBody.Replace($"x[{i}]", varNames[i]);
      }

      return exprBody.ToString();

    }

    // TODO: method to take an expression and extract all double constants as parameters
    // ( so that we can directly copy operon models into the code)
  }
}

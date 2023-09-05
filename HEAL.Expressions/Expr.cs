using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Linq.Expressions;
using System.Text.RegularExpressions;
using Type = System.Type;

// TODO:
// - refactor visitors (move into folder?)
// - extract patterns in txt files and read patterns as Expressions with a separate parser. Match the patterns against the expression tree
//   -> this is to simplify the pattern matching / expression simplification code
//   -> patterns can be build step by step by replacing one visitor at a time
//   -> pattern sets need to be grouped. E.g. there is a separate group of patterns to fold constants, take derivative, ...


namespace HEAL.Expressions {
  public class ParameterizedExpression {
    public readonly Expression<Expr.ParametricFunction> expr;
    public readonly ParameterExpression p;
    public readonly double[] pValues;

    public ParameterizedExpression(Expression<Expr.ParametricFunction> expr, ParameterExpression p, double[] pValues) {
      this.expr = expr;
      this.p = p;
      this.pValues = pValues;
    }
  }

  public static class Expr {
    // TODO: check which types we actually need
    public delegate double ParametricFunction(double[] theta, double[] x);

    public delegate void ParametricVectorFunction(double[] theta, double[,] X, double[] f);

    public delegate double ParametricGradientFunction(double[] theta, double[] x, double[] grad);

    public delegate double ParametricHessianFunction(double[] theta, double[] x, double[,] hessian);

    public delegate double ParametricHessianDiagFunction(double[] theta, double[] x, double[] hessianDiag);

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
    /// Takes an expression and generates its partial derivative (for p[dxIdx]).
    /// </summary>
    /// <param name="expr">A parametric expression. </param>
    /// <param name="dxIdx">The parameter index for which to calculate generate the partial derivative</param>
    /// <returns>A new expression that calculates the partial derivative of d expr(x) / d x_i</returns>
    public static Expression<ParametricFunction> Derive(Expression<ParametricFunction> expr, int dxIdx) {
      return Derive(expr, expr.Parameters[0], dxIdx);
    }

    /// <summary>
    /// Takes and expression and generates its partial derivative for param[dxIdx]
    /// </summary>
    /// <param name="expr"></param>
    /// <param name="param"></param>
    /// <param name="dxIdx"></param>
    /// <returns></returns>
    /// <exception cref="NotSupportedException"></exception>
    public static Expression<ParametricFunction> Derive(Expression<ParametricFunction> expr, ParameterExpression param, int dxIdx) {
      if (!CheckExprVisitor.CheckValid(expr)) throw new NotSupportedException(expr.ToString());
      var deriveVisitor = new DeriveVisitor(param, dxIdx);
      var df = (Expression<ParametricFunction>)deriveVisitor.Visit(expr);
      // here we do not care about parameter values
      var zero = new double[CountParametersVisitor.Count(df, df.Parameters[0])];
      return SimplifyWithoutReparameterization(new ParameterizedExpression(df, df.Parameters[0], zero)).expr;
    }

    // Symbolic gradient
    // returns function that also returns the gradient 
    public static Expression<ParametricGradientFunction> Gradient(Expression<ParametricFunction> expr, int numParam) {
      var gParam = Expression.Parameter(typeof(double[]), "grad");
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

    // symbolic Hessian
    public static Expression<ParametricHessianFunction> Hessian(Expression<ParametricFunction> expr, int numParam) {
      var hParam = Expression.Parameter(typeof(double[,]), "hessian");
      var fVar = Expression.Variable(typeof(double), "f");
      var H = new Expression<ParametricFunction>[numParam, numParam];
      for (int i = 0; i < numParam; i++) {
        var df_dx = Derive(expr, i);
        for (int j = i; j < numParam; j++) {
          H[i, j] = Derive(df_dx, j);
          if (i != j)
            H[j, i] = H[i, j];
        }
      }

      var statements = new List<Expression> {
        Expression.Assign(fVar, expr.Body)
      };

      // diagonal elements of Hessian
      for (int i = 0; i < numParam; i++) {
        statements.Add(Expression.Assign(Expression.ArrayAccess(hParam, Expression.Constant(i), Expression.Constant(i)), H[i, i].Body));
      }
      // off-diagonal elements of Hessian
      for (int i = 0; i < numParam; i++) {
        for (int j = i + 1; j < numParam; j++) {
          statements.Add(Expression.Assign(Expression.ArrayAccess(hParam, Expression.Constant(i), Expression.Constant(j)), H[i, j].Body));
          statements.Add(Expression.Assign(Expression.ArrayAccess(hParam, Expression.Constant(j), Expression.Constant(i)), Expression.ArrayAccess(hParam, Expression.Constant(i), Expression.Constant(j))));
          // Console.WriteLine(expressions[i + 1]);
        }
      }

      statements.Add(fVar); // result of the block is the result of the last expression

      var res = Expression.Lambda<ParametricHessianFunction>(
        Expression.Block(
            new[] { fVar }, // block local variables
            statements
          ),
        expr.Parameters.Concat(new[] { hParam }) // lambda parameters
        );
      return res;
    }

    // symbolic diagonal of Hessian matrix
    public static Expression<ParametricHessianDiagFunction> HessianDiag(Expression<ParametricFunction> expr, int numParam) {
      var hParam = Expression.Parameter(typeof(double[]), "hessianDiag");
      var fVar = Expression.Variable(typeof(double), "f");
      var H = new Expression<ParametricFunction>[numParam];
      for (int i = 0; i < numParam; i++) {
        H[i] = Derive(Derive(expr, i), i);
      }

      var statements = new List<Expression> {
        Expression.Assign(fVar, expr.Body)
      };

      // diagonal elements of Hessian
      for (int i = 0; i < numParam; i++) {
        statements.Add(Expression.Assign(Expression.ArrayAccess(hParam, Expression.Constant(i)), H[i].Body));
      }
      statements.Add(fVar); // result of the block is the result of the last expression

      var res = Expression.Lambda<ParametricHessianDiagFunction>(
        Expression.Block(
            new[] { fVar }, // block local variables
            statements
          ),
        expr.Parameters.Concat(new[] { hParam }) // lambda parameters
        );
      return res;
    }

    public static Expression<ParametricJacobianFunction> Jacobian(Expression<ParametricFunction> expr, int numParam) {
      return Broadcast(Gradient(expr, numParam));
    }


    /// <summary>
    /// Simplifies an expression by folding constants only. 
    /// </summary>
    /// <param name="expr">The expression to simplify</param>
    /// <returns>A new expression with folded constants.</returns>
    public static ParameterizedExpression FoldConstants(ParameterizedExpression expr) {
      return RuleBasedSimplificationVisitor.FoldConstants(expr);
    }

    /// <summary>
    /// Simplifies and expression without reparameterization (all parameters keep their values). 
    /// </summary>
    /// <param name="expr">The expression to simplify</param>
    /// <returns>A new expression with folded double constants.</returns>
    public static ParameterizedExpression SimplifyWithoutReparameterization(ParameterizedExpression expr) {
      return RuleBasedSimplificationVisitor.SimplifyWithoutReparameterization(expr);
    }

    // Reduces the length of the parameter vector to include only the parameters
    // that are still referenced in the simplified expression. 
    public static Expression<ParametricFunction> SimplifyAndRemoveParameters(Expression<ParametricFunction> expr, double[] thetaValues, out double[] newThetaValues) {
      var theta = expr.Parameters[0];
      var parameterizedExpr = new ParameterizedExpression(expr, theta, thetaValues);
      parameterizedExpr = RuleBasedSimplificationVisitor.Simplify(parameterizedExpr);
      var collectParamVisitor = new CollectParametersVisitor(parameterizedExpr.p, parameterizedExpr.pValues);
      var simplifiedExpr = (Expression<ParametricFunction>)collectParamVisitor.Visit(parameterizedExpr.expr);
      newThetaValues = collectParamVisitor.GetNewParameterValues;
      return simplifiedExpr;
    }


    // this calls FoldParameters repeately until the expression does not change anymore
    public static Expression<ParametricFunction> SimplifyRepeated(Expression<ParametricFunction> parametricExpr, double[] p, out double[] newP) {
      var simplifiedExpr = Simplify(parametricExpr, p, out newP);
      var newSimplifiedStr = simplifiedExpr.ToString();
      var exprSet = new HashSet<string>();
      // simplify until no change (TODO: this shouldn't be necessary if visitors are implemented carefully)
      do {
        exprSet.Add(newSimplifiedStr);
        simplifiedExpr = Simplify(simplifiedExpr, newP, out newP);
        // System.Console.WriteLine(Expr.ToString(simplifiedExpr, varNames, newP));
        newSimplifiedStr = simplifiedExpr.ToString();
      } while (!exprSet.Contains(newSimplifiedStr));
      return simplifiedExpr;
    }

    
    // Notes: 
    // px + px == (p+p) * x == p
    // was ist der Nutzen von Hashing?
    // - auch für lange expressions effizient (keine Strings notwendig)
    // alle parameter haben denselben Hashwert == Parameter sind austauschbar (p1 <-> p2)
    // alle sub-ausdrücke die nur Parameter und/oder Konstanten enthalten zusammenziehen




    public static Expression<ParametricFunction> Simplify(Expression<ParametricFunction> expr,
      double[] parameterValues, out double[] newParameterValues) {
      var theta = expr.Parameters[0];

      // TODO use parameterizedExpression in all visitors

      var parameterizedExpr = new ParameterizedExpression(expr, theta, parameterValues);

      parameterizedExpr = RuleBasedSimplificationVisitor.Simplify(parameterizedExpr);

      parameterizedExpr = LiftLinearParametersVisitor.LiftParameters(parameterizedExpr);
      // expr = LiftLinearParametersVisitor.LiftParameters(parameterizedExpr.expr, parameterizedExpr.p, parameterizedExpr.pValues, out var newPValues);
      // parameterizedExpr = new ParameterizedExpression(expr, parameterizedExpr.p, newPValues);

      parameterizedExpr = ExpandProductsVisitor.Expand(parameterizedExpr);
      // Console.WriteLine($"Folded parameters: {newExpr}");

      parameterizedExpr = FoldConstants(parameterizedExpr);

      expr = parameterizedExpr.expr;
      newParameterValues = parameterizedExpr.pValues;

      var collectVisitor = new CollectParametersVisitor(theta, newParameterValues);
      expr = (Expression<ParametricFunction>)collectVisitor.Visit(expr);
      newParameterValues = collectVisitor.GetNewParameterValues;
      return expr;
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
      // var parameterizedExpression = Expr.ReplaceParameterWithValues<Func<double[], double>>(expr, expr.Parameters[0], p);

      var exprBody = expr.Body.ToString();

      // replace all numbers with number"f" to mark them as "fixed"
      // but ignore array indexes
      var expPart = $"(('e'|'E')[-+]?[0-9][0-9]*)";
      var floatLit = $"(?<num>[0-9][0-9]*\\.[0-9]*{expPart}?)" + // float with digits before and after comma
        $"|(?<num>\\.[0-9][0-9]*{expPart}?)" + // float with digits only after comma
        $"|(?<num>[0-9][0-9]*{expPart})" + // float without digits after comma but with exponent
        $"|[^[](?<num>[0-9][0-9]*)[^].eE0-9]" // integer (but not as array index)
        ;
      var floatLitRegex = new Regex(floatLit, RegexOptions.ExplicitCapture);

      var match = floatLitRegex.Match(exprBody);
      while (match.Groups["num"].Success) {
        var numGroup = match.Groups["num"];
        exprBody = exprBody.Insert(numGroup.Index + numGroup.Length, "f"); // insert "f" after number
        match = floatLitRegex.Match(exprBody, numGroup.Index + numGroup.Length); // find next match
      }

      for (int i = 0; i < varNames.Length; i++) {
        exprBody = exprBody.Replace($"x[{i}]", varNames[i]);
      }


      for (int i = 0; i < p.Length; i++) {
        exprBody = exprBody.Replace($"{expr.Parameters[0].Name}[{i}]", p[i].ToString("g4", CultureInfo.InvariantCulture));
      }

      return exprBody.ToString();

    }

    public static int NumberOfNodes(Expression<ParametricFunction> parametricExpr) {
      return CountNodesVisitor.Count(parametricExpr);
    }

    public static int NumberOfParameters(Expression<ParametricFunction> parametricExpr) {
      return CountParametersVisitor.Count(parametricExpr, parametricExpr.Parameters[0]);
    }

    public static double[] CollectConstants(Expression<ParametricFunction> parametricExpr) {
      return CollectConstantsVisitor.CollectConstants(parametricExpr);
    }

    public static IEnumerable<string> CollectSymbols(Expression<ParametricFunction> parametricExpr) {
      return CollectSymbolsVisitor.CollectSymbols(parametricExpr, parametricExpr.Parameters[0]);
    }

    // returns k if expression has structure f(x) + p_k and p is the vector of parameters
    public static int FindOffsetParameterIndex(Expression<ParametricFunction> expr) {
      return FindOffsetParameterVisitor.FindOffsetParameter(expr.Body, expr.Parameters[0]);
    }


    // returns k if expression has structure f(x) * p_k and p is the vector of parameters
    public static int FindScalingParameterIndex(Expression<ParametricFunction> expr) {
      return FindScalingParameterVisitor.FindScalingParameter(expr.Body, expr.Parameters[0]);
    }

    // reparameterizes the function f(x,\theta) such that fx = f(x0, \theta) when theta[paramIdx] = paramValue
    // for a model with intercept f(x,\theta) = g(x,\theta) + \theta[paramIdx] this returns g(x,\theta) - g(x0,\theta) + theta[paramIdx] with theta[paramIdx] == fx
    public static Expression<ParametricFunction> ReparameterizeExpr(Expression<ParametricFunction> modelExpr, double[] x0, out int paramIdx) {
      var visitor = new ReparameterizeExprVisitor(modelExpr.Parameters[0], modelExpr.Parameters[1], x0);
      var expr = (Expression<ParametricFunction>)visitor.Visit(modelExpr);

      paramIdx = visitor.OutParamIdx;
      return expr;
    }
  }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Xml.Linq;

namespace HEAL.Expressions {

  public class ReverseAutoDiffVisitor : ExpressionVisitor {
    private static readonly MethodInfo abs = typeof(Math).GetMethod("Abs", new[] { typeof(double) });
    private static readonly MethodInfo log = typeof(Math).GetMethod("Log", new[] { typeof(double) });
    private static readonly MethodInfo exp = typeof(Math).GetMethod("Exp", new[] { typeof(double) });
    private static readonly MethodInfo sqrt = typeof(Math).GetMethod("Sqrt", new[] { typeof(double) });
    private static readonly MethodInfo sin = typeof(Math).GetMethod("Sin", new[] { typeof(double) });
    private static readonly MethodInfo cos = typeof(Math).GetMethod("Cos", new[] { typeof(double) });
    private static readonly MethodInfo sinh = typeof(Math).GetMethod("Sinh", new[] { typeof(double) });
    private static readonly MethodInfo cosh = typeof(Math).GetMethod("Cosh", new[] { typeof(double) });
    private static readonly MethodInfo tanh = typeof(Math).GetMethod("Tanh", new[] { typeof(double) });
    private static readonly MethodInfo pow = typeof(Math).GetMethod("Pow", new[] { typeof(double), typeof(double) });
    private static readonly MethodInfo cbrt = typeof(Functions).GetMethod("Cbrt", new[] { typeof(double) });
    private static readonly MethodInfo sign = typeof(Functions).GetMethod("Sign", new[] { typeof(double) });
    private static readonly MethodInfo aq = typeof(Functions).GetMethod("AQ", new[] { typeof(double), typeof(double) });
    private static readonly MethodInfo logistic = typeof(Functions).GetMethod("Logistic", new[] { typeof(double) });
    private static readonly MethodInfo invLogistic = typeof(Functions).GetMethod("InvLogistic", new[] { typeof(double) });
    private static readonly MethodInfo logisticPrime = typeof(Functions).GetMethod("LogisticPrime", new[] { typeof(double) });
    private static readonly MethodInfo logisticPrimePrime = typeof(Functions).GetMethod("LogisticPrimePrime", new[] { typeof(double) });
    private static readonly MethodInfo invLogisticPrime = typeof(Functions).GetMethod("InvLogisticPrime", new[] { typeof(double) });
    private static readonly MethodInfo invLogisticPrimePrime = typeof(Functions).GetMethod("InvLogisticPrimePrime", new[] { typeof(double) });

    private static readonly MethodInfo arrClear = typeof(Array).GetMethod("Clear", new[] { typeof(Array) });
    private readonly int numNodes;
    private readonly ParameterExpression x; // double[] from original expr
    private readonly ParameterExpression theta; // double[] from original expr

    private readonly ParameterExpression X; // double[,] for new expr
    private readonly ParameterExpression f; // double[] for new expr
    private readonly ParameterExpression Jac; // double[,] for new expr

    // variables for the new expr
    private readonly ParameterExpression eval;
    private readonly ParameterExpression diff;
    private readonly ParameterExpression rowIdx;

    private readonly List<Expression> forwardExpressions = new List<Expression>();
    private readonly List<Expression> backwardExpressions = new List<Expression>();
    private int revIdx;

    private readonly Dictionary<Expression, Expression> fwdExprMap = new Dictionary<Expression, Expression>();
    private readonly Dictionary<Expression, int> revExprMap = new Dictionary<Expression, int>();

    private ReverseAutoDiffVisitor(Expression<Expr.ParametricFunction> expr) {
      theta = expr.Parameters[0];
      x = expr.Parameters[1];
      numNodes = Expr.NumberOfNodes(expr);
      revIdx = 0;

      X = Expression.Parameter(typeof(double[,]), "X");
      f = Expression.Parameter(typeof(double[]), "f");
      Jac = Expression.Parameter(typeof(double[,]), "Jac");
      eval = Expression.Parameter(typeof(double[,]), "eval");
      diff = Expression.Parameter(typeof(double[,]), "revEval");
      rowIdx = Expression.Variable(typeof(int), "rowIdx");
    }

    // TODO
    // - batched evaluation
    // - try outer loop over rows instead of inner loops
    // - remove unnecessary reverse expressions (leading to variables), tracing visited tape elements
    // - investigate bad performance
    // - investigate why it does not work with RAR likelihood yet
    // - reverse call for constants not necessary
    // - check assembly of generated code. potentially change code generation to produce IL directly instead of Expressions
    // - check if generation of a static method (in a type) is faster https://stackoverflow.com/questions/5568294/compiled-c-sharp-lambda-expressions-performance#5573075 https://stackoverflow.com/questions/5053032/performance-of-compiled-to-delegate-expression
    // - code should be at least faster than reverse autodiff interpreter (ExpressionInterpreter) because it has to perform the same steps 
    public static Expr.ParametricJacobianFunction GenerateJacobianExpression(Expression<Expr.ParametricFunction> expr, int nRows) {
      var visitor = new ReverseAutoDiffVisitor(expr);
      
      visitor.Visit(expr.Body);

      // we assume f, and Jac for the return values are allocated by the user

      var initBlock = Expression.Block(
        // if (Jac != null) Array.Clear()
        Expression.IfThen(Expression.NotEqual(visitor.Jac, Expression.Constant(null)),
          Expression.Call(arrClear, visitor.Jac)) // clear Jac
        );
      #region forward 

      // loop over all rows
      LabelTarget[] loopEnds = new LabelTarget[visitor.forwardExpressions.Count + 1];
      Expression[] loops = new Expression[visitor.forwardExpressions.Count + 1]; // one additional loop for copying eval results to f[]

      // loops for each assignment expression
      for (int i = 0; i < visitor.forwardExpressions.Count; i++) {
        loopEnds[i] = Expression.Label($"endloop_{i}");
        loops[i] = Expression.Block(
          Expression.Assign(visitor.rowIdx, Expression.Constant(0)),
          Expression.Loop(
          // if (rowIdx == nRows) break;
          Expression.IfThenElse(Expression.Equal(visitor.rowIdx, Expression.Constant(nRows)),
            Expression.Break(loopEnds[i]),
            Expression.Block(

          // eval[...] = ...
          visitor.forwardExpressions[i],

          // rowIdx++
          Expression.PostIncrementAssign(visitor.rowIdx))),
          loopEnds[i]));
      }

      // loop for f[i] = eval[numNodes - 1]
      loopEnds[visitor.forwardExpressions.Count] = Expression.Label($"endloop_last");
      loops[visitor.forwardExpressions.Count] = Expression.Block(
          Expression.Assign(visitor.rowIdx, Expression.Constant(0)),
          Expression.Loop(
          // if (rowIdx == nRows) break;
          Expression.IfThenElse(Expression.Equal(visitor.rowIdx, Expression.ArrayLength(visitor.f)),
            Expression.Break(loopEnds[visitor.forwardExpressions.Count]),
          Expression.Block(
          // f[rowIdx] = eval[rowIdx, numNodes - 1]
          Expression.Assign(
            Expression.ArrayAccess(visitor.f, visitor.rowIdx),
            visitor.fwdExprMap[expr.Body]),

          // rowIdx++
          Expression.PostIncrementAssign(visitor.rowIdx))),
          loopEnds[visitor.forwardExpressions.Count]));



      var forwardLoops = Expression.Block(
        variables: new[] { visitor.rowIdx },
        expressions: loops);

      #endregion

      #region backward 

      Expression backwardLoops = null;
      if (visitor.backwardExpressions.Any()) {
        loopEnds = new LabelTarget[visitor.backwardExpressions.Count + 1];
        loops = new Expression[visitor.backwardExpressions.Count + 1]; // one additional loop for copying eval results to f[]

        // loop for diff[i] = 1.0
        loopEnds[0] = Expression.Label($"reverse_endloop_first");
        loops[0] = Expression.Block(
            Expression.Assign(visitor.rowIdx, Expression.Constant(0)),
            Expression.Loop(
            // if (rowIdx == nRows) break;
            Expression.IfThenElse(Expression.Equal(visitor.rowIdx, Expression.Constant(nRows)),
              Expression.Break(loopEnds[0]),
              Expression.Block(
            // diff[numNodes - 1, row] = 1.0
            Expression.Assign(
              Expression.ArrayAccess(visitor.diff, new Expression[] { Expression.Constant(visitor.numNodes - 1), visitor.rowIdx }),
              Expression.Constant(1.0)),

            // rowIdx++
            Expression.PostIncrementAssign(visitor.rowIdx))),
            loopEnds[0]));

        var loopIdx = 1;
        foreach (var backExpr in visitor.backwardExpressions.Reverse<Expression>()) {
          loopEnds[loopIdx] = Expression.Label($"reverse_endloop_{loopIdx}");
          loops[loopIdx] = Expression.Block(
            Expression.Assign(visitor.rowIdx, Expression.Constant(0)),
            Expression.Loop(
            // if (rowIdx == nRows) break;
            Expression.IfThenElse(Expression.Equal(visitor.rowIdx, Expression.Constant(nRows)),
              Expression.Break(loopEnds[loopIdx]),
              Expression.Block(
            backExpr,

            // rowIdx++
            Expression.PostIncrementAssign(visitor.rowIdx))),
            loopEnds[loopIdx]));
          loopIdx++;
        }
        backwardLoops = Expression.Block(
         variables: new[] { visitor.rowIdx },
         expressions: loops);
      } else {
        backwardLoops = Expression.Empty();
      }


      #endregion

      var newBody = Expression.Block(
        expressions: new Expression[] { initBlock, forwardLoops, backwardLoops }
        );

      return GenerateExpression(nRows, visitor.numNodes, newBody, visitor.eval, visitor.diff, visitor.theta, visitor.X, visitor.f, visitor.Jac);
    }

    private static Expr.ParametricJacobianFunction GenerateExpression(int nRows, int numNodes, BlockExpression newBody,
      ParameterExpression evalParam, ParameterExpression diffParam, ParameterExpression thetaParam, ParameterExpression XParam, ParameterExpression fParam, ParameterExpression JacParam) {
      // create buffers
      var eval = new double[numNodes, nRows];
      var diff = new double[numNodes, nRows];
      var expr = Expression.Lambda<Action<double[,], double[,], double[], double[,], double[], double[,]>>(newBody, evalParam, diffParam, thetaParam, XParam, fParam, JacParam);
      // System.Console.WriteLine(GetDebugView(expr));
      var func = expr.Compile();
      return (double[] theta, double[,] X, double[] f, double[,] Jac) =>
        func(eval, diff, theta, X, f, Jac);
    }


    protected override Expression VisitConstant(ConstantExpression node) {
      fwdExprMap.Add(node, node);

      // Forward(node);
      // Backpropagate(curIdx, Expression.Constant(0.0));
      // curIdx++;
      revExprMap.Add(node, revIdx++);
      return node;
    }

    protected override Expression VisitBinary(BinaryExpression node) {

      if (node.NodeType == ExpressionType.ArrayIndex) {
        // handle parameters
        if (node.Left == theta) {
          fwdExprMap.Add(node, Expression.ArrayIndex(theta, node.Right));
          // Forward(Expression.ArrayIndex(theta, node.Right));
          // collect into Jacobian
          backwardExpressions.Add(Expression.AddAssign(Expression.ArrayAccess(Jac, new[] { rowIdx, node.Right }), BufferAt(diff, revIdx)));
          revExprMap.Add(node, revIdx++);
          return node;
        } else if (node.Left == x) {
          fwdExprMap.Add(node, Expression.ArrayAccess(X, new[] { rowIdx, node.Right }));
          // Forward(Expression.ArrayAccess(X, new[] { rowIdx, node.Right }));
          // no need to back-prop into variables          
          revExprMap.Add(node, revIdx++);
          return node;
        } else throw new InvalidProgramException($"unknown array {node.Left}");
      } else {

        Visit(node.Left);
        Visit(node.Right);

        fwdExprMap.Add(node, BufferAt(eval, forwardExpressions.Count));

        // eval[curIdx] = eval[leftIdx] ° eval[rightIdx]
        Forward(node.Update(fwdExprMap[node.Left], null, fwdExprMap[node.Right]));

        switch (node.NodeType) {
          case ExpressionType.Add: {
              Backpropagate(node.Left, BufferAt(diff, revIdx));
              Backpropagate(node.Right, BufferAt(diff, revIdx));
              break;
            }
          case ExpressionType.Subtract: {
              Backpropagate(node.Left, BufferAt(diff, revIdx));
              Backpropagate(node.Right, Expression.Negate(BufferAt(diff, revIdx)));
              break;
            }
          case ExpressionType.Multiply: {
              // diff[left] = diff[cur] * eval[right]
              Backpropagate(node.Left,
                Expression.Multiply(BufferAt(diff, revIdx), fwdExprMap[node.Right]));
              // diff[right] = diff[cur] * eval[left]
              Backpropagate(node.Right,
                Expression.Multiply(BufferAt(diff, revIdx), fwdExprMap[node.Left]));
              break;
            }
          case ExpressionType.Divide: {
              // diff[left] = diff[cur] / eval[right];
              Backpropagate(node.Left,
                Expression.Divide(BufferAt(diff, revIdx), fwdExprMap[node.Right]));

              // diff[right] = -diff[cur] * eval[left] / (eval[right] * eval[right]);
              Backpropagate(node.Right,
                Expression.Multiply(
                  Expression.Negate(BufferAt(diff, revIdx)),
                  Expression.Divide(
                    fwdExprMap[node.Left],
                    Expression.Multiply(
                      fwdExprMap[node.Right],
                      fwdExprMap[node.Right]))));
              break;
            }
        }


        revExprMap.Add(node, revIdx++);

        return node;
      }
    }


    protected override Expression VisitUnary(UnaryExpression node) {

      Visit(node.Operand);

      fwdExprMap.Add(node, BufferAt(eval, forwardExpressions.Count));

      Forward(node.Update(fwdExprMap[node.Operand]));

      Backpropagate(node.Operand, node.Update(BufferAt(diff, revIdx)));
      revExprMap.Add(node, revIdx++);
      return node;
    }


    protected override Expression VisitMethodCall(MethodCallExpression node) {
      // if (!SupportedMethods.Contains(node.Method)) throw new NotSupportedException($"unsupported method {node.Method}");
      for (int i = 0; i < node.Arguments.Count; i++) {
        Visit(node.Arguments[i]);
      }

      fwdExprMap.Add(node, BufferAt(eval, forwardExpressions.Count));

      Forward(node.Update(node.Object, node.Arguments.Select(arg => fwdExprMap[arg])));

      if (node.Method == log) {
        Backpropagate(node.Arguments[0], Expression.Divide(BufferAt(diff, revIdx), fwdExprMap[node.Arguments[0]]));
      } else if (node.Method == abs) {
        Backpropagate(node.Arguments[0], Expression.Multiply(BufferAt(diff, revIdx), Expression.Call(sign, fwdExprMap[node.Arguments[0]])));
      } else if (node.Method == exp) {
        Backpropagate(node.Arguments[0], Expression.Multiply(BufferAt(diff, revIdx), fwdExprMap[node]));
      } else if (node.Method == sin) {
        Backpropagate(node.Arguments[0], Expression.Multiply(BufferAt(diff, revIdx), Expression.Call(cos, fwdExprMap[node.Arguments[0]])));
      } else if (node.Method == cos) {
        Backpropagate(node.Arguments[0], Expression.Multiply(BufferAt(diff, revIdx), Expression.Negate(Expression.Call(sin, fwdExprMap[node.Arguments[0]]))));
      } else if (node.Method == cosh) {
        Backpropagate(node.Arguments[0], Expression.Multiply(BufferAt(diff, revIdx), Expression.Call(sinh, fwdExprMap[node.Arguments[0]])));
      } else if (node.Method == tanh) {
        // diff * 2 / (cosh(2*eval) + 1)
        Backpropagate(node.Arguments[0],
          Expression.Multiply(
            BufferAt(diff, revIdx),
            Expression.Divide(
              Expression.Constant(2.0),
              Expression.Add(
                 Expression.Call(cosh,
                   Expression.Multiply(
                     Expression.Constant(2.0),
                     fwdExprMap[node.Arguments[0]])),
                 Expression.Constant(1.0)))));
      } else if (node.Method == pow) {
        // diff[left] = diff[cur] * eval[right] * eval[cur] / eval[left]; // diff[cur] * eval[right] * eval[left] ^ ( eval[right] - 1);
        Backpropagate(node.Arguments[0], Expression.Multiply(
          BufferAt(diff, revIdx),
          Expression.Multiply(
            fwdExprMap[node.Arguments[1]],
            Expression.Divide(
              fwdExprMap[node],
              fwdExprMap[node.Arguments[0]]))));

        // diff[right] = diff[cur] * eval[cur] * Math.Log(eval[left]);
        Backpropagate(node.Arguments[1],
          Expression.Multiply(
            BufferAt(diff, revIdx),
            Expression.Multiply(
              fwdExprMap[node],
              Expression.Call(log, fwdExprMap[node.Arguments[0]]))));
      } else if (node.Method == sqrt) {
        Backpropagate(node.Arguments[0], Expression.Multiply(
          BufferAt(diff, revIdx),
          Expression.Divide(
            Expression.Constant(0.5),
            fwdExprMap[node])));
      } else if (node.Method == cbrt) {
        Backpropagate(node.Arguments[0], Expression.Divide(
          BufferAt(diff, revIdx),
          Expression.Multiply(
            Expression.Constant(3.0),
            Expression.Multiply(fwdExprMap[node], fwdExprMap[node]))));
      } else if (node.Method == sign) {
        Backpropagate(node.Arguments[0], Expression.Constant(0.0));
      } else if (node.Method == logistic) {
        Backpropagate(node.Arguments[0], Expression.Multiply(
          BufferAt(diff, revIdx),
          Expression.Call(logisticPrime, fwdExprMap[node.Arguments[0]])));
      } else if (node.Method == invLogistic) {
        Backpropagate(node.Arguments[0], Expression.Multiply(
          BufferAt(diff, revIdx),
          Expression.Call(invLogisticPrime, fwdExprMap[node.Arguments[0]])));
      } else if (node.Method == logisticPrime) {
        Backpropagate(node.Arguments[0], Expression.Multiply(
          BufferAt(diff, revIdx),
          Expression.Call(logisticPrimePrime, fwdExprMap[node.Arguments[0]])));
      } else if (node.Method == invLogisticPrime) {
        Backpropagate(node.Arguments[0], Expression.Multiply(
          BufferAt(diff, revIdx),
          Expression.Call(invLogisticPrimePrime, fwdExprMap[node.Arguments[0]])));
      } else throw new NotSupportedException($"{node.Method}");


      revExprMap.Add(node, revIdx++);

      return node;
    }

    #region helper
    private void Forward(Expression expr) {
      forwardExpressions.Add(AssignBufferAt(eval, forwardExpressions.Count, expr));
    }

    private void Backpropagate(Expression prevExpr, Expression expr) {
      if (prevExpr is ConstantExpression || (prevExpr is BinaryExpression binaryExpression && binaryExpression.Left == x)) return; // do not generate expressions to backpropagate error for constants or variables
      var idx = revExprMap[prevExpr];
      backwardExpressions.Add(AssignBufferAt(diff, idx, expr));
    }
    private Expression AssignBufferAt(ParameterExpression evalBuffer, int idx, Expression expr) {
      return Expression.Assign(BufferAt(evalBuffer, idx), expr);
    }

    private Expression BufferAt(ParameterExpression evalBuffer, int idx) {
      return Expression.ArrayAccess(evalBuffer, new Expression[] { Expression.Constant(idx), rowIdx });
    }
    #endregion

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

    public static string GetDebugView(Expression exp) {
      if (exp == null)
        return null;

      var propertyInfo = typeof(Expression).GetProperty("DebugView", BindingFlags.Instance | BindingFlags.NonPublic);
      return propertyInfo.GetValue(exp) as string;
    }
  }
}

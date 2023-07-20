using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;

namespace HEAL.Expressions {

  public class ReverseAutoDiffVisitor : ExpressionVisitor {
    private static readonly MethodInfo abs = typeof(Math).GetMethod("Abs", new[] { typeof(double) });
    private static readonly MethodInfo log = typeof(Math).GetMethod("Log", new[] { typeof(double) });
    private static readonly MethodInfo exp = typeof(Math).GetMethod("Exp", new[] { typeof(double) });
    private static readonly MethodInfo sqrt = typeof(Math).GetMethod("Sqrt", new[] { typeof(double) });
    private static readonly MethodInfo sin = typeof(Math).GetMethod("Sin", new[] { typeof(double) });
    private static readonly MethodInfo cos = typeof(Math).GetMethod("Cos", new[] { typeof(double) });
    private static readonly MethodInfo cosh = typeof(Math).GetMethod("Cosh", new[] { typeof(double) });
    private static readonly MethodInfo tanh = typeof(Math).GetMethod("Tanh", new[] { typeof(double) });
    private static readonly MethodInfo pow = typeof(Math).GetMethod("Pow", new[] { typeof(double), typeof(double) });
    private static readonly MethodInfo cbrt = typeof(Functions).GetMethod("Cbrt", new[] { typeof(double) });
    private static readonly MethodInfo sign = typeof(Functions).GetMethod("Sign", new[] { typeof(double) });
    private static readonly MethodInfo aq = typeof(Functions).GetMethod("AQ", new[] { typeof(double), typeof(double) });
    private static readonly MethodInfo logistic = typeof(Functions).GetMethod("Logistic", new[] { typeof(double) });
    private static readonly MethodInfo invLogistic = typeof(Functions).GetMethod("InvLogistic", new[] { typeof(double) });
    private static readonly MethodInfo logisticPrime = typeof(Functions).GetMethod("LogisticPrime", new[] { typeof(double) });
    private static readonly MethodInfo invLogisticPrime = typeof(Functions).GetMethod("InvLogisticPrime", new[] { typeof(double) });

    private static readonly MethodInfo arrClear = typeof(Array).GetMethod("Clear", new[] { typeof(Array), typeof(int), typeof(int) });
    private static readonly MethodInfo arrLength = typeof(Array).GetMethod("GetLength", new[] { typeof(int) });
    private readonly int numNodes;
    private int curIdx;
    private readonly ParameterExpression x; // double[] from original expr
    private readonly ParameterExpression theta; // double[] from original expr

    private readonly ParameterExpression newTheta; // double[] for new expr
    private readonly ParameterExpression X; // double[,] for new expr
    private readonly ParameterExpression f; // double[] for new expr
    private readonly ParameterExpression Jac; // double[,] for new expr

    // variables for the new expr
    private readonly ParameterExpression eval;
    private readonly ParameterExpression diff;
    private readonly ParameterExpression rowIdx;

    private readonly List<Expression> forwardExpressions = new List<Expression>();
    private readonly List<Expression> backwardExpressions = new List<Expression>();

    private ReverseAutoDiffVisitor(Expression<Expr.ParametricFunction> expr) {
      theta = expr.Parameters[0];
      x = expr.Parameters[1];
      numNodes = Expr.NumberOfNodes(expr);
      curIdx = 0;

      newTheta = Expression.Parameter(typeof(double[]), "theta"); // TODO need two different theta parameters?
      X = Expression.Parameter(typeof(double[,]), "X");
      f = Expression.Parameter(typeof(double[]), "f");
      Jac = Expression.Parameter(typeof(double[,]), "Jac");
      eval = Expression.Variable(typeof(double[,]), "eval");
      diff = Expression.Variable(typeof(double[,]), "revEval");
      rowIdx = Expression.Variable(typeof(int), "rowIdx");
    }

    // TODO batched evaluation
    public static Expression<Expr.ParametricJacobianFunction> GenerateJacobianExpression(Expression<Expr.ParametricFunction> expr, int nRows) {
      var visitor = new ReverseAutoDiffVisitor(expr);

      visitor.Visit(expr.Body);

      // we assume f, and Jac for the return values are allocated by the user

      var initBlock = Expression.Block(
        // eval = new double[numNodes, numRows]        
        Expression.Assign(visitor.eval,
           Expression.NewArrayBounds(typeof(double),
                                     new Expression[] { Expression.Constant(visitor.numNodes),
                                                        Expression.Constant(nRows) })),

        // diff = new double[numNodes, numRows]
        Expression.Assign(visitor.diff,
           Expression.NewArrayBounds(typeof(double),
                                     new Expression[] { Expression.Constant(visitor.numNodes),
                                                        Expression.Constant(nRows) })),

        // if (Jac != null) Array.Clear(Jac, 0, Jac.GetLength(0))
        Expression.IfThen(Expression.NotEqual(visitor.Jac, Expression.Constant(null)),
          Expression.Call(arrClear, visitor.Jac, Expression.Constant(0), Expression.Call(visitor.Jac, arrLength, Expression.Constant(0)))) // clear Jac
        );
      #region forward 

      // loop over all rows
      LabelTarget[] loopEnds = new LabelTarget[visitor.numNodes + 1];
      Expression[] loops = new Expression[visitor.numNodes + 1]; // one additional loop for copying eval results to f[]

      // loops for each assignment expression
      for (int i = 0; i < visitor.numNodes; i++) {
        loopEnds[i] = Expression.Label($"endloop_{i}");
        loops[i] = Expression.Block(
          Expression.Assign(visitor.rowIdx, Expression.Constant(0)),
          Expression.Loop(Expression.Block(
          // if (rowIdx == nRows) break;
          Expression.IfThen(Expression.Equal(visitor.rowIdx, Expression.Constant(nRows)),
            Expression.Break(loopEnds[i])),

          // eval[...] = ...
          visitor.forwardExpressions[i],

          // rowIdx++
          Expression.PostIncrementAssign(visitor.rowIdx)),
          loopEnds[i]));
      }

      // loop for f[i] = eval[numNodes - 1]
      loopEnds[visitor.forwardExpressions.Count] = Expression.Label($"endloop_last");
      loops[visitor.forwardExpressions.Count] = Expression.Block(
          Expression.Assign(visitor.rowIdx, Expression.Constant(0)),
          Expression.Loop(Expression.Block(
          // if (rowIdx == nRows) break;
          Expression.IfThen(Expression.Equal(visitor.rowIdx, Expression.Constant(nRows)),
            Expression.Break(loopEnds[visitor.forwardExpressions.Count])),

          // f[rowIdx] = eval[rowIdx, numNodes - 1]
          Expression.Assign(
            Expression.ArrayAccess(visitor.f, visitor.rowIdx),
            visitor.BufferAt(visitor.eval, visitor.numNodes - 1)),

          // rowIdx++
          Expression.PostIncrementAssign(visitor.rowIdx)),
          loopEnds[visitor.forwardExpressions.Count]));



      var forwardLoops = Expression.Block(
        variables: new[] { visitor.rowIdx },
        expressions: loops);

      #endregion

      #region backward 
      // loop for diff[i] = 1.0
      loopEnds[0] = Expression.Label($"reverse_endloop_first");
      loops[0] = Expression.Block(
          Expression.Assign(visitor.rowIdx, Expression.Constant(0)),
          Expression.Loop(Expression.Block(
          // if (rowIdx == nRows) break;
          Expression.IfThen(Expression.Equal(visitor.rowIdx, Expression.Constant(nRows)),
            Expression.Break(loopEnds[0])),

          // diff[numNodes - 1, row] = 1.0
          Expression.Assign(
            Expression.ArrayAccess(visitor.diff, new Expression[] { Expression.Constant(visitor.numNodes - 1), visitor.rowIdx }),
            Expression.Constant(1.0)),

          // rowIdx++
          Expression.PostIncrementAssign(visitor.rowIdx)),
          loopEnds[0]));

      var loopIdx = 1;
      foreach(var backExpr in visitor.backwardExpressions.Reverse<Expression>()) {
        loopEnds[loopIdx] = Expression.Label($"reverse_endloop_{loopIdx}");
        loops[loopIdx] = Expression.Block(
          Expression.Assign(visitor.rowIdx, Expression.Constant(0)),
          Expression.Loop(Expression.Block(
          // if (rowIdx == nRows) break;
          Expression.IfThen(Expression.Equal(visitor.rowIdx, Expression.Constant(nRows)),
            Expression.Break(loopEnds[loopIdx])),

          backExpr,

          // rowIdx++
          Expression.PostIncrementAssign(visitor.rowIdx)),
          loopEnds[loopIdx]));
        loopIdx++;
      }

      var backwardLoops = Expression.Block(
       variables: new[] { visitor.rowIdx },
       expressions: loops);

      #endregion

      var newBody = Expression.Block(
        variables: new[] { visitor.eval, visitor.diff },
        expressions: new Expression[] { initBlock, forwardLoops, backwardLoops }
        );

      return Expression.Lambda<Expr.ParametricJacobianFunction>(newBody, visitor.theta, visitor.X, visitor.f, visitor.Jac);
    }


    // private static IEnumerable<Expression> Flatten(Expression expr) {
    //   if (expr is BlockExpression blockExpr) {
    //     return blockExpr.Expressions.SelectMany(Flatten);
    //   } else return new[] { expr };
    // }

    protected override Expression VisitConstant(ConstantExpression node) {
      Forward(node);
      Backpropagate(curIdx, Expression.Constant(0.0));
      curIdx++;
      return node;
    }

    protected override Expression VisitBinary(BinaryExpression node) {
      if (node.NodeType == ExpressionType.ArrayIndex) {
        // handle parameters
        if (node.Left == theta) {
          Forward(Expression.ArrayAccess(theta, node.Right));
          // collect into Jacobian
          backwardExpressions.Add(Expression.AddAssign(Expression.ArrayAccess(Jac, new[] { rowIdx, node.Right }), BufferAt(diff, curIdx)));
          curIdx++;
          return node;
        } else if (node.Left == x) {
          Forward(Expression.ArrayAccess(X, new[] { rowIdx, node.Right }));
          // no need to back-prop into variables
          curIdx++;
          return node;
        } else throw new InvalidProgramException($"unknown array {node.Left}");
      } else {

        Visit(node.Left); var leftIdx = curIdx - 1;
        Visit(node.Right); var rightIdx = curIdx - 1;

        // eval[curIdx] = eval[leftIdx] ° eval[rightIdx]
        Forward(node.Update(BufferAt(eval, leftIdx), null, BufferAt(eval, rightIdx)));

        switch (node.NodeType) {
          case ExpressionType.Add: {
              Backpropagate(leftIdx, BufferAt(diff, curIdx));
              Backpropagate(rightIdx, BufferAt(diff, curIdx));
              break;
            }
          case ExpressionType.Subtract: {
              Backpropagate(leftIdx, BufferAt(diff, curIdx));
              Backpropagate(rightIdx, Expression.Negate(BufferAt(diff, curIdx)));
              break;
            }
          case ExpressionType.Multiply: {
              // diff[left] = diff[cur] * eval[right]
              Backpropagate(leftIdx,
                Expression.Multiply(BufferAt(diff, curIdx), BufferAt(eval, rightIdx)));
              // diff[right] = diff[cur] * eval[left]
              Backpropagate(rightIdx,
                Expression.Multiply(BufferAt(diff, curIdx), BufferAt(eval, leftIdx)));
              break;
            }
          case ExpressionType.Divide: {
              // diff[left] = diff[cur] / eval[right];
              Backpropagate(leftIdx,
                Expression.Divide(BufferAt(diff, curIdx), BufferAt(eval, rightIdx)));

              // diff[right] = -diff[cur] * eval[left] / (eval[right] * eval[right]);
              Backpropagate(rightIdx,
                Expression.Multiply(
                  Expression.Negate(BufferAt(diff, curIdx)),
                  Expression.Divide(
                    BufferAt(eval, leftIdx),
                    Expression.Multiply(
                      BufferAt(eval, rightIdx),
                      BufferAt(eval, rightIdx)))));
              break;
            }
        }

        curIdx++;
        return node;
      }
    }


    protected override Expression VisitUnary(UnaryExpression node) {
      Visit(node.Operand); var opdIdx = curIdx - 1;
      Forward(node.Update(BufferAt(eval, opdIdx)));
      Backpropagate(opdIdx, node.Update(BufferAt(diff, curIdx)));
      curIdx++;
      return node;
    }


    protected override Expression VisitMethodCall(MethodCallExpression node) {
      // if (!SupportedMethods.Contains(node.Method)) throw new NotSupportedException($"unsupported method {node.Method}");
      var argIdx = new int[node.Arguments.Count];
      for (int i = 0; i < node.Arguments.Count; i++) {
        Visit(node.Arguments[i]);
        argIdx[i] = curIdx - 1;
      }

      Forward(node.Update(node.Object, argIdx.Select(i => BufferAt(eval, i))));

      if (node.Method == log) {
      } else if (node.Method == abs) {
      } else if (node.Method == exp) {
        Backpropagate(argIdx[0], Expression.Multiply(BufferAt(eval, curIdx), BufferAt(diff, curIdx)));
      } else if (node.Method == sin) {
      } else if (node.Method == cos) {
      } else if (node.Method == cosh) {
      } else if (node.Method == tanh) {
      } else if (node.Method == pow) {
      } else if (node.Method == sqrt) {
      } else if (node.Method == cbrt) {
      } else if (node.Method == sign) {
      } else if (node.Method == logistic) {
      } else if (node.Method == invLogistic) {
      } else if (node.Method == logisticPrime) {
      } else if (node.Method == invLogisticPrime) {

      } else throw new NotSupportedException($"{node.Method}");

      curIdx++;
      return node;
    }

    #region helper
    private void Forward(Expression expr) {
      forwardExpressions.Add(AssignBufferAt(eval, curIdx, expr));
    }

    private void Backpropagate(int idx, Expression expr) {
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
  }
}

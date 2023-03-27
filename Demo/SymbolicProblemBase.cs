using System;
using System.Linq.Expressions;
using System.Threading;
using HEAL.Expressions;

namespace HEAL.NonlinearRegression.Demo {

  /// <summary>
  /// Abstract base class for NLS problems where the model is specified as an Expression.
  /// </summary>
  public abstract class SymbolicProblemBase : INLSProblem {
    public abstract double[,] X { get; }
    public abstract double[] y { get; }

    public abstract double[] ThetaStart { get; }

    // compile the expressions only once
    private object compiledModelLocker = new();
    private Expr.ParametricVectorFunction compiledModel;
    private Expr.ParametricJacobianFunction compiledJacobian;
    
    public Expr.ParametricVectorFunction Model {
      get {
        lock (compiledModelLocker) {
          if (compiledModel == null) {
            compiledModel = Expr.Broadcast(ModelExpression).Compile();
          }
        }
        return compiledModel;
      }
    }
    public Expr.ParametricJacobianFunction ModelJacobian {
      get {
        lock (compiledModelLocker) {
          if (compiledJacobian == null) {
            compiledJacobian = Expr.Broadcast(Expr.Gradient(ModelExpression, ThetaStart.Length)).Compile();
          }
        }
        return compiledJacobian;
      }
    }

    public abstract Expression<Expr.ParametricFunction> ModelExpression { get; }
  }
}
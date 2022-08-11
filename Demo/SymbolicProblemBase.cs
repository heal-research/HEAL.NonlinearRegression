using System;
using System.Linq.Expressions;
using System.Threading;
using HEAL.Expressions;

namespace HEAL.NonlinearRegression {

  /// <summary>
  /// Abstract base class for NLS problems where the model is specified as an Expression.
  /// </summary>
  public abstract class SymbolicProblemBase : INLSProblem {
    public abstract double[,] X { get; }
    public abstract double[] y { get; }

    public abstract double[] ThetaStart { get; }

    public abstract Expression<Expr.ParametricFunction> ModelExpr { get; }
    
    // compile the expressions only once
    private object compiledModelLocker = new object();
    private Expr.ParametricVectorFunction compiledModel;
    private Expr.ParametricJacobianFunction compiledJacobian;
    
    public Expr.ParametricVectorFunction Model {
      get {
        lock (compiledModelLocker) {
          if (compiledModel == null) {
            compiledModel = Expr.Broadcast(ModelExpr).Compile();
          }
        }
        return compiledModel;
      }
    }
    public Expr.ParametricJacobianFunction ModelJacobian {
      get {
        lock (compiledModelLocker) {
          if (compiledJacobian == null) {
            compiledJacobian = Expr.Broadcast(Expr.Gradient(ModelExpr, ThetaStart.Length)).Compile();
          }
        }
        return compiledJacobian;
      }
    }

    public void Func(double[] theta, double[,] X, double[] f) {
      Model(theta, X, f);
    }

    public void Jacobian(double[] theta, double[,] X, double[] f, double[,] jac) {
      ModelJacobian(theta, X, f, jac);
    }

  }
}
using HEAL.Expressions;
using System.Linq.Expressions;

namespace HEAL.NonlinearRegression.Demo {
  /// <summary>
  /// Interface for nonlinear least squares problems
  /// </summary>
  public interface INLSProblem {
    /// <summary>
    /// Input values. size(X) = m * d
    /// </summary>
    double[,] X { get; }
    /// <summary>
    /// Target values. len(y) = m
    /// </summary>
    double[] y { get; }

    Expression<Expr.ParametricFunction> ModelExpression { get; }

    /// <summary>
    /// Model function 
    /// </summary>
    /// <param name="theta">Parameter vector. len(theta) = k </param>
    /// <param name="X">Input matrix. size(X) = m * d</param>
    /// <param name="f">Evaluation result for X. len(f) = m. Must be allocated by the caller.</param>
    void Func(double[] theta, double[,] X, double[] f);

    /// <summary>
    /// Model function with Jacobian
    /// </summary>
    /// <param name="theta">Parameter vector. len(theta) = k </param>
    /// <param name="X">Input matrix. size(X) = m * d</param>
    /// <param name="f">Evaluation result for X. len(f) = m. Must be allocated by the caller.</param>
    /// <param name="jac">Evaluation result for Jacobian J(f(X)) for X. size(jac) = m * k. Must be allocated by the caller.</param>
    void Jacobian(double[] theta, double[,] X, double[] f, double[,] jac);

    /// <summary>
    /// Initial values for theta. len(thetaStart) = k
    /// </summary>
    double[] ThetaStart { get; }

  }
}
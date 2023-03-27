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
    /// Initial values for theta. len(thetaStart) = k
    /// </summary>
    double[] ThetaStart { get; }

  }
}
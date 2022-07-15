using System;
using System.Collections.Generic;
using System.Text;


namespace HEAL.NonlinearRegression {

  /// <summary>
  /// Delegate type for parametric functions with inputs
  /// </summary>
  /// <param name="p">Vector of parameter values</param>
  /// <param name="X">Matrix of input values</param>
  /// <param name="f">Function evaluation result f(p,X). f must be allocated by the caller.</param>
  public delegate void Function(double[] p, double[,] X, double[] f);

  /// <summary>
  /// Delegate for the Jacobian of parametric functions with inputs
  /// </summary>
  /// <param name="p">Vector of parameter values.</param>
  /// <param name="X">Matrix of input values</param>
  /// <param name="f">Evaluation result f(p,X). f must be allocated by the caller.</param>
  /// <param name="jac">Evaluation result J(p,X). jac must be allocated by the caller.</param>
  public delegate void Jacobian(double[] p, double[,] X, double[] f, double[,] jac);

  /// <summary>
  /// Delegate type for parametric residual functions
  /// </summary>
  /// <param name="p">Vector of parameter values</param>
  /// <param name="f">Function evaluation result f(p). f must be allocated by the caller.</param>
  public delegate void ResidualFunction(double[] p, double[] f);

  /// <summary>
  /// Delegate for the Jacobian of parametric residual functions
  /// </summary>
  /// <param name="p">Vector of parameter values.</param>
  /// <param name="f">Evaluation result f(p). f must be allocated by the caller.</param>
  /// <param name="jac">Evaluation result J(p). jac must be allocated by the caller.</param>
  public delegate void ResidualJacobian(double[] p, double[] f, double[,] jac);
}

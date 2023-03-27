using HEAL.Expressions;
using System;
using System.Linq.Expressions;

namespace HEAL.NonlinearRegression.Demo {
  // BOD example from Nonlinear Regression Analysis and Its Applications, Bates and Watts, 1988

  internal class BODProblem : INLSProblem {
    // A 1.4     (there are two BOD datasets)
    private double[] days = new double[] { 1, 2, 3, 4, 5, 7 };
    private double[] BOD = new double[] {
                             8.3,
                             10.3,
                             19.0,
                             16.0,
                             15.6,
                             19.8
                           };

    public double[,] X => Util.ToMatrix(days);

    public double[] y => BOD;

    public double[] ThetaStart => new double[] { 20, 0.24 };

    public Expression<Expr.ParametricFunction> ModelExpression => (double[] theta, double[] x) => theta[0] * (1 - Math.Exp(-theta[1] * x[0]));

  }
}
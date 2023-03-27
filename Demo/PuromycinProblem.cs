using HEAL.Expressions;
using System;
using System.Linq.Expressions;

namespace HEAL.NonlinearRegression.Demo {

  // Puromycin example from Nonlinear Regression Analysis and Its Applications, Bates and Watts, 1988
  internal class PuromycinProblem : INLSProblem {
    // substrate concentration
    private double[] conc = new double[] {
                              0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56, 1.10, 1.10
                            };
    private double[] velocityTreated = new double[] {
                                         76, 47, 97, 107, 123, 139, 159, 152, 191, 201, 207, 200
                                       };
    public double[,] X => Util.ToMatrix(conc);

    public double[] y => velocityTreated;

    public double[] ThetaStart => new double[] { 205, 0.08 };  // Bates and Watts page 41

    public Expression<Expr.ParametricFunction> ModelExpression => (p, x) => p[0] * x[0] / (p[1] + x[0]);
  }
}
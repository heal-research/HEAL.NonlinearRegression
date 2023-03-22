using System;
using System.Linq.Expressions;
using HEAL.Expressions;

namespace HEAL.NonlinearRegression.Demo {

  // Puromycin example from Nonlinear Regression Analysis and Its Applications, Bates and Watts, 1988
  internal class PuromycinDSRProblem : SymbolicProblemBase {
    // substrate concentration
    private double[] conc = new double[] {
                              0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56, 1.10, 1.10
                            };
    private double[] velocityTreated = new double[] {
                                         76, 47, 97, 107, 123, 139, 159, 152, 191, 201, 207, 200
                                       };
    public override double[,] X => Util.ToMatrix(conc);

    public override double[] y => velocityTreated;

    private static double[] thetaStart => new double[] { -1.52730232e+002,  -6.38322549e+000, 2.00942899e+002};

    public override double[] ThetaStart => thetaStart;

    // From grammar enumeration
    //  exp(conc * -6.38322549e+000) * -1.52730232e+002 + 2.00942899e+002
    public override Expression<Expr.ParametricFunction> ModelExpression => (p, x) => 
      p[0] * Math.Exp(x[0] * p[1])  + p[2];
  }
}
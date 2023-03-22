using System;
using System.Linq;
using System.Linq.Expressions;
using System.Xml;
using HEAL.Expressions;

namespace HEAL.NonlinearRegression.Demo {
  internal class RatPol2DProblem : SymbolicProblemBase {

    public RatPol2DProblem() {
      int m = 50;
      var d = 2;
      X = new double[m, d];
      y = new double[m];
      var rand = new System.Random(1234);

      // generate data
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < d; j++) {
          X[i, j] = rand.NextDouble() * 6 +0.05; // u~(0.05, 6.05)
        }

        var x1 = X[i, 0];
        var x2 = X[i, 1];
        y[i] = (Math.Pow(x1-3, 4) + Math.Pow(x2-3, 3) -(x2-3)) / (Math.Pow(x2-2, 4) + 10);
      }
    }

    public override double[,] X { get; }

    public override double[] y { get; }

    private static double[] thetaStart => new double[] {
      9.71243692e-002, 6.39720658e-001, 7.89778563e-001,  
        4.57239812e+000, -2.08852212e-001, -1.26208390e+000, 8.07817393e-001,
        -5.52213198e-001, 2.69747722e-002,3.28722667e-001, -2.16429221e+000
    };

    // optimized
    // p_opt: 1.96002e-006 2.56579e+000 4.64090e-001 5.37162e+000 -1.54712e-001 -1.00020e+000 6.07912e-001 -3.92240e-003 1.73439e-001 3.17599e-001 -2.75960e+000
    
    public override double[] ThetaStart => thetaStart;

    // From grammar enumeration
    // exp(X1 * 6.39720658e-001) * 9.71243692e-002 +
    // X1 * 7.89778563e-001 +
    // exp(X2 * X2 * -2.08852212e-001) * exp(X1 * -1.26208390e+000) * exp(X2 * 8.07817393e-001) * 4.57239812e+000 +
    // X1 * exp(X1 * X2 * 2.69747722e-002) * -5.52213198e-001 +
    // X2 * 3.28722667e-001 +
    // -2.16429221e+000
    public override Expression<Expr.ParametricFunction> ModelExpression => (p, x) =>
      p[0] * Math.Exp(x[0] * p[1]) +
      p[2] * x[0] +
      p[3] * Math.Exp(p[4] * x[1] * x[1]) * Math.Exp(p[5] * x[0]) * Math.Exp(p[6] * x[1]) +
      p[7] * x[0] * Math.Exp(p[8] * x[0] * x[1]) +
      p[9] * x[1] +
      p[10];

  }
}
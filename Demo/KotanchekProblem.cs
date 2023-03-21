using System;
using System.Linq;
using System.Linq.Expressions;
using System.Xml;
using HEAL.Expressions;

namespace HEAL.NonlinearRegression.Demo {
  internal class KotanchekProblem : SymbolicProblemBase {

    public KotanchekProblem() {
      int m = 100;
      var d = 2;
      X = new double[m, d];
      y = new double[m];
      var rand = new System.Random(1234);

      // generate data
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < d; j++) {
          X[i, j] = rand.NextDouble() * 3.7 +0.3; // u~(0.3, 4)
        }

        y[i] = Math.Exp(-Math.Pow(X[i, 0] - 1, 2)) / (1.2 + Math.Pow(X[i, 1] - 2.5, 2));
      }
      // no noise
    }

    public override double[,] X { get; }

    public override double[] y { get; }

    private static double[] thetaStart => new double[] { 4.5274e-2, -3.14082e-1, 2.4327, -3.73031e-1, 1.2777, 
      -1.3847, -9.22e-1, -1.7189e-1 };

    public override double[] ThetaStart => thetaStart;

    // From grammar enumeration
    // X1 * 4.52740811e-002 +
    // log(X2 * -3.14082182e-001 + 2.43269857e+000) * X2 * exp(X1 * X1 * -3.73031009e-001) * 1.27768655e+000 +
    // exp(X1 * X1 * -1.38468721e+000) * -9.22012634e-001 +
    // -1.71892558e-001
    public override Expression<Expr.ParametricFunction> ModelExpr => (p, x) => 
      x[0]*x[0] * p[0] +
        Math.Log(x[1] * p[1] + p[2]) * x[1] * Math.Exp(x[0] * x[0] * p[3]) * p[4] +
        Math.Exp(x[0] * x[0] * p[5]) * p[6] +
        p[7];
  }
}
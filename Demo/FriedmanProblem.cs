using System;
using System.Linq.Expressions;
using HEAL.Expressions;

namespace HEAL.NonlinearRegression.Demo {
  internal class FriedmanProblem : SymbolicProblemBase {

    public FriedmanProblem() {
      int m = 100;
      var d = 10;
      X = new double[m, d];
      y = new double[m];

      // generate data
      var rand = new Random(1234);
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < d; j++) {
          X[i, j] = rand.NextDouble();
        }

        y[i] = 10 * Math.Sin(Math.PI * X[i, 0] * X[i, 1]) + 20 * Math.Pow(X[i, 2] - 0.5, 2) + 10 * X[i, 3] +
               5 * X[i, 4];
        y[i] += 1.0 / 4.8 * Util.RandNorm(rand, 0, 1);
      }

    }

    public override double[,] X { get; }

    public override double[] y { get; }

    public static double[] thetaStart = new double[] { 10.0, 20.0, -0.5, 10, 5 };
    public override double[] ThetaStart => thetaStart;


    // generating expression
    public override Expression<Expr.ParametricFunction> ModelExpression => (p, x) => 
        p[0] * Math.Sin(Math.PI * x[0] * x[1]) + p[1] * Math.Pow(x[2] - p[2], 2) + p[3] * x[3] + p[4] * x[4];
    
  }
}
using HEAL.Expressions;
using System;
using System.Linq.Expressions;

namespace HEAL.NonlinearRegression.Demo {
  internal class LinearProblem : INLSProblem {

    private static double[] pOpt = new double[] { 1, 2, 3, 4 };
    public LinearProblem() {
      int m = 20;
      var d = 4;
      X = new double[m, d];
      y = new double[m];
      var rand = new System.Random(1234);

      // generate data
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < d - 1; j++) {
          X[i, j] = rand.NextDouble() * 2 - 1; // u~(-1,1)
        }
        X[i, d - 1] = 1.0;
      }

      var func = Expr.Broadcast(ModelExpression).Compile();
      func(pOpt, X, y); // calculate target

      // and add noise
      for (int i = 0; i < m; i++) y[i] += rand.NextDouble() * 0.2 - 0.1;
    }

    public double[,] X { get; private set; }

    public double[] y { get; private set; }

    public double[] ThetaStart => new double[] { .1, .1, .1, .1 };

    public Expression<Expr.ParametricFunction> ModelExpression => (p, x) => p[0] * x[0] + p[1] * x[1] + p[2] * x[2] + p[3] * x[3];
  }
}
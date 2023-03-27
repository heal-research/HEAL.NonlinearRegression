using HEAL.Expressions;
using System;
using System.Linq.Expressions;

namespace HEAL.NonlinearRegression.Demo {
  internal class LinearUnivariateProblem : INLSProblem {

    private static double[] pOpt = new double[] { 2.5, 2 };
    public LinearUnivariateProblem() {
      int m = 20;
      var d = 2;
      X = new double[m, d];
      y = new double[m];

      // generate data
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < d - 1; j++) {
          X[i, j] = i / (double)m * 20 - 10; //  rand.NextDouble() * 20 - 10;
        }
        X[i, d - 1] = 1.0;
      }

      var func = Expr.Broadcast(ModelExpression).Compile();
      func(pOpt, X, y); // calculate target

      // and add noise
      var rand = new System.Random(1234);
      for (int i = 0; i < m; i++) y[i] += Util.RandNorm(rand, 0, stdDev: 10);
    }

    public double[,] X { get; private set; }

    public double[] y { get; private set; }

    public double[] ThetaStart => new double[] { .1, .1 };

    public Expression<Expr.ParametricFunction> ModelExpression => (p, x) => p[0] * x[0] + p[1] * x[1];
  }
}
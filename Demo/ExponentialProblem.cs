using HEAL.Expressions;
using System;
using System.Linq.Expressions;

namespace HEAL.NonlinearRegression.Demo {
  internal class ExponentialProblem : INLSProblem {

    private static double[] pOpt = new double[] { 0.2, -3.0 };
    public ExponentialProblem() {

      int m = 20;
      X = new double[m, 1];
      y = new double[m];
      var rand = new System.Random(1234);

      // generate data
      for (int i = 0; i < m; i++) {
        X[i, 0] = i / (double)m;
      }

      var func = Expr.Broadcast(ModelExpression).Compile();
      func(pOpt, X, y);
    }

    public double[,] X { get; private set; }

    public double[] y { get; private set; }

    public double[] ThetaStart => new double[] { 1.0, 1.0 };

    public Expression<Expr.ParametricFunction> ModelExpression => (double[] theta, double[] x) => theta[0] * Math.Exp(x[0] * theta[1]);

  }
}
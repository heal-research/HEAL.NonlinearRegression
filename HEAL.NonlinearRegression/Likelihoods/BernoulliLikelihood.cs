using HEAL.Expressions;
using System;
using System.Linq.Expressions;

namespace HEAL.NonlinearRegression.Likelihoods {

  internal class BernoulliLikelihood : LikelihoodBase {

    internal BernoulliLikelihood(BernoulliLikelihood original) : base(original) { }
    public BernoulliLikelihood(double[,] x, double[] y, Expression<Expr.ParametricFunction> modelExpr, int numModelParam) : base(modelExpr, x, y, numModelParam) { }

    public override double[,] FisherInformation(double[] p) {
      var m = y.Length;
      var n = p.Length;
      var yPred = new double[m];
      var yJac = new double[m, n];
      ModelJacobian(p, x, yPred, yJac);

      var hess = new double[n, n];
      var xi = new double[n];
      var modelHess = new double[n, n];
      for (int i = 0; i < m; i++) {
        var s = 1 / ((1 - yPred[i]) * (1 - yPred[i]) * yPred[i] * yPred[i]);
        Util.CopyRow(x, i, xi);
        ModelHessian(p, xi, modelHess);
        for (int j = 0; j < n; j++) {
          for (int k = 0; k < n; k++) {
            var hessianTerm = (yPred[i] - 1) * yPred[i] * modelHess[j, k] * (y[i] - yPred[i]);
            var gradientTerm = (-2 * y[i] * yPred[i] + yPred[i] * yPred[i] + y[i]) * yJac[i, j] * yJac[i, k];
            hess[j, k] += s * (hessianTerm + gradientTerm);
          }
        }
      }
      return hess;
    }

    public override double NegLogLikelihood(double[] p) {
      NegLogLikelihoodGradient(p, out var nll, nll_grad: null);
      return nll;
    }

    public override void NegLogLikelihoodGradient(double[] p, out double nll, double[]? nll_grad) {
      var m = y.Length;
      var n = p.Length;
      var yPred = new double[m];
      var yJac = new double[m, n];

      nll = 0.0;
      if (nll_grad == null) {
        ModelFunc(p, x, yPred);
      } else {
        ModelJacobian(p, x, yPred, yJac);
        Array.Clear(nll_grad, 0, n);
      }
      for (int i = 0; i < m; i++) {
        if (y[i] != 0.0 && y[i] != 1.0) throw new ArgumentException("target variable must be binary (0/1) for Bernoulli likelihood");
        if (y[i] == 1) {
          nll -= Math.Log(yPred[i]);
        } else {
          // y[i]==0
          nll -= Math.Log(1 - yPred[i]);
        }
        if (nll_grad != null) {
          for (int j = 0; j < n; j++) {
            nll_grad[j] -= (y[i] - yPred[i]) * yJac[i, j] / ((1 - yPred[i]) * yPred[i]);
          }
        }
      }
    }

    public override LikelihoodBase Clone() {
      return new BernoulliLikelihood(this);
    }
  }
}

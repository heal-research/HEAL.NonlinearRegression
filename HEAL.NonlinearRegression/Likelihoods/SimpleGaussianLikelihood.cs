using HEAL.Expressions;
using System;
using System.Linq.Expressions;

namespace HEAL.NonlinearRegression.Likelihoods {

  // errors are iid N(0, noise_sigma)
  internal class SimpleGaussianLikelihood : LikelihoodBase {
    private readonly double sErr;

    internal SimpleGaussianLikelihood(SimpleGaussianLikelihood original) : base(original) {
      this.sErr = original.sErr;
    }
    public SimpleGaussianLikelihood(double[,] x, double[] y, Expression<Expr.ParametricFunction> modelExpr, double noiseSigma = 1.0)
      : base(modelExpr, x, y, numLikelihoodParams: 1) {
      this.sErr = noiseSigma;
    }

    public override double[,] FisherInformation(double[] p) {
      var m = y.Length;
      var n = p.Length;
      var d = x.GetLength(0);
      var yPred = new double[m];
      var yJac = new double[m, n];
      ModelJacobian(p, x, yPred, yJac);

      var hess = new double[n, n];
      var modelHess = new double[n, n];
      var xi = new double[d];
      for (int i = 0; i < m; i++) {
        var res = y[i] - yPred[i];
        // evalute Hessian for current row
        Util.CopyRow(x, i, xi);
        ModelHessian(p, xi, modelHess);
        for (int j = 0; j < n; j++) {
          for (int k = 0; k < n; k++) {
            hess[j, k] += (yJac[i, j] * yJac[i, k] - res * modelHess[j, k]) / (sErr * sErr);
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
        var res = y[i] - yPred[i];
        nll += 0.5 * res * res / (sErr * sErr);

        if (nll_grad != null) {
          for (int j = 0; j < n; j++) {
            nll_grad[j] += -res * yJac[i, j] / (sErr * sErr);
          }
        }
      }
    }

    public override LikelihoodBase Clone() {
      return new SimpleGaussianLikelihood(this);
    }
  }
}

using HEAL.Expressions;
using System;
using System.Linq;
using System.Linq.Expressions;
namespace HEAL.NonlinearRegression {

  // errors are e_i ~ N(0, sigma_i)
  // each measurement has a different sigma
  public class GaussianLikelihood : LikelihoodBase {
    private readonly double[] sigma2; // σ² 

    internal GaussianLikelihood(GaussianLikelihood original) : base(original) {
      this.sigma2 = original.sigma2;
    }

    // TODO: change parameter to σ² 
    public GaussianLikelihood(double[,] x, double[] y, Expression<Expr.ParametricFunction> modelExpr, double[] invNoiseSigma)
      : base(modelExpr, x, y, numLikelihoodParams: 0) {
      this.sigma2 = invNoiseSigma.Select(invS => 1.0 / (invS * invS)).ToArray();
    }

    public override double[,] FisherInformation(double[] p) {
      var m = y.Length;
      var n = p.Length;
      var yJac = new double[m, n];
      var yHess = new double[n, m, n]; // parameters x rows x parameters
      var yHessJ = new double[m, n]; // buffer

      var yPred = interpreter.EvaluateWithJac(p, null, yJac);

      // evaluate hessian
      for (int j = 0; j < p.Length; j++) {
        gradInterpreter[j].EvaluateWithJac(p, null, yHessJ);
        Buffer.BlockCopy(yHessJ, 0, yHess, j * m * n * sizeof(double), m * n * sizeof(double));
        Array.Clear(yHessJ, 0, yHessJ.Length);
      }


      // FIM is the negative of the second derivative (Hessian) of the log-likelihood
      // -> FIM is the Hessian of the negative log-likelihood
      var hessian = new double[n, n];
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          var res = yPred[i] - y[i];
          for (int k = 0; k < n; k++) {
            hessian[j, k] += (yJac[i, j] * yJac[i, k] + res * yHess[j, i, k]) / sigma2[i];
          }
        }
      }

      return hessian;
    }

    // for the calculation of deviance
    public override double BestNegLogLikelihood(double[] p) {
      int m = y.Length;
      return Enumerable.Range(0, m).Sum(i => 0.5 * Math.Log(2.0 * Math.PI * sigma2[i])); // residuals are zero
    }

    public override double NegLogLikelihood(double[] p) {
      NegLogLikelihoodGradient(p, out var nll, nll_grad: null);
      return nll;
    }

    public override void NegLogLikelihoodGradient(double[] p, out double nll, double[] nll_grad) {
      var m = y.Length;
      var n = p.Length;
      double[,] yJac = null;

      nll = BestNegLogLikelihood(p);

      double[] yPred;
      if (nll_grad == null) {
        yPred = interpreter.Evaluate(p);
      } else {
        yJac = new double[m, n];
        yPred = interpreter.EvaluateWithJac(p, null, yJac);
        Array.Clear(nll_grad, 0, n);
      }

      for (int i = 0; i < m; i++) {
        var res = yPred[i] - y[i];
        nll += 0.5 * res * res / sigma2[i];

        if (nll_grad != null) {
          for (int j = 0; j < n; j++) {
            nll_grad[j] += res * yJac[i, j] / sigma2[i];
          }
        }
      }
      if (double.IsNaN(nll)) nll = 1e300;
    }

    public override LikelihoodBase Clone() {
      return new GaussianLikelihood(this);
    }
  }
}

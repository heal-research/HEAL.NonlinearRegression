using HEAL.Expressions;
using System;
using System.Linq.Expressions;
namespace HEAL.NonlinearRegression {

  // errors are iid N(0, noise_sigma)
  public class SimpleGaussianLikelihood : LikelihoodBase {
    private double sErr;
    public double SigmaError { get { return sErr; } set { sErr = value; } }
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
      var yJac = new double[m, n];
      var yHess = new double[n, m, n]; // parameters x rows x parameters
      var yHessJ = new double[m, n]; // buffer

      // var yPred = Expr.EvaluateFuncJac(ModelExpr, p, x, ref yJac);
      var yPred = interpreter.EvaluateWithJac(p, null, yJac);

      // evaluate hessian
      for (int j = 0; j < p.Length; j++) {
        gradInterpreter[j].EvaluateWithJac(p, null, yHessJ);
        // Expr.EvaluateFuncJac(ModelGradient[j], p, x, ref yHessJ);
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
            hessian[j, k] += (yJac[i, j] * yJac[i, k] + res * yHess[j, i, k]) / (sErr * sErr);
          }
        }
      }

      return hessian;
    }

    // for the calculation of deviance
    public override double BestNegLogLikelihood(double[] p) {
      int m = y.Length;
      return (m / 2.0) * Math.Log(2 * sErr * sErr * Math.PI); // residuals are zero
    }

    public override double NegLogLikelihood(double[] p) {
      NegLogLikelihoodGradient(p, out var nll, nll_grad: null);
      return nll;
    }

    public override void NegLogLikelihoodGradient(double[] p, out double nll, double[] nll_grad) {
      var m = y.Length;
      var n = p.Length;
      double[,] nll_jac = null;

      // get likelihoods and gradients for each row
      var nllArr = new double[m];
      if (nll_grad != null) {
        nll_jac = new double[m, n];
        Array.Clear(nll_grad, 0, n);
      }
      NegLogLikelihoodJacobian(p, nllArr, nll_jac);

      // sum over all rows
      nll = 0.0;
      for (int i = 0; i < m; i++) {
        nll += nllArr[i];

        if (nll_grad != null) {
          for (int j = 0; j < n; j++) {
            nll_grad[j] += nll_jac[i, j];
          }
        }
      }
    }

    public void NegLogLikelihoodJacobian(double[] p, double[] nll, double[,] nll_jac) {
      if (nll.Length != x.GetLength(0)) throw new ArgumentException("length != nrows(x)", nameof(nll));

      var yPred = interpreter.EvaluateWithJac(p, null, nll_jac);

      for (int i = 0; i < nll.Length; i++) {
        var res = yPred[i] - y[i];
        nll[i] = 0.5 * Math.Log(2 * sErr * sErr * Math.PI)
                 + 0.5 * res * res / (sErr * sErr);

        if (nll_jac != null) {
          for (int j = 0; j < p.Length; j++) {
            nll_jac[i, j] = nll_jac[i, j] * res / (sErr * sErr);
          }
        }
      }
    }

    public override LikelihoodBase Clone() {
      return new SimpleGaussianLikelihood(this);
    }
  }
}

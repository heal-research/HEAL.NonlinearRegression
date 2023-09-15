using HEAL.Expressions;
using System;
using System.Linq.Expressions;

namespace HEAL.NonlinearRegression {

  public class BernoulliLikelihood : LikelihoodBase {

    internal BernoulliLikelihood(BernoulliLikelihood original) : base(original) { }
    public BernoulliLikelihood(double[,] x, double[] y, Expression<Expr.ParametricFunction> modelExpr) : base(modelExpr, x, y, 0) { }

    public override double[,] FisherInformation(double[] p) {
      var m = y.Length;
      var n = p.Length;
      var yPred = new double[m];
      var tmp = new double[m];
      var yJac = new double[m, n];
      var yHess = new double[n, m, n]; // parameters x observations x parameters (collections of Jacobians)
      var yHessJ = new double[m, n]; // buffer

      // var yPred = Expr.EvaluateFuncJac(ModelExpr, p, x, ref yJac);
      Interpreter.EvaluateWithJac(p, yPred, null, yJac);

      // evaluate hessian
      for (int j = 0; j < p.Length; j++) {
        // Expr.EvaluateFuncJac(ModelGradient[j], p, x, ref yHessJ);
        GradInterpreter[j].EvaluateWithJac(p, tmp, null, yHessJ);
        Buffer.BlockCopy(yHessJ, 0, yHess, j * m * n * sizeof(double), m * n * sizeof(double));
        Array.Clear(yHessJ, 0, yHessJ.Length);
      }

      var hess = new double[n, n];

      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          var s = 1 / ((1 - yPred[i]) * (1 - yPred[i]) * yPred[i] * yPred[i]);
          for (int k = 0; k < n; k++) {
            var hessianTerm = (yPred[i] - 1) * yPred[i] * yHess[j, i, k] * (y[i] - yPred[i]);
            var gradientTerm = (-2 * y[i] * yPred[i] + yPred[i] * yPred[i] + y[i]) * yJac[i, j] * yJac[i, k];
            hess[j, k] += s * (hessianTerm + gradientTerm);
          }
        }
      }
      return hess;
    }

    public override double BestNegLogLikelihood(double[] p) => 0.0;
    public override double NegLogLikelihood(double[] p) {
      NegLogLikelihoodGradient(p, out var nll, nll_grad: null);
      return nll;
    }

    public override void NegLogLikelihoodGradient(double[] p, out double nll, double[] nll_grad) {
      var m = y.Length;
      var n = p.Length;
      double[,] yJac = null;

      nll = BestNegLogLikelihood(p);
      double[] yPred = new double[m];
      if (nll_grad == null) {
        Interpreter.Evaluate(p, yPred);
      } else {
        yJac = new double[m, nll_grad.Length];
        Interpreter.EvaluateWithJac(p, yPred, null, yJac);
        Array.Clear(nll_grad, 0, n);
      }

      for (int i = 0; i < m; i++) {
        if (y[i] != 0.0 && y[i] != 1.0) throw new ArgumentException("target variable must be binary (0/1) for Bernoulli likelihood");
        if (y[i] == 1) {
          nll -= Math.Log(yPred[i]);
          if (nll_grad != null) {
            for (int j = 0; j < n; j++) {
              nll_grad[j] -= yJac[i, j] / yPred[i];
            }
          }
        } else {
          // y[i]==0
          nll -= Math.Log(1 - yPred[i]);
          if (nll_grad != null) {
            for (int j = 0; j < n; j++) {
              nll_grad[j] += yJac[i, j] / (1 - yPred[i]);
            }
          }
        }
      }
    }

    public override LikelihoodBase Clone() {
      return new BernoulliLikelihood(this);
    }
  }
}

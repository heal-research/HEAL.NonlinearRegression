

using HEAL.Expressions;
using Microsoft.CodeAnalysis;
using System.Globalization;

namespace HEAL.NonlinearRegression.Console.Tests {
  [TestClass]
  public class MDLTests {
    [TestMethod]
    public void MDL_ExhaustiveSymbolicRegression() {
      // cosmic chronometer data used in https://arxiv.org/pdf/2211.11461.pdf
      // taken from https://raw.githubusercontent.com/DeaglanBartlett/ESR/main/ESR/data/CC_Table1_2201.07241.tsv

      CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
      CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

      Program.ReadData("cosmicChronometer.csv", "H", out var varNames, out var x, out var y);


      // In the paper they use (x=z+1) as input.
      // The second column in X is the error
      var hErr = new double[y.Length];
      for (int i = 0; i < y.Length; i++) {
        x[i, 0] = 1 + x[i, 0];
        hErr[i] = x[i, 1];
      }

      // in the paper they use sqrt(model) for the calculation of likelihoods
      var models = new string[] {
        "sqrt(3883.44 * x * x)",
//        "(3982.43 ^ x) ^ 0.22",
//        "1414.43 * 0.31^(-x)",
        "sqrt(3834.51 * x ^ 2.03)"
      };
      var numSym = new int[] {
        3, // 3 different symbols (TODO: check whether this is valid)
        //3,
        //,
        4
      };
      for (int i = 0; i < models.Length; i++) {
        var model = models[i];
        var numSymbols = numSym[i];
        Program.GenerateExpression(model, new[] { "x" }, out var expr, out var p);
        var SSR = Program.EvaluateSSR(expr, p, x, y, out var yPred);

        // var SSR = 0.0;
        // for (int r = 0; r < y.Length; r++) {
        //   yPred[r] = Math.Sqrt(r);
        //   var res = y[r] - yPred[r];
        //   SSR += res * res;
        // }

        var nmse = SSR / y.Length / Util.Variance(y);

        var _jac = Expr.Jacobian(expr, p.Length).Compile();
        void jac(double[] p, double[,] X, double[] f, double[,] jac) => _jac(p, X, f, jac);
        var stats = new LeastSquaresStatistics(y.Length, p.Length, SSR, yPred, p, jac, x);

        var constants = Expr.CollectConstants(expr);
        var numNodes = Expr.NumberOfNodes(expr) - 1; // remove the sqrt which is not included in the ESR paper

        // calculation as in ESR code: 0.5 * np.dot((mu_pred - self.yvar), np.dot(self.inv_cov,(mu_pred - self.yvar)))
        var negLogLik = 0.0;
        var hess = Expr.Hessian(expr, p.Length).Compile();
        var grad = Expr.Gradient(expr, p.Length).Compile();

        var H = new double[p.Length, p.Length]; // buffer for Hessian calculation
        var Hi = new double[p.Length, p.Length]; // buffer for Hessian calculation for each row
        var gi = new double[p.Length]; // buffer for gradient calculation for each row
        var xi = new double[1];
        for (int r = 0; r < y.Length; r++) {
          var res = y[r] - yPred[r];

          negLogLik += 0.5 / (hErr[r] * hErr[r]) * res * res; // dropped some terms from the likelihood

          xi[0] = x[r, 0];
          hess(p, xi, Hi);
          grad(p, xi, gi);

          for (int k = 0; k < p.Length; k++)
            for (int l = 0; l < p.Length; l++) {
              H[k, l] += (res * Hi[k, l] - gi[k] * gi[l]) / (hErr[r] * hErr[r]); // full Hessian
            }
        }


        // I = -H
        var I = new double[p.Length, p.Length];
        for (int k = 0; k < p.Length; k++) {
          for (int l = 0; l < p.Length; l++) {
            I[k, l] = -H[k, l];
          }
        }
        int numParam = stats.n;
        // calculate the three terms of the code length separately as in the ESR code to compare
        var t1 = negLogLik; // residuals
        var t2 = +numNodes * Math.Log(numSymbols) + constants.Sum(ci => Math.Log(Math.Abs(ci))); // function
        var t3 = -numParam / 2.0 * Math.Log(3.0) + Enumerable.Range(0, numParam).Sum(i => 0.5 * Math.Log(I[i, i]) + Math.Log(Math.Abs(stats.paramEst[i])));
        System.Console.WriteLine($"{model} nll: {t1} function: {t2} parameters: {t3}");



        // expected results:
        // complexity = number of nodes = 5
        // code length T1: neg. log. likeihood = 8.36
        // code length T2: function complexity = k log n + sum_j log(c_j)
        // code length T3: parameter complexity = 2.53
        // total code length: 16.39
        var mdl = MinimumDescriptionLength.MDL(expr, p, y, x, numNodes, numSymbols, constants);
        System.Console.WriteLine($"{model} LogLik: {stats.LogLikelihood} MDL: {mdl} DoF: {p.Length} NumNodes: {Expr.NumberOfNodes(expr)} num constants: {Expr.CollectConstants(expr).Length}");
      }
    }
  }
}

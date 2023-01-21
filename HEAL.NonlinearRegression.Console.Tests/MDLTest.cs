

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

      // in the paper they use H² as target and (x=z+1) as input
      for (int i = 0; i < y.Length; i++) {
        y[i] = y[i] * y[i];
        x[i, 0] = 1 + x[i,0];
      }

      var models = new string[] {
        "3883.44 * x * x",
//        "(3982.43 ^ x) ^ 0.22",
//        "1414.43 * 0.31^(-x)",
        "3834.51 * x ^ 2.03"
      };
      var numSym = new int[] {
        3, // 3 different symbols (TODO: check whether this is valid)
        //3,
        //,
        4
      };
      for(int i=0;i<models.Length;i++)
      {
        var model = models[i];
        var numSymbols = numSym[i];
        Program.GenerateExpression(model, new[] { "x" }, out var expr, out var p);
        var SSR = Program.EvaluateSSR(expr, p, x, y, out var yPred);
        var nmse = SSR / y.Length / Util.Variance(y);

        var _jac = Expr.Jacobian(expr, p.Length).Compile();
        void jac(double[] p, double[,] X, double[] f, double[,] jac) => _jac(p, X, f, jac);
        var stats = new LeastSquaresStatistics(y.Length, p.Length, SSR, yPred, p, jac, x);

        var mdl = MinimumDescriptionLength.MDL(stats, Expr.NumberOfNodes(expr), numSymbols, Expr.CollectConstants(expr));

        // expected results:
        // complexity = number of nodes = 5
        // code length T1: neg. log. likeihood = 8.36
        // code length T2: function complexity = k log n + sum_j log(c_j)
        // code length T3: parameter complexity = 2.53
        // total code length: 16.39

        System.Console.WriteLine($"{model} LogLik: {stats.LogLikelihood} MDL: {mdl} DoF: {p.Length} NumNodes: {Expr.NumberOfNodes(expr)} num constants: {Expr.CollectConstants(expr).Length}");
      }
    }
  }
}

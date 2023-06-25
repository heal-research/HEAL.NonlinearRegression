using NUnit.Framework;
using System.Globalization;

namespace HEAL.NonlinearRegression.Console.Tests {
  public class ModelSelection {
    [SetUp]
    public void Setup() {
      CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
      CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;
      Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
      Thread.CurrentThread.CurrentUICulture = CultureInfo.InvariantCulture;
    }

    [Test]
    public void FitPuromycin() {
      #region data
      var x = new double[,] {
                             { 0.02 },
                             { 0.02 },
                             { 0.06 },
                             { 0.06 },
                             { 0.11 },
                             { 0.11 },
                             { 0.22 },
                             { 0.22 },
                             { 0.56 },
                             { 0.56 },
                             { 1.10 },
                             { 1.10 }};
      var y = new double[] {76
                           ,47
                           ,97
                           ,107
                           ,123
                           ,139
                           ,159
                           ,152
                           ,191
                           ,201
                           ,207
                           ,200 };
      #endregion

      var nlr = new NonlinearRegression();
      var modelExpr = "0.1 * x0 / (1.0f + 0.1 * x0)";
      var parser = new HEAL.Expressions.Parser.ExprParser(modelExpr, 
        new[] { "x0" },
        System.Linq.Expressions.Expression.Parameter(typeof(double[]), "x"), 
        System.Linq.Expressions.Expression.Parameter(typeof(double[]), "p"));
      var likelihood = new SimpleGaussianLikelihood(x, y, parser.Parse());
      nlr.Fit(parser.ParameterValues, likelihood);
      // HEAL.NonlinearRegression.ModelAnalysis.NestedModels(likelihood.ModelExpr, nlr.ParamEst, nlr.LaplaceApproximation);
      // HEAL.NonlinearRegression.ModelSelection.MDLLatticeWithIntegerSnap(nlr.ParamEst, likelihood, nlr.LaplaceApproximation, out var bestExpr, out var bestParamEst);
    }
  }
}

using NUnit.Framework;
using System.Globalization;

namespace HEAL.NonlinearRegression.Console.Tests {
  public class NLR {
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

      System.Console.WriteLine($"Deviance: {nlr.Deviance:e4}, BIC: {nlr.BIC:f2}");
      Assert.AreEqual(96.91354730673082, nlr.BIC, 1e-5);

      var prediction = nlr.PredictWithIntervals(x, IntervalEnum.LaplaceApproximation);
      System.Console.WriteLine($"pred: {prediction[0, 0]}, low: {prediction[0, 2]}, high: {prediction[0, 3]}");
      Assert.AreEqual(50.565977770482867, prediction[0, 0], 1e-4);
      Assert.AreEqual(41.543747215791058, prediction[0, 2], 1e-4);
      Assert.AreEqual(59.588208325174676, prediction[0, 3], 1e-6);
    }
  }
}

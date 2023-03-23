using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Threading.Tasks;
using CommandLine;
using HEAL.Expressions;
using HEAL.Expressions.Parser;

// TODO:
//   - Remove everything that is specific to Gaussian likelihoods (s, SSR) (in model analysis, impact calculation, model evaluation, ...) use deviance instead
//   - Option to set likelihood function on the command line
//   - Clean up expression code to provide functions with inverses and derivatives via mapping functions (no special handling of inverse functions and derivatives within visitors)
//   - Implement likelihoods symbolically (on top of the model functions) to support autodiff for likelihood gradients and likelihood Hessians. They are now hard-coded.
//   - Numerically stable implementation for Bernoulli likelihood (using log1p() ?)
//   - alglib is GPL, should switch to .NET numerics (MIT) instead.
//   - iterative pruning based on subtree impacts or likelihood ratios for nested models
//   - variable impacts for combinations of variables (tuples, triples). Contributions to individual variables via Shapely values?
//   - nested model analysis for combinations of parameters (for the case where a parameter can only be set to zero if another parameter is also set to zero)
//   - If a range is specified (training, test) then only read the relevant rows of data

namespace HEAL.NonlinearRegression.Console {
  // Intended to be used together with Operon.
  // Use the suffix 'f' to mark real literals in the model as fixed instead of a parameter.
  // e.g. 10 * x^2f ,  2 is a fixed constant, 10 is a parameter of the model
  public class Program {

    public static void Main(string[] args) {
      System.Globalization.CultureInfo.DefaultThreadCurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
      var parserResult = Parser.Default.ParseArguments<PredictOptions, FitOptions, SimplifyOptions, NestedModelsOptions, PruneOptions,
        SubtreeImportanceOptions, CrossValidationOptions, VariableImpactOptions, EvalOptions, PairwiseProfileOptions, ProfileOptions,
        RankDeterminationOptions>(args)
        .WithParsed<PredictOptions>(options => Predict(options))
        .WithParsed<FitOptions>(options => Fit(options))
        .WithParsed<EvalOptions>(options => Evaluate(options))
        .WithParsed<SimplifyOptions>(options => Simplify(options))
        .WithParsed<NestedModelsOptions>(options => GenerateNestedModels(options))
        .WithParsed<PruneOptions>(options => Prune(options))
        .WithParsed<SubtreeImportanceOptions>(options => SubtreeImportance(options))
        .WithParsed<CrossValidationOptions>(options => CrossValidate(options))
        .WithParsed<VariableImpactOptions>(options => CalculateVariableImpacts(options))
        .WithParsed<PairwiseProfileOptions>(options => PairwiseProfiles(options))
        .WithParsed<ProfileOptions>(options => Profiles(options))
        .WithParsed<RankDeterminationOptions>(options => CalculateRank(options))
        ;
      ;
    }



    #region verbs
    private static void Fit(FitOptions options) {
      PrepareData(options, out var varNames, out var x, out var y, out var trainStart, out var trainEnd, out var testStart, out var testEnd, out var trainX, out var trainY);
      foreach (var model in GetModels(options.Model)) {
        GenerateExpression(model, varNames, out var parametricExpr, out var p);

        var nls = new NonlinearRegression();
        try {
          nls.Fit(p, parametricExpr, options.Likelihood, trainX, trainY, options.MaxIterations);
        } catch (Exception e) {
          System.Console.WriteLine("There was a problem while fitting.");
          continue;
        }
        if (nls.OptReport.Success) {
          System.Console.WriteLine($"p_opt: {string.Join(" ", p.Select(pi => pi.ToString("e5")))}");
          System.Console.WriteLine($"{nls.OptReport}");
          nls.WriteStatistics();
          System.Console.WriteLine($"Optimized: {Expr.ToString(parametricExpr, varNames, p)}");
        } else {
          System.Console.WriteLine("There was a problem while fitting.");
        }
      }
    }

    private static IEnumerable<string> GetModels(string optionsModel) {
      if (File.Exists(optionsModel)) {
        return File.ReadLines(optionsModel);
      } else {
        return optionsModel.Split("\n");
      }
    }

    private static void Evaluate(EvalOptions options) {
      ReadData(options.Dataset, options.Target, out var varNames, out var x, out var y);

      // default is full dataset
      var start = 0;
      var end = y.Length - 1;
      if (options.Range != null) {
        var toks = options.Range.Split(":");
        start = int.Parse(toks[0]);
        end = int.Parse(toks[1]);
      }

      Split(x, y, start, end, start, end, out x, out y, out _, out _);

      foreach (var model in GetModels(options.Model)) {
        try {
          // TODO: most of the result values are specific to Gaussian likelihood
          GenerateExpression(model, varNames, out var parametricExpr, out var p);

          var nlr = new NonlinearRegression();
          nlr.SetModel(p, parametricExpr, options.Likelihood, x, y);
          var m = y.Length;
          var SSR = nlr.Statistics.SSR;

          var noiseSigma = options.NoiseSigma ?? nlr.Statistics.s; // use estimated noise standard error as default
          var nmse = SSR / y.Length / Util.Variance(y);
          
          var stats = nlr.Statistics;
          var mdl = MinimumDescriptionLength.MDL(parametricExpr, p, -nlr.NegLogLikelihood, y, noiseSigma, x, approxHessian: true);
          var freqMdl = MinimumDescriptionLength.MDLFreq(parametricExpr, p, -nlr.NegLogLikelihood, y, noiseSigma, x, approxHessian: false);

          var logLik = -nlr.NegLogLikelihood;
          var aicc = nlr.AICc;
          var bic = nlr.BIC;

          System.Console.WriteLine($"SSR: {SSR} MSE: {SSR / y.Length} RMSE: {Math.Sqrt(SSR / y.Length)} NMSE: {nmse} R2: {1 - nmse} LogLik: {logLik} AICc: {aicc} BIC: {bic} MDL: {mdl} MDL(freq): {freqMdl} DoF: {p.Length}");
        } catch (Exception e) {
          System.Console.WriteLine($"Could not evaluate model {model}");
        }
      }
    }

    public static double EvaluateSSR(Expression<Expr.ParametricFunction> parametricExpr, double[] p, double[,] x, double[] y, out double[] yPred) {
      var func = Expr.Broadcast(parametricExpr).Compile();

      int m;
      double SSR;
      m = y.Length;
      yPred = new double[m];
      func(p, x, yPred);

      SSR = 0.0;
      for (int i = 0; i < m; i++) {
        var r = y[i] - yPred[i];
        SSR += r * r;
      }
      return SSR;
    }

    private static void Predict(PredictOptions options) {
      PrepareData(options, out var varNames, out var x, out var y, out var trainStart, out var trainEnd, out var testStart, out var testEnd, out var trainX, out var trainY);

      if(options.Range != null) {
        // calc predictions only for the specified range
        var rangeTokens = options.Range.Split(":");
        var predStart = int.Parse(rangeTokens[0]);
        var predEnd = int.Parse(rangeTokens[1]);
        Split(x, y, predStart, predEnd, 0, x.GetLength(0) - 1, out var predX, out var predY, out var _, out var _);
        x = predX;
        y = predY;
      } 
      foreach (var model in GetModels(options.Model)) {
        GenerateExpression(model, varNames, out var parametricExpr, out var p);

        var nlr = new NonlinearRegression();
        if (options.NoOptimization) {
          nlr.SetModel(p, parametricExpr, options.Likelihood, trainX, trainY);
        } else {
          nlr.Fit(p, parametricExpr, options.Likelihood, trainX, trainY);
        }

        var predict = nlr.PredictWithIntervals(x, options.Interval);

        // generate output for full dataset
        if (options.Interval == IntervalEnum.None) {
          System.Console.WriteLine($"{string.Join(",", varNames)},y,residual,yPred,isTrain,isTest"); // header without interval
        } else {
          System.Console.WriteLine($"{string.Join(",", varNames)},y,residual,yPred,yPredLow,yPredHigh,isTrain,isTest"); // header
        }
        for (int i = 0; i < x.GetLength(0); i++) {
          for (int j = 0; j < x.GetLength(1); j++) {
            System.Console.Write($"{x[i, j]},");
          }
          System.Console.Write($"{y[i]},");
          System.Console.Write($"{y[i] - predict[i, 0]},");
          System.Console.Write($"{predict[i, 0]},");
          if (options.Interval == IntervalEnum.LaplaceApproximation)
            System.Console.Write($"{predict[i, 2]},{predict[i, 3]},");
          else if (options.Interval == IntervalEnum.TProfile)
            System.Console.Write($"{predict[i, 1]},{predict[i, 2]},");
          System.Console.Write($"{((i >= trainStart && i <= trainEnd) ? 1 : 0)},"); // isTrain
          System.Console.Write($"{((i >= testStart && i <= testEnd) ? 1 : 0)}"); // isTest
          System.Console.WriteLine();
        }
      }
    }


    private static void CrossValidate(CrossValidationOptions options) {
      ReadData(options.Dataset, options.Target, out var varNames, out var x, out var y);

      // default is full dataset
      var trainStart = 0;
      var trainEnd = y.Length - 1;
      if (options.TrainingRange != null) {
        var toks = options.TrainingRange.Split(":");
        trainStart = int.Parse(toks[0]);
        trainEnd = int.Parse(toks[1]);
      }

      Split(x, y, trainStart, trainEnd, trainStart, trainEnd, out var trainX, out var trainY, out _, out _);

      foreach (var model in GetModels(options.Model)) {
        try {
          GenerateExpression(model, varNames, out var parametricExpr, out var p);

          var cvmse = CrossValidate(parametricExpr, options.Likelihood, p, trainX, trainY, shuffle: options.Shuffle, seed: options.Seed);
          var stddev = Math.Sqrt(Util.Variance(cvmse.ToArray()));
          System.Console.WriteLine($"mean(MSE): {cvmse.Average():e4} stddev(MSE): {stddev:e4} se(MSE): {stddev / Math.Sqrt(cvmse.Count())}"); // Elements of Statistical Learning
        } catch (Exception e) {
          System.Console.WriteLine($"Error in fitting model {model}");
        }
      }
    }

    private static List<double> CrossValidate(Expression<Expr.ParametricFunction> parametricExpr, LikelihoodEnum likelihood, double[] p, double[,] X, double[] y, int folds = 10, bool shuffle = false, int? seed = null) {
      Random rand;
      if (seed.HasValue) {
        rand = new Random(seed.Value);
      } else {
        rand = new Random();
      }

      if (shuffle) {
        Shuffle(X, y, rand);
      }


      var foldSize = (int)Math.Truncate((y.Length + 1) / (double)folds);
      var mse = new List<double>();
      Parallel.For(0, folds, (f) => {
        var foldStart = f * foldSize;
        var foldEnd = (f + 1) * foldSize - 1;
        if (f == folds - 1) {
          foldEnd = y.Length - 1; // include remaining part in last fold
        }

        DeletePartition(X, y, foldStart, foldEnd, out var foldTrainX, out var foldTrainY);
        Split(X, y, foldStart, foldEnd, foldStart, foldEnd, out var foldTestX, out var foldTestY, out _, out _);

        var nls = new NonlinearRegression();
        nls.Fit(p, parametricExpr, likelihood, foldTrainX, foldTrainY, maxIterations: 5000); // TODO make CLI parameter

        var foldPred = nls.Predict(foldTestX);
        var SSRtest = 0.0;
        for (int i = 0; i < foldTestY.Length; i++) {
          var r = foldTestY[i] - foldPred[i];
          SSRtest += r * r;
        }
        lock (mse) {
          mse.Add(SSRtest / foldTestY.Length);
        }
      });
      return mse;
    }

    private static void Simplify(SimplifyOptions options) {
      var varNames = options.Variables.Split(',').Select(vn => vn.Trim()).ToArray();

      foreach (var model in GetModels(options.Model)) {
        GenerateExpression(model, varNames, out var parametricExpr, out var p);
        Expression<Expr.ParametricFunction> simplifiedExpr;

        Simplify(parametricExpr, p, out simplifiedExpr, out var newP);

        // System.Console.WriteLine(simplifiedExpr);
        // System.Console.WriteLine($"theta: {string.Join(",", newP.Select(pi => pi.ToString()))}");

        System.Console.WriteLine(Expr.ToString(simplifiedExpr, varNames, newP));
      }
    }

    private static void Simplify(Expression<Expr.ParametricFunction> parametricExpr, double[] p, out Expression<Expr.ParametricFunction> simplifiedExpr, out double[] newP) {
      simplifiedExpr = Expr.FoldParameters(parametricExpr, p, out newP);
      var newSimplifiedStr = simplifiedExpr.ToString();
      var exprSet = new HashSet<string>();
      // simplify until no change (TODO: this shouldn't be necessary if visitors are implemented carefully)
      do {
        exprSet.Add(newSimplifiedStr);
        simplifiedExpr = Expr.FoldParameters(simplifiedExpr, newP, out newP);
        // System.Console.WriteLine(Expr.ToString(simplifiedExpr, varNames, newP));
        newSimplifiedStr = simplifiedExpr.ToString();
      } while (!exprSet.Contains(newSimplifiedStr));
    }

    private static void GenerateNestedModels(NestedModelsOptions options) {
      ReadData(options.Dataset, options.Target, out var varNames, out var x, out var y);

      // default is full dataset
      var trainStart = 0;
      var trainEnd = y.Length - 1;
      var testStart = 0;
      var testEnd = y.Length - 1;
      if (options.TrainingRange != null) {
        var toks = options.TrainingRange.Split(":");
        trainStart = int.Parse(toks[0]);
        trainEnd = int.Parse(toks[1]);
        testStart = trainEnd + 1;
        testEnd = y.Length - 1;
      }

      Split(x, y, trainStart, trainEnd, testStart, testEnd, out var trainX, out var trainY, out var testX, out var testY);


      foreach (var model in GetModels(options.Model)) {
        GenerateExpression(model, varNames, out var parametricExpr, out var p);

        // calculate ref stats for full model
        var nlr = new NonlinearRegression();
        nlr.Fit(p, parametricExpr, options.Likelihood, trainX, trainY, options.MaxIterations);
        var refStats = nlr.Statistics;
        var noiseSigma = refStats.s;
        var refAicc = nlr.AICc;
        var refBic = nlr.BIC;

        var allModels = new List<Tuple<Expression<Expr.ParametricFunction>, double[]>>();
        var modelQueue = new Queue<Tuple<Expression<Expr.ParametricFunction>, double[]>>();
        modelQueue.Enqueue(Tuple.Create(parametricExpr, (double[])p.Clone()));



        while (modelQueue.Any()) {

          (parametricExpr, p) = modelQueue.Dequeue();
          allModels.Add(Tuple.Create(parametricExpr, p));
          var impacts = ModelAnalysis.NestedModelLiklihoodRatios(parametricExpr, options.Likelihood, trainX, trainY, (double[])p.Clone(), options.MaxIterations, options.Verbose);

          // order by SSRfactor and use the best as an alternative (if it has delta AICc < deltaAIC)
          var alternative = impacts.OrderBy(tup => tup.Item2).Where(tup => tup.Item3 < options.DeltaAIC).FirstOrDefault();
          if (alternative != null) {
            var ssrFactor = alternative.Item2;
            var deltaAICc = alternative.Item3;
            var reducedExpr = alternative.Item4;
            var reducedParam = alternative.Item5;
            System.Console.Error.WriteLine($"New model {reducedParam.Length} {ssrFactor:e4} {deltaAICc:e4} {reducedExpr}");
            modelQueue.Enqueue(Tuple.Create(reducedExpr, reducedParam));
          }
        }
        System.Console.WriteLine();
        System.Console.WriteLine($"SSR_Factor,numPar,RMSE_tr,RMSE_te,RMSE_cv,RMSE_cv_std,AICc,dAICc,BIC,dBIC,Model");
        foreach (var subModel in allModels) {
          (parametricExpr, p) = subModel;

          // output training, CV, and test result for original model
          nlr = new NonlinearRegression();
          nlr.SetModel(p, parametricExpr, options.Likelihood, trainX, trainY);
          var stats = nlr.Statistics;
          var ssrTrain = stats.SSR;
          var rmseTrain = Math.Sqrt(ssrTrain / trainY.Length);
          var ssrTest = EvaluateSSR(parametricExpr, p, testX, testY, out _);
          var rmseTest = Math.Sqrt(ssrTest / testY.Length);
          var cvrmseMean = double.NaN;
          var cvrmseStd = double.NaN;
          try {
            var cvrmse = CrossValidate(parametricExpr, options.Likelihood, p, trainX, trainY, folds: 5);
            cvrmseMean = cvrmse.Average();
            cvrmseStd = Math.Sqrt(Util.Variance(cvrmse.ToArray()));
          } catch (Exception) { }
          var aicc = nlr.AICc;
          var bic = nlr.BIC;
          System.Console.WriteLine($"{stats.SSR / refStats.SSR:e4},{stats.n},{rmseTrain:e4},{rmseTest:e4},{cvrmseMean:e4},{cvrmseStd:e4},{aicc:e4},{aicc - refAicc:e4},{bic:e4},{bic - refBic:e4},{Expr.ToString(parametricExpr, varNames, p)}");
        }
      }
    }

    // similar to generate nested models but returns only a single model
    private static void Prune(PruneOptions options) {
      ReadData(options.Dataset, options.Target, out var varNames, out var x, out var y);

      // default is full dataset
      var trainStart = 0;
      var trainEnd = y.Length - 1;
      var testStart = 0;
      var testEnd = y.Length - 1;
      if (options.TrainingRange != null) {
        var toks = options.TrainingRange.Split(":");
        trainStart = int.Parse(toks[0]);
        trainEnd = int.Parse(toks[1]);
        testStart = trainEnd + 1;
        testEnd = y.Length - 1;
      }

      Split(x, y, trainStart, trainEnd, testStart, testEnd, out var trainX, out var trainY, out var testX, out var testY);

      foreach (var model in GetModels(options.Model)) {
        GenerateExpression(model, varNames, out var origExpr, out var origParam);

        // calculate ref stats for full model
        var nlr = new NonlinearRegression();
        nlr.Fit(origParam, origExpr, options.Likelihood, trainX, trainY, options.MaxIterations);
        var refStats = nlr.Statistics;
        var noiseSigma = refStats.s; // assume RMSE of original model for noiseSigma
        if (refStats == null) {
          // could not fit expression
          System.Console.WriteLine($"deltaAICc: {double.NaN}, deltaN: {double.NaN}, {Expr.ToString(origExpr, varNames, origParam)}");
          continue;
        }

        var allModels = new List<Tuple<Expression<Expr.ParametricFunction>, double[]>>();
        var modelQueue = new Queue<Tuple<Expression<Expr.ParametricFunction>, double[]>>();
        modelQueue.Enqueue(Tuple.Create(origExpr, (double[])origParam.Clone()));

        while (modelQueue.Any()) {

          (var expr, var param) = modelQueue.Dequeue();
          allModels.Add(Tuple.Create(expr, param));
          var impacts = ModelAnalysis.NestedModelLiklihoodRatios(expr, options.Likelihood, trainX, trainY, (double[])param.Clone(), options.MaxIterations, options.Verbose);

          // order by SSRfactor and use the best as an alternative (if it has delta AICc < deltaAIC)
          var alternative = impacts.OrderBy(tup => tup.Item2).Where(tup => tup.Item3 < options.DeltaAIC).FirstOrDefault();
          if (alternative != null) {
            var ssrFactor = alternative.Item2;
            var deltaAICc = alternative.Item3;
            var reducedExpr = alternative.Item4;
            var reducedParam = alternative.Item5;
            System.Console.Error.WriteLine($"New model {reducedParam.Length} {ssrFactor:e4} {deltaAICc:e4} {reducedExpr}");
            modelQueue.Enqueue(Tuple.Create(reducedExpr, reducedParam));
          }
        }

        // find smallest model 
        var bestModel = origExpr;
        var bestParam = origParam;
        var bestNumParam = refStats.n;
        var refAICc = nlr.AICc;
        var bestAICc = refAICc;

        foreach (var subModel in allModels) {
          (var expr, var param) = subModel;

          // eval all models
          nlr = new NonlinearRegression();
          nlr.SetModel(param, expr, options.Likelihood, trainX, trainY);
          var stats = nlr.Statistics;
          var aicc = nlr.AICc;
          if (aicc - refAICc < options.DeltaAIC && stats.n <= bestNumParam) {
            bestModel = expr;
            bestNumParam = param.Length;
            bestAICc = aicc;
            bestParam = param;
          }
        }
        System.Console.WriteLine($"deltaAICc: {bestAICc - refAICc}, deltaN: {bestNumParam - refStats.n}, {Expr.ToString(bestModel, varNames, bestParam)}");
      }
    }


    private static void SubtreeImportance(SubtreeImportanceOptions options) {
      // TODO combine with nested models
      ReadData(options.Dataset, options.Target, out var varNames, out var x, out var y);

      // default is full dataset
      var trainStart = 0;
      var trainEnd = y.Length - 1;
      if (options.TrainingRange != null) {
        var toks = options.TrainingRange.Split(":");
        trainStart = int.Parse(toks[0]);
        trainEnd = int.Parse(toks[1]);
      }

      Split(x, y, trainStart, trainEnd, trainStart, trainEnd, out var trainX, out var trainY, out var testX, out var testY);

      foreach (var model in GetModels(options.Model)) {
        GenerateExpression(model, varNames, out var parametricExpr, out var p);

        var subExprImportance = ModelAnalysis.SubtreeImportance(parametricExpr, options.Likelihood, trainX, trainY, p);

        Dictionary<Expression, double> saturation = null;
        if (options.GraphvizFilename != null) {
          saturation = new Dictionary<Expression, double>();
          saturation[parametricExpr] = 0.0; // reference value for the importance
        }

        System.Console.WriteLine($"{"SSR_factor",-11} {"deltaAIC",-11} {"deltaBIC",-11} {"Subtree"}");
        foreach (var tup in subExprImportance.OrderByDescending(tup => tup.Item2)) { // TODO better interface
          System.Console.WriteLine($"{tup.Item2,-11:e4} {tup.Item3,-11:f1} {tup.Item4,-11:f1} {tup.Item1}");
          if (saturation != null) {
            saturation[tup.Item1] = Math.Max(0, Math.Log(tup.Item2)); // use log scale for coloring
          }
        }


        if (options.GraphvizFilename != null) {
          using (var writer = new StreamWriter(options.GraphvizFilename)) {
            writer.WriteLine(Expr.ToGraphViz(parametricExpr,
              paramValues: options.HideParameters ? null : p,
              varNames: options.HideVariables ? null : varNames,
              saturation));
          }
        }
      }
    }


    private static void CalculateVariableImpacts(VariableImpactOptions options) {
      // TODO combine with subtree impacts
      ReadData(options.Dataset, options.Target, out var varNames, out var x, out var y);

      // default is full dataset
      var trainStart = 0;
      var trainEnd = y.Length - 1;
      if (options.TrainingRange != null) {
        var toks = options.TrainingRange.Split(":");
        trainStart = int.Parse(toks[0]);
        trainEnd = int.Parse(toks[1]);
      }

      Random rand;
      if (options.Seed.HasValue) {
        rand = new Random(options.Seed.Value);
      } else {
        rand = new Random();
      }

      if (options.Shuffle) {
        Shuffle(x, y, rand);
      }

      Split(x, y, trainStart, trainEnd, trainStart, trainEnd, out var trainX, out var trainY, out _, out _);

      foreach (var model in GetModels(options.Model)) {
        GenerateExpression(model, varNames, out var parametricExpr, out var p);

        var varImportance = ModelAnalysis.VariableImportance(parametricExpr, options.Likelihood, trainX, trainY, p);


        System.Console.WriteLine($"{"variable",-11} {"VarExpl",-11}");
        foreach (var tup in varImportance.OrderByDescending(tup => tup.Value)) { // TODO better interface
          var varName = varNames[tup.Key];
          System.Console.WriteLine($"{varName,-11} {tup.Value * 100,-11:f2}%");
        }
      }
    }

    private static void PairwiseProfiles(PairwiseProfileOptions options) {
      // TODO combine with subtree impacts
      ReadData(options.Dataset, options.Target, out var varNames, out var x, out var y);

      // default is full dataset
      var trainStart = 0;
      var trainEnd = y.Length - 1;
      if (options.TrainingRange != null) {
        var toks = options.TrainingRange.Split(":");
        trainStart = int.Parse(toks[0]);
        trainEnd = int.Parse(toks[1]);
      }

      Split(x, y, trainStart, trainEnd, trainStart, trainEnd, out var trainX, out var trainY, out _, out _);

      foreach (var model in GetModels(options.Model)) {
        GenerateExpression(model, varNames, out var parametricExpr, out var parameters);

        var nlr = new NonlinearRegression();
        nlr.Fit(parameters, parametricExpr, options.Likelihood, trainX, trainY);


        var folder = Path.GetDirectoryName(options.Dataset);
        var filename = Path.GetFileNameWithoutExtension(options.Dataset);

        var tProfile = new TProfile(nlr.Statistics, nlr.NegLogLikelihoodFunc);
        var numPairs = (parameters.Length * (parameters.Length - 1) / 2.0);
        for (int i = 0; i < parameters.Length - 1; i++) {
          for (int j = i + 1; j < parameters.Length; j++) {
            System.Console.WriteLine($"{numPairs--}");
            var outfilename = Path.Combine(folder, filename + $"_{i}_{j}.csv");
            tProfile.ApproximateProfilePairContour(i, j, alpha: 0.20, out var taup95, out var tauq95, out var p95, out var q95);
            tProfile.ApproximateProfilePairContour(i, j, alpha: 0.50, out var taup50, out var tauq50, out var p50, out var q50);
            using (var writer = new StreamWriter(new FileStream(outfilename, FileMode.Create))) {
              writer.WriteLine("taup80,tauq80,p80,q80,taup50,tauq50,p50,q50");
              for (int l = 0; l < taup95.Length; l++) {
                writer.WriteLine($"{taup95[l]},{tauq95[l]},{p95[l]},{q95[l]},{taup50[l]},{tauq50[l]},{p50[l]},{q50[l]}");
              }
            }
          }
        }
      }
    }

    private static void Profiles(ProfileOptions options) {
      // TODO combine with subtree impacts
      ReadData(options.Dataset, options.Target, out var varNames, out var x, out var y);

      // default is full dataset
      var trainStart = 0;
      var trainEnd = y.Length - 1;
      if (options.TrainingRange != null) {
        var toks = options.TrainingRange.Split(":");
        trainStart = int.Parse(toks[0]);
        trainEnd = int.Parse(toks[1]);
      }

      Split(x, y, trainStart, trainEnd, trainStart, trainEnd, out var trainX, out var trainY, out _, out _);

      foreach (var model in GetModels(options.Model)) {
        GenerateExpression(model, varNames, out var parametricExpr, out var parameters);

        var nlr = new NonlinearRegression();
        nlr.Fit(parameters, parametricExpr, options.Likelihood, trainX, trainY);
        var folder = Path.GetDirectoryName(options.Dataset);
        var filename = Path.GetFileNameWithoutExtension(options.Dataset);

        var tProfile = new TProfile(nlr.Statistics, nlr.NegLogLikelihoodFunc);

        // TODO: output profile based confidence intervals

        for (int pIdx = 0; pIdx < parameters.Length; pIdx++) {
          tProfile.GetProfile(pIdx, out var p, out var tau, out var p_stud);
          var outfilename = Path.Combine(folder, filename + $"_profile_{pIdx}.csv");
          using (var writer = new StreamWriter(new FileStream(outfilename, FileMode.Create))) {
            writer.WriteLine("tau,p,p_stud");
            for (int i = 0; i < p.Length; i++) {
              writer.WriteLine($"{tau[i]},{p[i]},{p_stud[i]}");
            }
          }
        }
      }
    }



    private static void CalculateRank(RankDeterminationOptions options) {
      // TODO combine with subtree impacts
      ReadData(options.Dataset, options.Target, out var varNames, out var x, out var y);

      // default is full dataset
      var trainStart = 0;
      var trainEnd = y.Length - 1;
      if (options.TrainingRange != null) {
        var toks = options.TrainingRange.Split(":");
        trainStart = int.Parse(toks[0]);
        trainEnd = int.Parse(toks[1]);
      }

      Split(x, y, trainStart, trainEnd, trainStart, trainEnd, out var trainX, out var trainY, out _, out _);

      foreach (var model in GetModels(options.Model)) {
        GenerateExpression(model, varNames, out var parametricExpr, out var parameters);

        if (!options.NoOptimization) {
          var nlr = new NonlinearRegression();
          nlr.Fit(parameters,  parametricExpr, options.Likelihood, trainX, trainY, maxIterations: 3000);
        }

        var _jac = Expr.Jacobian(parametricExpr, parameters.Length).Compile();

        var m = trainY.Length;
        var n = parameters.Length;
        var f = new double[m];
        var jac = new double[m, n];
        _jac(parameters, trainX, f, jac); // get Jacobian
        alglib.rmatrixsvd(jac, m, n, 0, 0, 0, out var w, out var u, out var vt);

        if (w.Any(wi => double.IsNaN(wi))) {
          System.Console.WriteLine("Jacobian undefined");
        } else {

          var eps = 2.2204460492503131E-16; // the difference between 1.0 and the next larger double value
                                            // var eps = 1.192092896e-7f; for floats
          var tol = m * eps;
          var rank = 0;
          for (int i = 0; i < n; i++) {
            if (w[i] > tol * w[0]) rank++;
          }
          // full condition number largest singular value over smallest singular value
          var k = w[0] / w[n - 1];
          var k_subset = w[0] / w[rank - 1]; // condition number without the redundant parameters
          System.Console.WriteLine($"Num param: {n} rank: {rank} log10_K(J): {Math.Log10(k)} log10_K(J_rank): {Math.Log10(k_subset)}");
        }
      }
    }
    #endregion

    #region  options
    public abstract class OptionsBase {
      [Option('d', "dataset", Required = true, HelpText = "Filename with dataset in csv format.")]
      public string Dataset { get; set; }

      [Option('t', "target", Required = true, HelpText = "Target variable name.")]
      public string Target { get; set; }

      [Option('m', "model", Required = true, HelpText = "The model in infix form as produced by Operon. Separate multiple models with newlines.")]
      public string Model { get; set; }

      [Option('l', "likelihood", Required = false, HelpText = "The likelihood function for the model (Gaussian or Bernoulli) (default: Gaussian)", Default = LikelihoodEnum.Gaussian)]
      public LikelihoodEnum Likelihood { get; set; }

      [Option("train", Required = false, HelpText = "The training range <firstRow>:<lastRow> in the dataset (inclusive).")]
      public string TrainingRange { get; set; }

      [Option("test", Required = false, HelpText = "The testing range <firstRow>:<lastRow> in the dataset (inclusive).")]
      public string TestingRange { get; set; }

      [Option("shuffle", Required = false, Default = false, HelpText = "Switch to shuffle the dataset before fitting.")]
      public bool Shuffle { get; set; }

      [Option("seed", Required = false, HelpText = "Random seed for shuffling.")]
      public int? Seed { get; set; }
    }

    [Verb("predict", HelpText = "Calculate predictions and intervals for a model and a dataset (includes prior fitting).")]
    public class PredictOptions : OptionsBase {
      [Option("no-optimization", Required = false, Default = false, HelpText = "Switch to skip parameter fitting.")]
      public bool NoOptimization { get; set; }

      [Option("interval", Required = false, Default = IntervalEnum.LaplaceApproximation, HelpText = "Prediction interval type.")]
      public IntervalEnum Interval { get; set; }

      [Option("range", Required = false, HelpText ="The range of index values for which the predictions should be calculated (default: whole dataset)")]
      public string? Range { get; set; }
    }

    [Verb("fit", HelpText = "Fit a model using a dataset.")]
    public class FitOptions : OptionsBase {
      [Option("maxIter", Required = false, HelpText = "The maximum number of Levenberg-Marquardt iterations.", Default = 10000)]
      public int MaxIterations { get; set; }
    }

    [Verb("evaluate", HelpText = "Evaluate a model on a dataset without fitting.")]
    public class EvalOptions {
      [Option('d', "dataset", Required = true, HelpText = "Filename with dataset in csv format.")]
      public string Dataset { get; set; }

      [Option('t', "target", Required = true, HelpText = "Target variable name.")]
      public string Target { get; set; }

      [Option('m', "model", Required = true, HelpText = "The model in infix form as produced by Operon.")]
      public string Model { get; set; }

      [Option('l', "likelihood", Required = false, HelpText = "The likelihood function for the model (Gaussian or Bernoulli) (default: Gaussian)", Default = LikelihoodEnum.Gaussian)]
      public LikelihoodEnum Likelihood { get; set; }

      [Option("range", Required = false, HelpText = "The range <firstRow>:<lastRow> in the dataset (inclusive).")]
      public string Range { get; set; }
      [Option("noiseSigma", Required = false, HelpText = "The standard deviation of noise in the target if it is known.")]
      public double? NoiseSigma { get; set; } // TODO: only for Gaussian likelihood and allow string value to refer to a variable in the dataset
    }

    [Verb("simplify", HelpText = "Remove redundant parameters.")]
    public class SimplifyOptions {
      [Option('m', "model", Required = true, HelpText = "The model in infix form as produced by Operon.")]
      public string Model { get; set; }

      [Option('v', "varNames", Required = true, HelpText = "Comma-separated list of variables.")]
      public string Variables { get; set; }
    }

    [Verb("nested", HelpText = "Analyse nested models.")]
    public class NestedModelsOptions {
      [Option('d', "dataset", Required = true, HelpText = "Filename with dataset in csv format.")]
      public string Dataset { get; set; }

      [Option('t', "target", Required = true, HelpText = "Target variable name.")]
      public string Target { get; set; }

      [Option('m', "model", Required = true, HelpText = "The model in infix form as produced by Operon.")]
      public string Model { get; set; }

      [Option('l', "likelihood", Required = false, HelpText = "The likelihood function for the model (Gaussian or Bernoulli) (default: Gaussian)", Default = LikelihoodEnum.Gaussian)]
      public LikelihoodEnum Likelihood { get; set; }

      [Option("train", Required = false, HelpText = "The training range <firstRow>:<lastRow> in the dataset (inclusive).")]
      public string TrainingRange { get; set; }

      [Option("verbose", Required = false, HelpText = "Produced more detailed output.")]
      public bool Verbose { get; set; }
      [Option("deltaAIC", Required = false, HelpText = "The maximum deltaAIC that is still accepted (e.g. 3.0)", Default = 3.0)]
      public double DeltaAIC { get; set; }

      [Option("maxIter", Required = false, HelpText = "The maximum number of Levenberg-Marquardt iterations.", Default = 10000)]
      public int MaxIterations { get; set; }

    }

    [Verb("prune", HelpText = "Prune model by removing parameters")]
    public class PruneOptions {
      [Option('d', "dataset", Required = true, HelpText = "Filename with dataset in csv format.")]
      public string Dataset { get; set; }

      [Option('t', "target", Required = true, HelpText = "Target variable name.")]
      public string Target { get; set; }

      [Option('m', "model", Required = true, HelpText = "The model in infix form as produced by Operon.")]
      public string Model { get; set; }

      [Option('l', "likelihood", Required = false, HelpText = "The likelihood function for the model (Gaussian or Bernoulli) (default: Gaussian)", Default = LikelihoodEnum.Gaussian)]
      public LikelihoodEnum Likelihood { get; set; }

      [Option("train", Required = false, HelpText = "The training range <firstRow>:<lastRow> in the dataset (inclusive).")]
      public string TrainingRange { get; set; }

      [Option("verbose", Required = false, HelpText = "Produced more detailed output.")]
      public bool Verbose { get; set; }
      [Option("deltaAIC", Required = false, HelpText = "The maximum deltaAIC that is still accepted (e.g. 3.0)", Default = 3.0)]
      public double DeltaAIC { get; set; }

      [Option("maxIter", Required = false, HelpText = "The maximum number of Levenberg-Marquardt iterations.", Default = 10000)]
      public int MaxIterations { get; set; }

    }

    [Verb("subtrees", HelpText = "Subtree importance")]
    public class SubtreeImportanceOptions {
      // TODO combine verbs nested and subtrees
      [Option('d', "dataset", Required = true, HelpText = "Filename with dataset in csv format.")]
      public string Dataset { get; set; }

      [Option('t', "target", Required = true, HelpText = "Target variable name.")]
      public string Target { get; set; }

      [Option('m', "model", Required = true, HelpText = "The model in infix form as produced by Operon.")]
      public string Model { get; set; }

      [Option('l', "likelihood", Required = false, HelpText = "The likelihood function for the model (Gaussian or Bernoulli) (default: Gaussian)", Default = LikelihoodEnum.Gaussian)]
      public LikelihoodEnum Likelihood { get; set; }

      [Option("train", Required = false, HelpText = "The training range <firstRow>:<lastRow> in the dataset (inclusive).")]
      public string TrainingRange { get; set; }

      [Option("graphviz", Required = false, HelpText = "File name for graphviz output.")]
      public string? GraphvizFilename { get; set; }

      [Option("hideParam", Required = false, Default = false, HelpText = "Switch to hide parameter values in the graphviz output.")]
      public bool HideParameters { get; set; }
      [Option("hideVar", Required = false, Default = false, HelpText = "Switch to hide the variable names in the graphviz output.")]
      public bool HideVariables { get; set; }
    }

    [Verb("crossvalidate", HelpText = "Cross-validation")]
    public class CrossValidationOptions {
      [Option('d', "dataset", Required = true, HelpText = "Filename with dataset in csv format.")]
      public string Dataset { get; set; }

      [Option('t', "target", Required = true, HelpText = "Target variable name.")]
      public string Target { get; set; }

      [Option('m', "model", Required = true, HelpText = "The model in infix form as produced by Operon.")]
      public string Model { get; set; }

      [Option('l', "likelihood", Required = false, HelpText = "The likelihood function for the model (Gaussian or Bernoulli) (default: Gaussian)", Default = LikelihoodEnum.Gaussian)]
      public LikelihoodEnum Likelihood { get; set; }

      [Option("train", Required = false, HelpText = "The range <firstRow>:<lastRow> in the dataset (inclusive) used for cross-validation.")]
      public string TrainingRange { get; set; }

      [Option("folds", Required = false, Default = 10, HelpText = "The number of folds")]
      public int Folds { get; set; }

      [Option("shuffle", Required = false, Default = false, HelpText = "Switch to shuffle the dataset before splitting into CV folds.")]
      public bool Shuffle { get; set; }

      [Option("seed", Required = false, HelpText = "Random seed for shuffling.")]
      public int? Seed { get; set; }

    }

    [Verb("variableimpact", HelpText = "Calculate variable impacts")]
    // TODO combine with subtree impacts (and potentially nested models)
    public class VariableImpactOptions {
      [Option('d', "dataset", Required = true, HelpText = "Filename with dataset in csv format.")]
      public string Dataset { get; set; }

      [Option('t', "target", Required = true, HelpText = "Target variable name.")]
      public string Target { get; set; }

      [Option('m', "model", Required = true, HelpText = "The model in infix form as produced by Operon.")]
      public string Model { get; set; }

      [Option('l', "likelihood", Required = false, HelpText = "The likelihood function for the model (Gaussian or Bernoulli) (default: Gaussian)", Default = LikelihoodEnum.Gaussian)]
      public LikelihoodEnum Likelihood { get; set; }

      [Option("train", Required = false, HelpText = "The range <firstRow>:<lastRow> in the dataset (inclusive) used for variable impact calculation.")]
      public string TrainingRange { get; set; }

      [Option("shuffle", Required = false, Default = false, HelpText = "Switch to shuffle the dataset before fitting.")]
      public bool Shuffle { get; set; }

      [Option("seed", Required = false, HelpText = "Random seed for shuffling.")]
      public int? Seed { get; set; }

    }

    [Verb("pairwise", HelpText = "Produce data for pairwise profile plots")]
    // TODO combine with subtree impacts (and potentially nested models)
    public class PairwiseProfileOptions {
      [Option('d', "dataset", Required = true, HelpText = "Filename with dataset in csv format.")]
      public string Dataset { get; set; }

      [Option('t', "target", Required = true, HelpText = "Target variable name.")]
      public string Target { get; set; }

      [Option('m', "model", Required = true, HelpText = "The model in infix form as produced by Operon.")]
      public string Model { get; set; }

      [Option('l', "likelihood", Required = false, HelpText = "The likelihood function for the model (Gaussian or Bernoulli) (default: Gaussian)", Default = LikelihoodEnum.Gaussian)]
      public LikelihoodEnum Likelihood { get; set; }

      [Option("train", Required = false, HelpText = "The range <firstRow>:<lastRow> in the dataset (inclusive) used for profile calculation.")]
      public string TrainingRange { get; set; }
    }

    [Verb("profile", HelpText = "Produce data for profile plots")]
    public class ProfileOptions {
      [Option('d', "dataset", Required = true, HelpText = "Filename with dataset in csv format.")]
      public string Dataset { get; set; }

      [Option('t', "target", Required = true, HelpText = "Target variable name.")]
      public string Target { get; set; }

      [Option('m', "model", Required = true, HelpText = "The model in infix form as produced by Operon.")]
      public string Model { get; set; }

      [Option('l', "likelihood", Required = false, HelpText = "The likelihood function for the model (Gaussian or Bernoulli) (default: Gaussian)", Default = LikelihoodEnum.Gaussian)]
      public LikelihoodEnum Likelihood { get; set; }

      [Option("train", Required = false, HelpText = "The range <firstRow>:<lastRow> in the dataset (inclusive) used for profile calculation.")]
      public string TrainingRange { get; set; }
    }

    [Verb("rank", HelpText = "Determine numeric rank of Jacobian matrix")]
    public class RankDeterminationOptions {
      [Option('d', "dataset", Required = true, HelpText = "Filename with dataset in csv format.")]
      public string Dataset { get; set; }

      [Option('t', "target", Required = true, HelpText = "Target variable name.")]
      public string Target { get; set; }

      [Option('m', "model", Required = true, HelpText = "The model in infix form as produced by Operon.")]
      public string Model { get; set; }

      [Option('l', "likelihood", Required = false, HelpText = "The likelihood function for the model (Gaussian or Bernoulli) (default: Gaussian)", Default = LikelihoodEnum.Gaussian)]
      public LikelihoodEnum Likelihood { get; set; }
      
      [Option("train", Required = false, HelpText = "The range <firstRow>:<lastRow> in the dataset (inclusive) used for profile calculation.")]
      public string TrainingRange { get; set; }

      [Option("no-optimization", Required = false, Default = false, HelpText = "Switch to skip nonlinear least squares fitting.")]
      public bool NoOptimization { get; set; }

    }

    #endregion

    #region helper


    private static void PrepareData(OptionsBase options, out string[] varNames, out double[,] x, out double[] y, out int trainStart, out int trainEnd, out int testStart, out int testEnd, out double[,] trainX, out double[] trainY) {
      ReadData(options.Dataset, options.Target, out varNames, out x, out y);

      // default split is 66/34%
      var m = x.GetLength(0);
      trainStart = 0;
      trainEnd = (int)Math.Round(m * 2 / 3.0);
      testStart = (int)Math.Round(m * 2 / 3.0) + 1;
      testEnd = m - 1;
      if (options.TrainingRange != null) {
        var toks = options.TrainingRange.Split(":");
        trainStart = int.Parse(toks[0]);
        trainEnd = int.Parse(toks[1]);
      }
      if (options.TestingRange != null) {
        var toks = options.TestingRange.Split(":");
        testStart = int.Parse(toks[0]);
        testEnd = int.Parse(toks[1]);
      }

      var randSeed = new Random().Next();
      if (options.Seed != null) {
        randSeed = options.Seed.Value;
      }

      if (options.Shuffle) {
        var rand = new Random(randSeed);
        Shuffle(x, y, rand);
      }

      Split(x, y, trainStart, trainEnd, testStart, testEnd, out trainX, out trainY, out var testX, out var testY);
    }

    // start and end are inclusive
    private static void Split(double[,] x, double[] y, int trainStart, int trainEnd, int testStart, int testEnd,
      out double[,] trainX, out double[] trainY,
      out double[,] testX, out double[] testY) {
      if (trainStart < 0) throw new ArgumentException("Negative index for training start");
      if (trainEnd >= y.Length) throw new ArgumentException($"End of training range: {trainEnd} but dataset has only {x.GetLength(0)} rows. Training range is inclusive.");
      if (testStart < 0) throw new ArgumentException("Negative index for training start");
      if (testEnd >= y.Length) throw new ArgumentException($"End of testing range: {testEnd} but dataset has only {x.GetLength(0)} rows. Testing range is inclusive.");

      var dim = x.GetLength(1);
      var trainRows = trainEnd - trainStart + 1;
      var testRows = testEnd - testStart + 1;
      trainX = new double[trainRows, dim]; trainY = new double[trainRows];
      testX = new double[testRows, dim]; testY = new double[testRows];
      Buffer.BlockCopy(x, trainStart * dim * sizeof(double), trainX, 0, trainRows * dim * sizeof(double));
      Array.Copy(y, trainStart, trainY, 0, trainRows);
      Buffer.BlockCopy(x, testStart * dim * sizeof(double), testX, 0, testRows * dim * sizeof(double));
      Array.Copy(y, testStart, testY, 0, testRows);
    }


    private static void DeletePartition(double[,] X, double[] y, int start, int end, out double[,] reducedX, out double[] reducedY) {
      var removedRows = (end - start + 1);
      var d = X.GetLength(1);
      var m = X.GetLength(0);
      reducedX = new double[X.GetLength(0) - removedRows, d];
      Buffer.BlockCopy(X, 0, reducedX, 0, start * d * sizeof(double));
      Buffer.BlockCopy(X, (end + 1) * d * sizeof(double), reducedX, start * d * sizeof(double), (m - end - 1) * d * sizeof(double));


      reducedY = new double[y.Length - removedRows];
      Array.Copy(y, 0, reducedY, 0, start);
      Array.Copy(y, end + 1, reducedY, start, m - end - 1);
    }

    private static void Shuffle(double[,] x, double[] y, Random rand) {
      var n = y.Length;
      var d = x.GetLength(1);
      // via using sorting
      var idx = Enumerable.Range(0, n).ToArray();
      var r = idx.Select(_ => rand.NextDouble()).ToArray();
      Array.Sort(r, idx);

      var shufX = new double[n, d];
      var shufY = new double[y.Length];
      for (int i = 0; i < n; i++) {
        shufY[idx[i]] = y[i];
        Buffer.BlockCopy(y, i * sizeof(double) * d, shufX, idx[i] * sizeof(double) * d, sizeof(double) * d); // copy a row shufX[idx[i], :] = x[i, :]
      }

      // overwrite x,y with shuffled data
      Array.Copy(shufY, y, y.Length);
      Buffer.BlockCopy(shufX, 0, shufY, 0, shufY.Length * sizeof(double));
    }


    public static void GenerateExpression(string model, string[] varNames, out Expression<Expr.ParametricFunction> parametricExpr, out double[] p) {
      var varValues = Expression.Parameter(typeof(double[]), "x");
      var paramValues = Expression.Parameter(typeof(double[]), "p");
      var parser = new ExprParser(model, varNames, varValues, paramValues);
      parametricExpr = parser.Parse();
      p = parser.ParameterValues;
    }

    public static void ReadData(string filename, string targetVariable, out string[] variableNames, out double[,] x, out double[] y) {
      using (var reader = new StreamReader(filename)) {
        var allVarNames = reader.ReadLine().Split(',');
        var yIdx = Array.IndexOf(allVarNames, targetVariable);
        if (yIdx < 0) throw new FormatException($"Variable {targetVariable} not found in {filename}");

        // keep only variablesNames \ { targetVariable } in x and variableNames

        variableNames = new string[allVarNames.Length - 1];
        Array.Copy(allVarNames, variableNames, yIdx);
        Array.Copy(allVarNames, yIdx + 1, variableNames, yIdx, allVarNames.Length - (yIdx + 1));

        var dim = variableNames.Length;
        var rows = new List<double[]>(); var yRows = new List<double>();
        var lineNr = 1;
        while (!reader.EndOfStream) {
          var strValues = reader.ReadLine().Split(',');
          lineNr++;
          if (strValues.Length != allVarNames.Length) throw new FormatException($"The number of columns does not match in line {lineNr}");
          var row = new double[dim];
          var varIdx = 0;
          var colIdx = 0;
          while (colIdx < strValues.Length) {
            if (colIdx == yIdx) {
              yRows.Add(double.Parse(strValues[colIdx++]));
            } else {
              row[varIdx++] = double.Parse(strValues[colIdx++]);
            }
          }
          rows.Add(row);
        }

        // copy all rows to matrix x
        x = new double[rows.Count, dim];
        for (int i = 0; i < rows.Count; i++) {
          Buffer.BlockCopy(rows[i], 0, x, i * dim * sizeof(double), dim * sizeof(double));
        }
        y = yRows.ToArray();
      }
    }
    #endregion
  }
}
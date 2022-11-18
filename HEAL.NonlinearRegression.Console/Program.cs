﻿using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.Tracing;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Net.WebSockets;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using CommandLine;
using HEAL.Expressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Scripting;
using Microsoft.CodeAnalysis.Scripting;
using static alglib;

// TODO:
//  'verbs' for different usages shown in Demo project (already implemented):
//   - subtree importance and graphviz output
//   - variable impacts
//   - Generate comparison outputs for Puromycin and BOD for linear prediction intervals and compare to book (use Gnuplot)
//   - Generate comparison outputs for pairwise profile plots and compare to book. (use Gnuplot)
// 
//  more ideas (not yet implemented)
//   - alglib is GPL, should switch to .NET numerics (MIT) instead.
//   - iterative pruning based on subtree impacts or likelihood ratios for nested models
//   - variable impacts for combinations of variables (tuples, triples). Contributions to individual variables via Shapely values?
//   - nested model analysis for combinations of parameters (for the case where a parameter can only be set to zero if another parameter is also set to zero)
//   - If a range is specified (training, test) then only read the relevant rows of data
//   - execute verbs for multiple models from file

namespace HEAL.NonlinearRegression.Console {
  // Takes a dataset, target variable, and a model from the command line and runs NLR and calculates all statistics.
  // Range for training set can be specified optionally.
  // Intended to be used together with Operon.
  // Use the suffix 'f' to mark real literals in the model as fixed instead of a parameter.
  // e.g. 10 * x^2f ,  2 is a fixed constant, 10 is a parameter of the model
  public class Program {

    public static void Main(string[] args) {
      System.Globalization.CultureInfo.DefaultThreadCurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
      var parserResult = Parser.Default.ParseArguments<PredictOptions, FitOptions, SimplifyOptions, NestedModelsOptions,
        SubtreeImportanceOptions, CrossValidationOptions, VariableImpactOptions, EvalOptions, PairwiseProfileOptions, ProfileOptions,
        RankDeterminationOptions>(args)
        .WithParsed<PredictOptions>(options => Predict(options))
        .WithParsed<FitOptions>(options => Fit(options))
        .WithParsed<EvalOptions>(options => Evaluate(options))
        .WithParsed<SimplifyOptions>(options => Simplify(options))
        .WithParsed<NestedModelsOptions>(options => GenerateNestedModels(options))
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

      var modelExpression = PreprocessModelString(options.Model, varNames, out var constants);
      // System.Console.WriteLine(modelExpression);

      var parametricExpr = GenerateExpression(modelExpression, constants, out var p);

      var nls = new NonlinearRegression();
      nls.Fit(p, parametricExpr, trainX, trainY, maxIterations: 200);

      if (nls.OptReport.Success) {
        System.Console.WriteLine($"p_opt: {string.Join(" ", p.Select(pi => pi.ToString("e5")))}");
        System.Console.WriteLine($"{nls.OptReport}");
        nls.WriteStatistics();
        System.Console.WriteLine($"Optimized: {Expr.ToString(parametricExpr, varNames, p)}");
      } else {
        System.Console.WriteLine("There was a problem while fitting.");
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

      var modelExpression = PreprocessModelString(options.Model, varNames, out var constants);
      // System.Console.WriteLine(modelExpression);

      var parametricExpr = GenerateExpression(modelExpression, constants, out var p);
      // System.Console.WriteLine(parametricExpr);

      var SSR = EvaluateSSR(parametricExpr, p, x, y, out var yPred);
      var nmse = SSR / y.Length / Util.Variance(y);

      var _jac = Expr.Jacobian(parametricExpr, p.Length).Compile();
      void jac(double[] p, double[,] X, double[] f, double[,] jac) => _jac(p, X, f, jac);
      var stats = new LeastSquaresStatistics(y.Length, p.Length, SSR, yPred, p, jac, x);

      System.Console.WriteLine($"SSR: {SSR} MSE: {SSR / y.Length} RMSE: {Math.Sqrt(SSR / y.Length)} NMSE: {nmse} R2: {1 - nmse} LogLik: {stats.LogLikelihood} AICc: {stats.AICc} BIC: {stats.BIC} DoF: {p.Length}");
    }

    private static double EvaluateSSR(Expression<Expr.ParametricFunction> parametricExpr, double[] p, double[,] x, double[] y, out double[] yPred) {
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

      var modelExpression = PreprocessModelString(options.Model, varNames, out var constants);
      //System.Console.WriteLine(modelExpression);

      var parametricExpr = GenerateExpression(modelExpression, constants, out var p);
      // System.Console.WriteLine(parametricExpr);

      var nlr = new NonlinearRegression();
      if (options.NoOptimization) {
        nlr.SetModel(p, parametricExpr, trainX, trainY);
      } else {
        nlr.Fit(p, parametricExpr, trainX, trainY);
      }

      var predict = nlr.PredictWithIntervals(x, options.Interval, includeNoise: true); // TODO includeNoise as CLI option

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
        if (options.Interval == IntervalEnum.LinearApproximation)
          System.Console.Write($"{predict[i, 2]},{predict[i, 3]},");
        else if (options.Interval == IntervalEnum.TProfile)
          System.Console.Write($"{predict[i, 1]},{predict[i, 2]},");
        System.Console.Write($"{((i >= trainStart && i <= trainEnd) ? 1 : 0)},"); // isTrain
        System.Console.Write($"{((i >= testStart && i <= testEnd) ? 1 : 0)}"); // isTest
        System.Console.WriteLine();
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

      var modelExpression = PreprocessModelString(options.Model, varNames, out var constants);
      //System.Console.WriteLine(modelExpression);

      var parametricExpr = GenerateExpression(modelExpression, constants, out var p);

      try {
        var cvrmse = CrossValidate(parametricExpr, p, trainX, trainY, shuffle: options.Shuffle, seed: options.Seed);
        System.Console.WriteLine($"CV RMSE mean: {cvrmse.Average():e4} stddev: {Math.Sqrt(Util.Variance(cvrmse.ToArray())):e4}");
      } catch (Exception e) {
        System.Console.WriteLine($"Error in fitting");
      }

    }

    private static List<double> CrossValidate(Expression<Expr.ParametricFunction> parametricExpr, double[] p, double[,] X, double[] y, int folds = 10, bool shuffle = false, int? seed = null) {
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
      var rmse = new List<double>();
      Parallel.For(0, folds, (f) => {
        var foldStart = f * foldSize;
        var foldEnd = (f + 1) * foldSize - 1;
        if (f == folds - 1) {
          foldEnd = y.Length - 1; // include remaining part in last fold
        }

        DeletePartition(X, y, foldStart, foldEnd, out var foldTrainX, out var foldTrainY);
        Split(X, y, foldStart, foldEnd, foldStart, foldEnd, out var foldTestX, out var foldTestY, out _, out _);

        var nls = new NonlinearRegression();
        nls.Fit(p, parametricExpr, foldTrainX, foldTrainY, maxIterations: 5000); // TODO make CLI parameter

        var foldPred = nls.Predict(foldTestX);
        var SSRtest = 0.0;
        for (int i = 0; i < foldTestY.Length; i++) {
          var r = foldTestY[i] - foldPred[i];
          SSRtest += r * r;
        }
        lock (rmse) {
          rmse.Add(Math.Sqrt(SSRtest / foldTestY.Length));
        }
      });
      return rmse;
    }

    private static void Simplify(SimplifyOptions options) {
      var varNames = options.Variables.Split(',').Select(vn => vn.Trim()).ToArray();

      var modelExpression = PreprocessModelString(options.Model, varNames, out var constants);
      var parametricExpr = GenerateExpression(modelExpression, constants, out var p);
      Expression<Expr.ParametricFunction> simplifiedExpr;

      Simplify(parametricExpr, p, varNames, out simplifiedExpr, out var newP);

      System.Console.WriteLine(simplifiedExpr);
      System.Console.WriteLine($"theta: {string.Join(",", newP.Select(pi => pi.ToString()))}");

      System.Console.WriteLine(Expr.ToString(simplifiedExpr, varNames, newP));
    }

    private static void Simplify(Expression<Expr.ParametricFunction> parametricExpr, double[] p, string[] varNames, out Expression<Expr.ParametricFunction> simplifiedExpr, out double[] newP) {
      simplifiedExpr = Expr.FoldParameters(parametricExpr, p, out newP);
      var newSimplifiedStr = simplifiedExpr.ToString();
      var exprSet = new HashSet<string>();
      // simplify until no change (TODO: this shouldn't be necessary if a visitors are implemented carefully)
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

      var modelExpression = PreprocessModelString(options.Model, varNames, out var constants);
      var parametricExpr = GenerateExpression(modelExpression, constants, out var p);

      // calculate ref stats for full model
      var nlr = new NonlinearRegression();
      nlr.Fit(p, parametricExpr, trainX, trainY);
      var refStats = nlr.Statistics;

      var allModels = new List<Tuple<Expression<Expr.ParametricFunction>, double[]>>();
      var modelQueue = new Queue<Tuple<Expression<Expr.ParametricFunction>, double[]>>();
      modelQueue.Enqueue(Tuple.Create(parametricExpr, (double[])p.Clone()));



      while (modelQueue.Any()) {

        (parametricExpr, p) = modelQueue.Dequeue();
        allModels.Add(Tuple.Create(parametricExpr, p));
        var impacts = ModelAnalysis.NestedModelLiklihoodRatios(parametricExpr, trainX, trainY, (double[])p.Clone(), options.Verbose);

        // order by SSRfactor and use the best as an alternative (if it has delta AICc < 5.0)
        var alternative = impacts.OrderBy(tup => tup.Item2).Where(tup => tup.Item3 < 5.0).FirstOrDefault();
        if (alternative != null) {
          var ssrFactor = alternative.Item2;
          var deltaAICc = alternative.Item3;
          var reducedExpr = alternative.Item4;
          var reducedParam = alternative.Item5;
          System.Console.Error.WriteLine($"New model {reducedParam.Length} {ssrFactor:e4} {deltaAICc:e4} {reducedExpr}");
          modelQueue.Enqueue(Tuple.Create(reducedExpr, reducedParam));
        }

        // foreach (var tup in impacts) {
        //   var paramIdx = tup.Item1;
        //   var ssrFactor = tup.Item2;
        //   var deltaAICc = tup.Item3;
        //   var reducedExpr = tup.Item4;
        //   var reducedParam = tup.Item5;
        //   // TODO make CLI parameter
        //   if (deltaAICc < 5.0) {
        //     // System.Console.Write(".");
        //     System.Console.WriteLine($"New model {reducedParam.Length} {ssrFactor:e4} {deltaAICc:e4} {reducedExpr.ToString()}");
        //     if (modelQueue.All(tup => tup.Item1.ToString() != reducedExpr.ToString()))
        //       modelQueue.Enqueue(Tuple.Create(reducedExpr, reducedParam));
        //   }
        // }
      }
      System.Console.WriteLine();
      System.Console.WriteLine($"SSR_Factor\tnumPar\tRMSE_tr\tRMSE_te\tRMSE_cv\tRMSE_cv_std\tAICc\tdAICc\tBIC\tdBIC\tModel");
      foreach (var model in allModels) {
        (parametricExpr, p) = model;

        // output training, CV, and test result for original model
        nlr = new NonlinearRegression();
        nlr.SetModel(p, parametricExpr, trainX, trainY);
        var stats = nlr.Statistics;
        var ssrTrain = stats.SSR;
        var rmseTrain = Math.Sqrt(ssrTrain / trainY.Length);
        var ssrTest = EvaluateSSR(parametricExpr, p, testX, testY, out _);
        var rmseTest = Math.Sqrt(ssrTest / testY.Length);
        var cvrmseMean = double.NaN;
        var cvrmseStd = double.NaN;
        try {
          var cvrmse = CrossValidate(parametricExpr, p, trainX, trainY, folds: 3);
          cvrmseMean = cvrmse.Average();
          cvrmseStd = Math.Sqrt(Util.Variance(cvrmse.ToArray()));
        } catch (Exception) { }
        System.Console.WriteLine($"{stats.SSR / refStats.SSR:e4}\t{stats.n}\t{rmseTrain:e4}\t{rmseTest:e4}\t{cvrmseMean:e4}\t{cvrmseStd:e4}\t{stats.AICc:e4}\t{stats.AICc - refStats.AICc:e4}\t{stats.BIC:e4}\t{stats.BIC - refStats.BIC:e4}\t{Expr.ToString(parametricExpr, varNames, p)}");
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

      var modelExpression = PreprocessModelString(options.Model, varNames, out var constants);
      //System.Console.WriteLine(modelExpression);

      var parametricExpr = GenerateExpression(modelExpression, constants, out var p);
      // System.Console.WriteLine(parametricExpr);

      var subExprImportance = ModelAnalysis.SubtreeImportance(parametricExpr, trainX, trainY, p);

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
        using (var writer = new System.IO.StreamWriter(options.GraphvizFilename)) {
          writer.WriteLine(Expr.ToGraphViz(parametricExpr,
            paramValues: options.HideParameters ? null : p,
            varNames: options.HideVariables ? null : varNames,
            saturation));
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

      var modelExpression = PreprocessModelString(options.Model, varNames, out var constants);
      //System.Console.WriteLine(modelExpression);

      var parametricExpr = GenerateExpression(modelExpression, constants, out var p);
      // System.Console.WriteLine(parametricExpr);

      var varImportance = ModelAnalysis.VariableImportance(parametricExpr, trainX, trainY, p);


      System.Console.WriteLine($"{"variable",-11} {"VarExpl",-11}");
      foreach (var tup in varImportance.OrderByDescending(tup => tup.Value)) { // TODO better interface
        var varName = varNames[tup.Key];
        System.Console.WriteLine($"{varName,-11} {tup.Value * 100,-11:f2}%");
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

      var modelExpression = PreprocessModelString(options.Model, varNames, out var constants);
      //System.Console.WriteLine(modelExpression);

      var parametricExpr = GenerateExpression(modelExpression, constants, out var parameters);
      // System.Console.WriteLine(parametricExpr);

      var nlr = new NonlinearRegression();
      nlr.Fit(parameters, parametricExpr, trainX, trainY);

      var _func = Expr.Broadcast(parametricExpr).Compile();
      var _jac = Expr.Jacobian(parametricExpr, parameters.Length).Compile();
      void func(double[] p, double[,] x, double[] f) => _func(p, x, f);
      void jac(double[] p, double[,] x, double[] f, double[,] j) => _jac(p, x, f, j);

      var folder = Path.GetDirectoryName(options.Dataset);
      var filename = Path.GetFileNameWithoutExtension(options.Dataset);

      var tProfile = new TProfile(trainY, trainX, nlr.Statistics, func, jac);
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

      var modelExpression = PreprocessModelString(options.Model, varNames, out var constants);
      //System.Console.WriteLine(modelExpression);

      var parametricExpr = GenerateExpression(modelExpression, constants, out var parameters);
      // System.Console.WriteLine(parametricExpr);

      var nlr = new NonlinearRegression();
      nlr.Fit(parameters, parametricExpr, trainX, trainY);

      var _func = Expr.Broadcast(parametricExpr).Compile();
      var _jac = Expr.Jacobian(parametricExpr, parameters.Length).Compile();
      void func(double[] p, double[,] x, double[] f) => _func(p, x, f);
      void jac(double[] p, double[,] x, double[] f, double[,] j) => _jac(p, x, f, j);

      var folder = Path.GetDirectoryName(options.Dataset);
      var filename = Path.GetFileNameWithoutExtension(options.Dataset);

      var tProfile = new TProfile(trainY, trainX, nlr.Statistics, func, jac);

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

      var modelExpression = PreprocessModelString(options.Model, varNames, out var constants);
      //System.Console.WriteLine(modelExpression);

      var parametricExpr = GenerateExpression(modelExpression, constants, out var parameters);
      // System.Console.WriteLine(parametricExpr);

      if (!options.NoOptimization) {
        var nlr = new NonlinearRegression();
        nlr.Fit(parameters, parametricExpr, trainX, trainY, maxIterations: 3000);
      }

      var _func = Expr.Broadcast(parametricExpr).Compile();
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
        var tol = n * eps;
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
    #endregion

    #region  options
    public abstract class OptionsBase {
      [Option('d', "dataset", Required = true, HelpText = "Filename with dataset in csv format.")]
      public string Dataset { get; set; }

      [Option('t', "target", Required = true, HelpText = "Target variable name.")]
      public string Target { get; set; }

      [Option('m', "model", Required = true, HelpText = "The model in infix form as produced by Operon.")]
      public string Model { get; set; }

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
      [Option("no-optimization", Required = false, Default = false, HelpText = "Switch to skip nonlinear least squares fitting.")]
      public bool NoOptimization { get; set; }

      [Option("interval", Required = false, Default = IntervalEnum.LinearApproximation, HelpText = "Prediction interval type.")]
      public IntervalEnum Interval { get; set; }
    }

    [Verb("fit", HelpText = "Fit a model using a dataset.")]
    public class FitOptions : OptionsBase {
    }

    [Verb("evaluate", HelpText = "Evaluate a model on a dataset without fitting.")]
    public class EvalOptions {
      [Option('d', "dataset", Required = true, HelpText = "Filename with dataset in csv format.")]
      public string Dataset { get; set; }

      [Option('t', "target", Required = true, HelpText = "Target variable name.")]
      public string Target { get; set; }

      [Option('m', "model", Required = true, HelpText = "The model in infix form as produced by Operon.")]
      public string Model { get; set; }

      [Option("range", Required = false, HelpText = "The range <firstRow>:<lastRow> in the dataset (inclusive).")]
      public string Range { get; set; }
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

      [Option("train", Required = false, HelpText = "The training range <firstRow>:<lastRow> in the dataset (inclusive).")]
      public string TrainingRange { get; set; }

      [Option("verbose", Required = false, HelpText = "Produced more detailed output.")]
      public bool Verbose { get; set; }
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

      var randSeed = new System.Random().Next();
      if (options.Seed != null) {
        randSeed = options.Seed.Value;
      }

      if (options.Shuffle) {
        var rand = new System.Random(randSeed);
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
      throw new NotImplementedException();
    }

    private static Expression<Expr.ParametricFunction> GenerateExpression(string modelExpression, double[] constants, out double[] p) {
      var options = ScriptOptions.Default
        .AddReferences(typeof(Expression).Assembly)
        .AddReferences(typeof(HEAL.Expressions.Expr).Assembly) // for Functions class
        .AddImports("System")
        .AddImports("HEAL.Expressions") // for Functions class
        .WithEmitDebugInformation(false)
        .WithOptimizationLevel(Microsoft.CodeAnalysis.OptimizationLevel.Release);

      // use Microsoft.CodeAnalysis.CSharp.Scripting to compile the model string
      var expr = CSharpScript.EvaluateAsync<Expression<Func<double[], double[], double>>>(modelExpression, options).Result;
      var newExpr = Expr.ReplaceNumbersWithParameter(expr, out p);
      var constantsParameter = expr.Parameters.First(p => p.Name == "constants");
      return Expr.ReplaceParameterWithValues<Expr.ParametricFunction>(newExpr, constantsParameter, constants);
    }

    private static string PreprocessModelString(string model, string[] varNames, out double[] constants) {
      model = TranslateFunctionCalls(model);
      model = ReparameterizeModel(model, varNames); // replaces variables names with references to x[i] 
      model = ReplaceFloatLiteralsWithParameter(model, out constants); // replaces and float literals (e.g. 1.0f) with constants.

      // System.Console.WriteLine(model);
      var modelExpression = "(double[] x, double[] constants) => " + model;
      return modelExpression;
    }

    private static Regex logRegex = new Regex(@"([^a-zA-Z.])log\(");
    private static Regex plogRegex = new Regex(@"([^a-zA-Z.])plog\(");

    private static string TranslateFunctionCalls(string model) {

      model = model.Replace("pow(", "Math.Pow(", StringComparison.InvariantCultureIgnoreCase);
      model = TranslatePower(TranslateSqr(TranslateCube(model)))
        .Replace("abs(", "Math.Abs(", StringComparison.InvariantCultureIgnoreCase)
        .Replace("exp(", "Math.Exp(", StringComparison.InvariantCultureIgnoreCase)
        .Replace("sin(", "Math.Sin(", StringComparison.InvariantCultureIgnoreCase)
        .Replace("cos(", "Math.Cos(", StringComparison.InvariantCultureIgnoreCase)
        .Replace("tan(", "Math.Tan(", StringComparison.InvariantCultureIgnoreCase)
        .Replace("sqrt(", "Math.Sqrt(", StringComparison.InvariantCultureIgnoreCase)
        .Replace("tanh(", "Math.Tanh(", StringComparison.InvariantCultureIgnoreCase)
        .Replace("sinh(", "Math.Sinh(", StringComparison.InvariantCultureIgnoreCase)
        .Replace("cosh(", "Math.Cosh(", StringComparison.InvariantCultureIgnoreCase)
        .Replace("asin(", "Math.Asin(", StringComparison.InvariantCultureIgnoreCase)
        .Replace("acos(", "Math.Acos(", StringComparison.InvariantCultureIgnoreCase)
        .Replace("atan(", "Math.Atan(", StringComparison.InvariantCultureIgnoreCase)
        .Replace("asinh(", "Math.Asinh(", StringComparison.InvariantCultureIgnoreCase)
        .Replace("acosh(", "Math.Acosh(", StringComparison.InvariantCultureIgnoreCase)
        .Replace("atanh(", "Math.Atanh(", StringComparison.InvariantCultureIgnoreCase)
        .Replace("cbrt(", "Math.Cbrt(", StringComparison.InvariantCultureIgnoreCase)
        ;
      model = plogRegex.Replace(model, @"$1Functions.plog(");
      model = logRegex.Replace(model, @"$1Math.Log(");
      return model;
    }

    private static string TranslatePower(string model) {
      // https://stackoverflow.com/questions/19596502/regex-nested-parentheses
      string oldModel;
      do {
        oldModel = model;
        model = Regex.Replace(model, @"(\w*\((?>\((?<DEPTH>)|\)(?<-DEPTH>)|[^()]+)*\)(?(DEPTH)(?!)))\s*\^\s*([^-+*/) ]+)", "Math.Pow($1, $2)"); // only supports integer powers
      } while (model != oldModel);
      return model;
    }
    private static string TranslateSqr(string model) {
      string oldModel;
      do {
        oldModel = model;
        // https://stackoverflow.com/questions/19596502/regex-nested-parentheses
        model = Regex.Replace(model, @"sqr(\((?>\((?<DEPTH>)|\)(?<-DEPTH>)|[^()]+)*\)(?(DEPTH)(?!)))", "Math.Pow($1, 2f)", RegexOptions.IgnoreCase);
      } while (model != oldModel);
      return model;
    }
    private static string TranslateCube(string model) {
      string oldModel;
      do {
        oldModel = model;
        // https://stackoverflow.com/questions/19596502/regex-nested-parentheses
        model = Regex.Replace(model, @"cube(\((?>\((?<DEPTH>)|\)(?<-DEPTH>)|[^()]+)*\)(?(DEPTH)(?!)))", "Math.Pow($1, 3f)", RegexOptions.IgnoreCase);
      } while (model != oldModel);
      return model;
    }


    private static string ReparameterizeModel(string model, string[] varNames) {
      model = " " + model + " "; // so that we don't need special regex for matching at start and end of expression
      // TODO handle the case when the variable names are x or x[1] or similar
      for (int i = 0; i < varNames.Length; i++) {
        // We have to be careful to only replace variable names and keep function calls unchanged.
        // A variable must be followed by an operator (+,-,*,/),',', ' ', or ')'.
        var varRegex = new Regex("([^a-zA-Z])(" + varNames[i] + @")([ \+\-\*\/\),])");
        // we do this in a loop because of potential overlapping matches that might be missed
        string origModel;
        do {
          origModel = model;
          model = varRegex.Replace(model, $"$1x[{i}]$3");
        } while (origModel != model);
      }

      // System.Console.WriteLine(model);
      return model;
    }



    /* From C# lexical specification:
     *  Real_Literal
            : Decimal_Digit Decimal_Digit* '.'
              Decimal_Digit Decimal_Digit* Exponent_Part? Real_Type_Suffix?
            | '.' Decimal_Digit Decimal_Digit* Exponent_Part? Real_Type_Suffix?
            | Decimal_Digit Decimal_Digit* Exponent_Part Real_Type_Suffix?
            | Decimal_Digit Decimal_Digit* Real_Type_Suffix
            ;

        fragment Exponent_Part
            : ('e' | 'E') Sign? Decimal_Digit Decimal_Digit*
            ;

        fragment Sign
            : '+' | '-'
            ;

        fragment Real_Type_Suffix
            : 'F' | 'f' | 'D' | 'd' | 'M' | 'm'
            ;
     */
    private static string ReplaceFloatLiteralsWithParameter(string model, out double[] constants) {
      var expPart = $"(('e'|'E')[-+]?[0-9][0-9]*)";
      var floatLit = $"[0-9][0-9]*\\.[0-9]*{expPart}?f|\\.[0-9][0-9]*{expPart}?f|[0-9][0-9]*{expPart}f|[0-9][0-9]*f";
      var floatLitRegex = new Regex(floatLit);
      var newModel = model;
      int idx = 0;
      var constList = new List<double>();
      int startat = 0;
      do {
        var replacement = $"constants[{idx++}]";
        model = newModel;

        var match = floatLitRegex.Match(model, startat);
        if (!match.Success) {
          break;
        }
        constList.Add(float.Parse(match.Value.TrimEnd('f')));
        newModel = floatLitRegex.Replace(model, replacement, 1, match.Index);
        startat = match.Index + replacement.Length;
      } while (newModel != model);

      constants = constList.ToArray();
      return model;
    }


    private static void ReadData(string filename, string targetVariable, out string[] variableNames, out double[,] x, out double[] y) {
      using (var reader = new System.IO.StreamReader(filename)) {
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
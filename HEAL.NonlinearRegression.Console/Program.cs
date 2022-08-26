using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using CommandLine;
using HEAL.Expressions;
using Microsoft.CodeAnalysis.CSharp.Scripting;
using Microsoft.CodeAnalysis.Scripting;

// TODO:
//  'verbs' for different usages shown in Demo project (already implemented):
//   - prediction with linear-approximation and profile-based intervals (current functionality)
//   - fitting and parameter statistics
//   - subtree importance and graphviz output
//   - variable impacts
//   - nested model analysis and likelihood ratios
//   - Generate comparison outputs for Puromycin and BOD for linear prediction intervals and compare to book (use Gnuplot)
//   - Generate comparison outputs for pairwise profile plots and compare to book. (use Gnuplot)
// 
//  more ideas (not yet implemented)
//   - iterative pruning based on subtree impacts or likelihood ratios for nested models
//   - variable impacts for combinations of variables (tuples, triples). Contributions to individual variables via Shapely values?
//   - nested model analysis for combinations of parameters (for the case where a parameter can only be set to zero if another parameter is also set to zero)

namespace HEAL.NonlinearRegression.Console {
  // Takes a dataset, target variable, and a model from the command line and runs NLR and calculates all statistics.
  // Range for training set can be specified optionally.
  // Intended to be used together with Operon.
  // Use the suffix 'f' to mark real literals in the model as fixed instead of a parameter.
  // e.g. 10 * x^2f ,  2 is a fixed constant, 10 is a parameter of the model
  public class Program {
    // TODO general options for multiple verbs


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
      [Option("no-optimization", Required = false, Default=false, HelpText = "Switch to skip nonlinear least squares fitting.")]
      public bool NoOptimization { get; set; }
    }

    [Verb("fit", HelpText = "Fit a model using a dataset.")]
    public class FitOptions : OptionsBase {
    }



    public static void Main(string[] args) {
      var parserResult = Parser.Default.ParseArguments<PredictOptions,FitOptions>(args)
        .WithParsed<PredictOptions>(options => Predict(options))
        .WithParsed<FitOptions>(options => Fit(options));
    }

    private static void Fit(FitOptions options) {
      PrepareData(options, out var varNames, out var x, out var y, out var trainStart, out var trainEnd, out var testStart, out var testEnd, out var trainX, out var trainY);

      var modelExpression = PreprocessModelString(options.Model, varNames, out var constants);
      //System.Console.WriteLine(modelExpression);

      var parametricExpr = GenerateExpression(modelExpression, constants, out var p);
      // System.Console.WriteLine(parametricExpr);

      var nls = new NonlinearRegression();
      nls.Fit(p, parametricExpr, trainX, trainY);

      if (nls.OptReport.Success) {
        System.Console.WriteLine($"p_opt: {string.Join(" ", p.Select(pi => pi.ToString("e5")))}");
        System.Console.WriteLine($"{nls.OptReport}");
        nls.WriteStatistics();
      } else {
        System.Console.WriteLine("There was a problem while fitting.");
      }
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

      var predictProfile = nlr.PredictWithIntervals(x, IntervalEnum.TProfile, includeNoise: true);
      var predictApprox = nlr.PredictWithIntervals(x, IntervalEnum.LinearApproximation, includeNoise: true);

      // generate output for full dataset
      System.Console.WriteLine($"{string.Join(",", varNames)},y,residual,yPred,resStdErrorLinear,yPredLowLinear,yPredHighLinear,yPredLowProfile,yPredHighProfile,isTrain,isTest"); // header
      for (int i = 0; i < x.GetLength(0); i++) {
        for (int j = 0; j < x.GetLength(1); j++) {
          System.Console.Write($"{x[i, j]},");
        }
        System.Console.Write($"{y[i]},");
        System.Console.Write($"{y[i] - predictApprox[i, 0]},");
        System.Console.Write($"{predictApprox[i, 0]},{predictApprox[i, 1]},{predictApprox[i, 2]},{predictApprox[i, 3]},");
        System.Console.Write($"{predictProfile[i, 1]},{predictProfile[i, 2]},"); // mean prediction for approx and profile is the same
        System.Console.Write($"{((i >= trainStart && i <= trainEnd) ? 1 : 0)},"); // isTrain
        System.Console.Write($"{((i >= testStart && i <= testEnd) ? 1 : 0)}"); // isTest
        System.Console.WriteLine();
      }
    }

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

    private static void Shuffle(double[,] x, double[] y, Random rand) {
      throw new NotImplementedException();
    }

    private static Expression<Expr.ParametricFunction> GenerateExpression(string modelExpression, double[] constants, out double[] p) {
      var options = ScriptOptions.Default
        .AddReferences(typeof(Expression).Assembly)
        .AddImports("System")
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

      var modelExpression = "(double[] x, double[] constants) => " + model;
      return modelExpression;
    }

    private static string TranslateFunctionCalls(string model) {
      return TranslatePower(model)
        .Replace("log(", "Math.Log(", StringComparison.InvariantCultureIgnoreCase)
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
    }

    public static string TranslatePower(string model) {
      // https://stackoverflow.com/questions/19596502/regex-nested-parentheses
      return Regex.Replace(model, @"(\((?>\((?<DEPTH>)|\)(?<-DEPTH>)|[^()]+)*\)(?(DEPTH)(?!)))\s*\^\s*([^-+*/) ]+)", "Math.Pow($1, $2)"); // only supports integer powers
    }


    private static string ReparameterizeModel(string model, string[] varNames) {
      for (int i = 0; i < varNames.Length; i++) {
        // We have to be careful to only replace variable names and keep function calls unchanged.
        // A variable must be followed by an operator (+,-,*,/), ' ', or ')' or end-of-string .
        var varRegex = new System.Text.RegularExpressions.Regex("(" + varNames[i] + @")([ +\-*/\)])");
        var varEolRegex = new System.Text.RegularExpressions.Regex("(" + varNames[i] + @")\z"); // variable at end of line
        model = varRegex.Replace(model, $"x[{i}]$2");
        model = varEolRegex.Replace(model, $"x[{i}]");
      }

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
  }
}
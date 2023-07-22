using BenchmarkDotNet.Attributes;
using System.Linq.Expressions;

namespace HEAL.Expressions.Benchmark {
  // benchmark JacobianEvaluation performance (AutoDiff and Compilation)
  public class JacobianEvaluation {

    // different data sizes
    [Params(/*128, 256, 512,*/ 1024/*, 2048, 4096, 8192*/)]
    public int N;

    // number of evaluations (after compilation) (as e.g. in CG optimizer)
    [Params(1, 10 /* , 50*/, 100, 1000)]
    public int numEvals;

    [Params(/*4, 8, 16, 32, 64, 128, 256, 512,*/ 1024)]
    public int batchSize;

    public int Dim = 3;
    public int ParamDim = 2;
    public double[,] data;
    public double[][] dataCols;

    // the test expression with a mix of functions
    // Expression<Expr.ParametricFunction> expr = (p, x) => p[0] * x[0]
    //                                                      + Math.Log(Math.Abs(p[1] * x[1] + p[2] * x[2])) * Math.Sqrt(Math.Abs(p[3] * x[3]))
    //                                                      + Math.Pow(x[4], 3.0)
    //                                                      + Math.Sin(p[4] * x[4]);

    // the likelihood expression for a RAR model
    static Expression<Expr.ParametricFunction> model = (p, x) => Math.Log(Math.Pow(Math.Pow(p[0], x[0]) / x[0], p[1]) + x[0]) / Math.Log(10);
    // df/dgbar
    static Expression<Expr.ParametricFunction> dModel = Expr.Derive(model, model.Parameters[1], 0); // ((2.302585092994046 * (((Pow((Pow(p[0], x[0]) / x[0]), (p[1] - 1)) * (p[1] * (((x[0] * (Pow(p[0], (x[0] - 1)) * (p[0] * Log(p[0])))) - Pow(p[0], x[0])) / Pow(x[0], 2)))) + 1) / (Pow((Pow(p[0], x[0]) / x[0]), p[1]) + x[0]))) / 5.301898110478399)

    // df/d log gbar
    // 5.301898110478399 * x[0] * ((2.302585092994046 * (((Pow((Pow(p[0], x[0]) / x[0]), (p[1] - 1)) * (p[1] * (((x[0] * (Pow(p[0], (x[0] - 1)) * (p[0] * Log(p[0])))) - Pow(p[0], x[0])) / Pow(x[0], 2)))) + 1) / (Pow((Pow(p[0], x[0]) / x[0]), p[1]) + x[0]))) / 5.301898110478399)

    // sigma2_tot = e_loggobs**2 + (gobs1_diff*e_loggbar)**2
    static Expression<Expr.ParametricFunction> sigma_tot = (p, x) => (x[1] * x[1] + Math.Pow(
      5.301898110478399 * x[0] * ((2.302585092994046 * (((Math.Pow((Math.Pow(p[0], x[0]) / x[0]), (p[1] - 1)) * (p[1] * (((x[0] * (Math.Pow(p[0], (x[0] - 1)) * (p[0] * Math.Log(p[0])))) - Math.Pow(p[0], x[0])) / Math.Pow(x[0], 2)))) + 1) / (Math.Pow((Math.Pow(p[0], x[0]) / x[0]), p[1]) + x[0]))) / 5.301898110478399)
      * x[2], 2.0));

    // the full (expanded) model
    //  0.5 * np.sum((np.log10(gobs) - np.log10(gobs1))**2 ./ sigma2_tot + np.log(2.* np.pi * sigma2_tot))
    static Expression<Expr.ParametricFunction> expr = (p, x) => 0.5 * Math.Pow(x[0] - Math.Log(Math.Pow(Math.Pow(p[0], x[0]) / x[0], p[1]) + x[0]) / Math.Log(10), 2.0) / (x[1] * x[1] + Math.Pow(
      5.301898110478399 * x[0] * ((2.302585092994046 * (((Math.Pow((Math.Pow(p[0], x[0]) / x[0]), (p[1] - 1)) * (p[1] * (((x[0] * (Math.Pow(p[0], (x[0] - 1)) * (p[0] * Math.Log(p[0])))) - Math.Pow(p[0], x[0])) / Math.Pow(x[0], 2)))) + 1) / (Math.Pow((Math.Pow(p[0], x[0]) / x[0]), p[1]) + x[0]))) / 5.301898110478399)
      * x[2], 2.0)) + Math.Log(2.0 * Math.PI * (x[1] * x[1] + Math.Pow(
      5.301898110478399 * x[0] * ((2.302585092994046 * (((Math.Pow((Math.Pow(p[0], x[0]) / x[0]), (p[1] - 1)) * (p[1] * (((x[0] * (Math.Pow(p[0], (x[0] - 1)) * (p[0] * Math.Log(p[0])))) - Math.Pow(p[0], x[0])) / Math.Pow(x[0], 2)))) + 1) / (Math.Pow((Math.Pow(p[0], x[0]) / x[0]), p[1]) + x[0]))) / 5.301898110478399)
      * x[2], 2.0)));

    [GlobalSetup]
    public void Setup() {
      data = new double[N, Dim];
      dataCols = new double[Dim][];
      var rand = new System.Random(1234);
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < Dim; j++) {
          data[i, j] = rand.NextDouble();
          if (dataCols[j] == null) dataCols[j] = new double[N];
          dataCols[j][i] = data[i, j];
        }
      }
    }

    [Benchmark]
    public double[] EvalInterpreted() {
      var theta = new double[] { 1.0, 2.0};
      double[] f = null;
      var interpreter = new ExpressionInterpreter(expr, dataCols, N, batchSize );
      for (int i = 0; i < numEvals; i++) {
        f = interpreter.Evaluate(theta);
      }
      return f;
    }

    [Benchmark]
    public double[] EvalCompiled() {
      var theta = new double[] { 1.0, 2.0 };
      var func = Expr.Broadcast(expr).Compile();
      var f = new double[N];
      for (int i = 0; i < numEvals; i++) {
        func(theta, data, f);
      }
      return f;
    }

    [Benchmark]
    public double[,] EvalInterpretedJacobian() {
      var theta = new double[] { 1.0, 2.0 };
      double[] f;
      double[,] jacP = new double[N, theta.Length];
      double[,] jacX = new double[N, Dim];
      var interpreter = new ExpressionInterpreter(expr, dataCols, N, batchSize);
      for (int i = 0; i < numEvals; i++) {
        f = interpreter.EvaluateWithJac(theta, jacX, jacP);
      }
      return jacP;
    }

    [Benchmark]
    public double[,] EvalCompiledJacobian() {
      var theta = new double[] { 1.0, 2.0 };
      var funcJac = Expr.Jacobian(expr, theta.Length).Compile();
      double[,] jac = new double[N, theta.Length];
      var f = new double[N];
      for (int i = 0; i < numEvals; i++) {
        funcJac(theta, data, f, jac);
      }
      return jac;
    }

    [Benchmark]
    public double[,] EvalCompiledJacobianReverseAutoDiff() {
      var theta = new double[] { 1.0, 2.0 };
      var funcJac = ReverseAutoDiffVisitor.GenerateJacobianExpression(expr, N);
      double[,] jac = new double[N, theta.Length];
      var f = new double[N];
      for (int i = 0; i < numEvals; i++) {
        funcJac(theta, data, f, jac);
      }
      return jac;
    }
  }
}

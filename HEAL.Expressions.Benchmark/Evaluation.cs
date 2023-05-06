using BenchmarkDotNet.Attributes;
using System.Linq.Expressions;

namespace HEAL.Expressions.Benchmark {
  // benchmark JacobianEvaluation performance (AutoDiff and Compilation)
  public class JacobianEvaluation {

    // different data sizes
    [Params(/*128, 256, 512, */ 1024 /*, 2048, 4096, 8192*/)]
    public int N;

    // number of evaluations (after compilation) (as e.g. in CG optimizer)
    [Params(1, 10, 50)]
    public int numEvals;

    public int Dim = 10;
    public double[,] data;

    // the test expression with a mix of functions
    Expression<Expr.ParametricFunction> expr = (p, x) => p[0] * x[0]
                                                         + Math.Log(Math.Abs(p[1] * x[1] + p[2] * x[2])) * Math.Sqrt(Math.Abs(p[3] * x[3]))
                                                         + Math.Pow(x[4], 3.0)
                                                         + Math.Sin(p[4] * x[4]);

    [GlobalSetup]
    public void Setup() {
      data = new double[N, Dim];
      var rand = new System.Random(1234);
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < Dim; j++) {
          data[i, j] = rand.NextDouble();
        }
      }
    }

    [Benchmark]
    public double[] EvalWithInterpreter() {
      var theta = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
      double[] f = null;
      for (int i = 0; i < numEvals; i++) {
        f = Expr.EvaluateFunc(expr, theta, data);
      }
      return f;
    }

    [Benchmark]
    public double[] EvalWithCompiler() {
      var theta = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
      var func = Expr.Broadcast(expr).Compile();
      var f = new double[N];
      for (int i = 0; i < numEvals; i++) {
        func(theta, data, f);
      }
      return f;
    }

    [Benchmark]
    public double[,] EvalJacobianWithInterpreter() {
      var theta = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
      double[,] jac = new double[N, theta.Length];
      for (int i = 0; i < numEvals; i++) {
        Expr.EvaluateFuncJac(expr, theta, data, ref jac);
      }
      return jac;
    }

    [Benchmark]
    public double[,] EvalJacobianWithCompiler() {
      var theta = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
      var funcJac = Expr.Jacobian(expr, theta.Length).Compile();
      double[,] jac = new double[N, theta.Length];
      var f = new double[N];
      for (int i = 0; i < numEvals; i++) {
        funcJac(theta, data, f, jac);
      }
      return jac;
    }
  }
}

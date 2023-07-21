using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

namespace HEAL.Expressions.Benchmark {
  internal class Program {
    static void Main(string[] args) {
      var summary = BenchmarkRunner.Run<JacobianEvaluation>(/* new DebugInProcessConfig()*/ ); // for running benchmarks in debug builds
    }
  }
}
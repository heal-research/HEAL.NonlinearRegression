using BenchmarkDotNet.Running;

namespace HEAL.Expressions.Benchmark {
  internal class Program {
    static void Main(string[] args) {
      var summary = BenchmarkRunner.Run<JacobianEvaluation>();
    }
  }
}
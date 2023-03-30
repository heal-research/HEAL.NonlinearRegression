using System;
using System.Linq;

namespace HEAL.NonlinearRegression {
  public static class Util {

    public static double Variance(double[] x) {
      var xm = x.Average();
      var SSR = 0.0;
      for (int i = 0; i < x.Length; i++) {
        var r = x[i] - xm;
        SSR += r * r;
      }
      return SSR / x.Length;
    }

    internal static void CopyRow(double[,] x, int rowIdx, double[] xi) {
      var m = x.GetLength(0);
      if (rowIdx >= m) throw new ArgumentException();
      var n = x.GetLength(1);
      Buffer.BlockCopy(x, sizeof(double) * n * rowIdx, xi, 0, sizeof(double) * n);
    }
  }
}

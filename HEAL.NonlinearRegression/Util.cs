using System;
using System.Linq;

namespace HEAL.NonlinearRegression {
  public static class Util {

    public static double Variance(double[] x) {
      return alglib.samplevariance(x) * (x.Length -1) / x.Length; // population variance
    }

    internal static void CopyRow(double[,] x, int rowIdx, double[] xi) {
      var m = x.GetLength(0);
      if (rowIdx >= m) throw new ArgumentException();
      var n = x.GetLength(1);
      Buffer.BlockCopy(x, sizeof(double) * n * rowIdx, xi, 0, sizeof(double) * n);
    }
  }
}

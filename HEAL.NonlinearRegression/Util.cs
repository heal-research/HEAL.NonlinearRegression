using System;

namespace HEAL.NonlinearRegression {
  public static class Util {
    public static double SSR(double[] y, double[] yPred) {
      if (y.Length != yPred.Length) throw new ArgumentException("arrays must have the same length");
      var ssr = 0.0;
      for (int i = 0; i < y.Length; i++) {
        var r = yPred[i] - y[i];
        ssr += r * r;
      }
      return ssr;
    }

    public static double MAE(double[] y, double[] yPred) {
      if (y.Length != yPred.Length) throw new ArgumentException("arrays must have the same length");
      var sae = 0.0;
      for (int i = 0; i < y.Length; i++) {
        sae += Math.Abs(yPred[i] - y[i]);
      }
      return sae / y.Length;
    }

    public static double Variance(double[] x) {
      return alglib.samplevariance(x) * (x.Length - 1) / x.Length; // population variance
    }

    public static void CopyRow(double[,] x, int rowIdx, double[] xi) {
      var m = x.GetLength(0);
      if (rowIdx >= m) throw new ArgumentException();
      var n = x.GetLength(1);
      Buffer.BlockCopy(x, sizeof(double) * n * rowIdx, xi, 0, sizeof(double) * n);
    }

    public static double[][] ToColumns(double[,] x) {
      if (x == null) return null;
      var d = x.GetLength(1);
      var m = x.GetLength(0);
      var xc = new double[d][];
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < d; j++) {
          if (xc[j] == null) xc[j] = new double[m];
          xc[j][i] = x[i, j];
        }
      }
      return xc;
    }
  }
}

using System;

namespace HEAL.NonlinearRegression {
  internal class Util {
    public static double RandNorm(Random rand, int mean, int stdDev) {
      double u, v, s;
      do {
        u = rand.NextDouble() * 2 - 1;
        v = rand.NextDouble() * 2 - 1;
        s = u * u + v * v;
      } while (s >= 1 || s == 0);
      s = Math.Sqrt(-2.0 * Math.Log(s) / s);
      return mean + stdDev * u * s;
    }

    public static double[,] ToMatrix(double[] x) {
      // create a matrix from the vector x
      var X = new double[x.Length, 1];
      Buffer.BlockCopy(x, 0, X, 0, x.Length * sizeof(double));

      return X;
    }
  }
}
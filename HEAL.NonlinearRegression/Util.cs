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
  }
}

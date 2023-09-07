using System;

namespace HEAL.Expressions {
  public static class Functions {
    public static double Cbrt(double x) {
      if (x < 0) return -Math.Pow(-x, 1.0 / 3.0);
      else return Math.Pow(x, 1.0 / 3.0);
    }
    public static double PowAbs(double b, double e) => Math.Pow(Math.Abs(b), e);
    public static double AQ(double a, double b) {
      return a / Math.Sqrt(1 + b * b);
    }

    public static double Logistic(double x) {
      return 1.0 / (1 + Math.Exp(-x));
    }

    public static double LogisticPrime(double x) {
      var expx = Math.Exp(-x);
      var expx1 = expx + 1;
      return expx / (expx1 * expx1);
    }

    /*
                                  - 2 x          - x
                               2 %e             %e
    (%o31)                    ------------ - ------------
                                 - x     3      - x     2
                              (%e    + 1)    (%e    + 1)
    */
    public static double LogisticPrimePrime(double x) {
      var expx = Math.Exp(-x);
      var expx1 = expx + 1;
      return 2 * expx * expx / (expx1 * expx1 * expx1) - expx / (expx1 * expx1);
    }

    public static double InvLogistic(double p) {
      return -Math.Log(1 / p - 1);
    }

    public static double InvLogisticPrime(double p) {
      return 1.0 / (p - p * p);
    }

    //                                      1 - 2 p
    // (%o1)                             - ---------
    //                                           2 2
    //                                     (p - p )
    public static double InvLogisticPrimePrime(double p) {
      var denom = p - p * p;
      return -(1 - 2 * p) / (denom * denom);
    }

    public static double Sign(double x) {
      if (double.IsNaN(x)) return double.NaN;
      return Math.Sign(x); // a version of sign that returns double
    }

  }
}

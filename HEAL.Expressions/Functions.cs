using System;

namespace HEAL.Expressions {
  public static class Functions {
    public static double Cbrt(double x) {
      if (x < 0) return -Math.Pow(-x, 1.0 / 3.0);
      else return Math.Pow(x, 1.0 / 3.0);
    }
    public static double AQ(double a, double b) {
      return a / Math.Sqrt(1 + b * b);
    }
  }
}

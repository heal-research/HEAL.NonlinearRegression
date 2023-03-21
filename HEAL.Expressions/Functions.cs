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

    // TODO: prevent numerical problems when using Bernoulli log likelihood with logistic link function
    // uses a threshold to prevent numeric problems
    public static double Logistic(double x) {
      // var xLim = Math.Max(-15, Math.Min(15, x));
      var xLim = x;
      return 1.0 / (1 + Math.Exp(-xLim));
    }

    // use same threshold as above
    public static double LogisticPrime(double x) {
      // if (x > 15 || x < -15) return 0.0;
      // else 
        return Math.Exp(x) / Math.Pow(Math.Exp(x) + 1, 2);
    }

    public static double InvLogistic(double p) {
      return -Math.Log(1 / p - 1);
    }

    public static double InvLogisticPrime(double p) {
      return 1.0 / (p - p * p);
    }

  }
}

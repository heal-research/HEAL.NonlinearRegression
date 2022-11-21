using System;

namespace HEAL.Expressions {
  public static class Functions {
    // protected log (used in compiled expressions)
    public static double plog(double x) {
      return Math.Log(Math.Abs(x));
      // return x >= 0.001 ? Math.Log(x) : 0.0;
    }
  }
}

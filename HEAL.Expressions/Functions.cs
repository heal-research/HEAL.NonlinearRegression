using System;

namespace HEAL.Expressions {
  public static class Functions {
    // protected log (used in compiled expressions)
    public static double plog(double x) {
      if (x <= 0) return 0.0;
      else return Math.Log(x);
    }
  }
}

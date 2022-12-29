﻿using System;

namespace HEAL.Expressions {
  public static class Functions {
    // protected log (used in compiled expressions)
    public static double plog(double x) {
      return Math.Log(Math.Abs(x));
    }
    public static double psqrt(double x) {
      return Math.Sqrt(Math.Abs(x));
    }
  }
}

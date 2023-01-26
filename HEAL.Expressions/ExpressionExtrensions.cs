﻿using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using System.Text;

namespace HEAL.Expressions {
  internal static class ExpressionExtrensions {
    // Deconstruct methods for switch pattern matching
    public static void Deconstruct(this UnaryExpression expr, out ExpressionType nodeType, out Expression operand) {
      nodeType = expr.NodeType;
      operand = expr.Operand;
    }
    public static void Deconstruct(this BinaryExpression expr, out ExpressionType nodeType, out Expression left, out Expression right) {
      nodeType = expr.NodeType;
      left = expr.Left;
      right = expr.Right;
    }

    public static void Deconstruct(this ConstantExpression expr, out object val) {
      val = expr.Value;
    }

    public static void test() {
      List<(Expression,Expression)> bla = new List<(Expression,Expression)>() {
        ( (Expression<Func<double,double>>)((double x) =>  0.0 + x), (Expression<Func<double,double>>)((double x) =>  0.0 + x))
      };
    }
  }
}

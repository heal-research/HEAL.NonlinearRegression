using System;
using System.Linq.Expressions;
using HEAL.Expressions;

namespace HEAL.NonlinearRegression {
  internal class PagieProblem : INLSProblem {

    private static double[] pOpt = new double[] { 1, 2, 3, 4 };
    public PagieProblem() {
      int m = 676;
      var d = 2;
      X = new double[m, d];
      y = new double[m];

      // generate data
      int rowIdx = 0;
      for (var x1 = -5m; x1 <= 5; x1 += 0.4m) {
        for (var x2 = -5m; x2 <= 5; x2 += 0.4m) {
          X[rowIdx, 0] = (double)x1;
          X[rowIdx, 1] = (double)x2;
          rowIdx++;
        }
      }

      // generating function (Pagie)
      // no noise
      for (int i = 0; i < y.Length; i++) {
        y[i] = 1.0 / (1 + Math.Pow(X[i, 0], -4)) + 1.0 / (1.0 + Math.Pow(X[i, 1], -4));
      }
      
      // noise
      
    }

    public double[,] X { get; private set; }

    public double[] y { get; private set; }

    public static double[] thetaStart = new double[] {
      /*
      // original
      2.0241e-3,   // 0
      -1.3725e-2,  // 1
      -9.6464e-1,  // 2
      -4.566e-1,   // 3
      1.417,       // 4
      -9.4935,     // 5
      -1.874e1,    // 6
      2.8035e1,    // 7
      -2.6003e1,   // 8
      2.59476e1,   // 9 
      -7.79373e-2, // 10
      1.61634e0    // 11
      */
      
       // reduced parameters
      2.0241e-3,   // 0
      -1.3725e-2,  // 1
      -9.6464e-1,  // 2
      -4.566e-1,   // 3
      1.417,       // 4
      -9.4935 / 2.8035e1,     // 5
      -1.874e1 / 2.8035e1,    // 6
      // 2.8035e1,    // 7
      -2.6003e1 * Math.Pow(-7.79373e-2, 3),   // 8
      2.59476e1 * Math.Pow(-7.79373e-2, 3),   // 9 
      // , // 10
      1.61634e0    // 11
      
    };
    public double[] ThetaStart => thetaStart;

    
    // GrammarEnum result
    // X * X * 2.02414448e-003 +
    // Y * Y * -1.37255775e-002 +
    // exp(Y * Y * -9.64638596e-001) * -4.56602262e-001 +
    // 1/(exp(X * X * 1.41695844e+000) * -9.49349871e+000 + -1.87396062e+001) * 2.80349976e+001 +
    // cbrt(Y * Y * -2.60030338e+001 + 2.59476220e+001) * -7.79373531e-002 +
    // 1.61634069e+000
    
    public static Expression<Expr.ParametricFunction> ModelExpr = (theta, x) =>

        /*
        // original
        x[0] * x[0] * theta[0] +
        x[1] * x[1] * theta[1] +
        Math.Exp(x[1] * x[1] * theta[2]) * theta[3] +
        1.0 / (Math.Exp(x[0] * x[0] * theta[4]) * theta[5] + theta[6]) * theta[7] +
        Math.Cbrt(x[1] * x[1] * theta[8] + theta[9]) * theta[10] + 
        theta[11]
        */
      
      // reduced parameters
        x[0] * x[0] * theta[0] +
        x[1] * x[1] * theta[1] +
        Math.Exp(x[1] * x[1] * theta[2]) * theta[3] +
        1.0 / (Math.Exp(x[0] * x[0] * theta[4]) * theta[5] + theta[6]) +
        Math.Cbrt(x[1] * x[1] * theta[7] + theta[8]) + 
        theta[9]
        
      ;
      

    // generating expression
    //public static double[] thetaStart => new[] { 1.0, 1.0 };
    //public static Expression<Expr.ParametricFunction> ModelExpr = (theta, x) =>
    //  theta[0] / (1.0 + Math.Pow(x[0], -4)) + theta[1] / (1.0 + Math.Pow(x[1], -4));
    

    // theta, X, f, ignored_result
    public static Expression<Expr.ParametricVectorFunction> BroadcastModelExpr = Expr.Broadcast(ModelExpr);

    // theta, X, f, J
    private static Expr.ParametricVectorFunction ModelFunc = BroadcastModelExpr.Compile();
    
    private static Expression<Expr.ParametricGradientFunction> GradientExpr = Expr.Gradient(ModelExpr, thetaStart.Length);
    private static Expression<Expr.ParametricJacobianFunction> BroadcastJacobianExpr = Expr.Broadcast(GradientExpr);
    private static Expr.ParametricJacobianFunction JacobianFunc = BroadcastJacobianExpr.Compile();
    public void Func(double[] theta, double[,] X, double[] f) {
      ModelFunc(theta, X, f);
    }

    public void Jacobian(double[] theta, double[,] X, double[] f, double[,] jac) {
      JacobianFunc(theta, X, f, jac);
    }
  }
}
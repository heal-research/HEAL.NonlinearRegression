using System;
using System.Linq.Expressions;
using HEAL.Expressions;

namespace HEAL.NonlinearRegression.Demo {
  // PCB example from Nonlinear Regression Analysis and Its Applications, Bates and Watts, 1988

  internal class PCBDSRProblem : SymbolicProblemBase {
    private double[] age = new double[] {
                  1,
                  1,
                  1,
                  1,
                  2,
                  2,
                  2,
                  3,
                  3,
                  3,
                  4,
                  4,
                  4,
                  5,
                  6,
                  6,
                  6,
                  7,
                  7,
                  7,
                  8,
                  8,
                  8,
                  9,
                  11,
                  12,
                  12,
                  12
                };

    private double[] PCB = new double[] {
                  0.6,
                  1.6,
                  0.5,
                  1.2,
                  2.0,
                  1.3,
                  2.5,
                  2.2,
                  2.4,
                  1.2,
                  3.5,
                  4.1,
                  5.1,
                  5.7,
                  3.4,
                  9.7,
                  8.6,
                  4.0,
                  5.5,
                  10.5,
                  17.5,
                  13.4,
                  4.5,
                  30.4,
                  12.4,
                  13.4,
                  26.2,
                  7.4
                };
    public PCBDSRProblem() {
      

      var m = PCB.Length;

      
      X = new double[m, 1];
      y = PCB;
      for (int i = 0; i < m; i++) {
        X[i, 0] = age[i];
      }
    }


    public override double[,] X { get;}

    public override double[] y { get;  }
    private static double[] thetaStart => new double[] { 1.37606397e+000, 6.60401507e-002, -6.96730099e-002, 6.30467914e-001, -1.34198596e+000};
// p_opt p_opt: 1.37607e+000 6.60443e-002 -6.96729e-002 6.30467e-001 -1.34200e+000
    
    public override double[] ThetaStart => thetaStart;

    // From grammar enumeration
    // age * 1.37606397e+000 + 1/(age * -6.96730099e-002 + 6.30467914e-001) * 6.60401507e-002 + -1.34198596e+000
    public override Expression<Expr.ParametricFunction> ModelExpression => (p, x) => 
      p[0] * x[0] + 
      p[1]/(x[0] * p[2] + p[3])
      + p[4];

  }
}
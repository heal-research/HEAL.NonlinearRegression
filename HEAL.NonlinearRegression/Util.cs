namespace HEAL.NonlinearRegression {
  internal static class Util {
    public static ResidualFunction CreateResidualFunction(Function func, double[,] x, double[] y) {
      return (p, f) => {
        func(p, x, f);
        for (int i = 0; i < f.Length; i++) {
          f[i] = f[i] - y[i];
        }
      };
    }

    public static ResidualJacobian CreateResidualJacobian(Jacobian jacobian, double[,] x, double[] y) {
      return (p, f, jac) => {
        jacobian(p, x, f, jac);
        for (int i = 0; i < f.Length; i++) {
          f[i] = f[i] - y[i];
        }
        // Jacobian can be passed through ( ∇(f(x,p) - y) == ∇f(x,p) )
      };
    }

    public static ResidualJacobian FixParameter(ResidualJacobian jacobian, int idx) {
      return (p, f, jac) => {
        jacobian(p, f, jac);
        for (int i = 0; i < f.Length; i++) {
          jac[i, idx] = 0.0; // derivative of fixed parameter is zero
        }
      };
    }

    public static alglib.ndimensional_fvec CreateAlgibResidualFunction(ResidualFunction func) {
      return (double[] p, double[] f, object o) => {
        func(p, f);
      };
    }
    public static alglib.ndimensional_jac CreateAlgibResidualJacobian(ResidualJacobian jac) {
      return (double[] p, double[] f, double[,] j, object o) => {
        jac(p, f, j);
      };
    }
  }
}

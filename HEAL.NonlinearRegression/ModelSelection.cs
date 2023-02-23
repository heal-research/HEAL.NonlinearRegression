using HEAL.Expressions;
using System;
using System.Linq;


namespace HEAL.NonlinearRegression {
  public static class ModelSelection {
    // TODO: parameter should be fitted nlr object?
    public static double LogLikelihood(double[] y, double[] yPred, double noiseSigma) {
      if (y.Length != yPred.Length) throw new ArgumentException();
      var SSR = y.Zip(yPred, (y, yp) => (y - yp) * (y - yp)).Sum();
      int n = y.Length;
      var s2 = noiseSigma * noiseSigma;
      return -n / 2.0 * Math.Log(2 * Math.PI * s2) - SSR / (2.0 * s2);

    }

    public static double AIC(double[] y, double[] yPred, double modelDoF, double noiseSigma) {
      return 2 * (modelDoF + 1) - 2 * LogLikelihood(y, yPred, noiseSigma); // also count noise sigma as a parameter
    }
    public static double AICc(double[] y, double[] yPred, double modelDoF, double noiseSigma) {
      // noise sigma is counted as a parameter
      return AIC(y, yPred, modelDoF, noiseSigma) + 2 * (modelDoF + 1) * (modelDoF + 2) / (y.Length - (modelDoF + 1) - 1);
    }

    public static double BIC(double[] y, double[] yPred, double modelDoF, double noiseSigma) {
      return (modelDoF + 1) * Math.Log(y.Length) - 2 * LogLikelihood(y, yPred, noiseSigma);
    }
  }
}

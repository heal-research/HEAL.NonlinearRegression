using System;

namespace HEAL.NonlinearRegression {
  public static class ModelSelection {
    // TODO move MDL code here
    public static double AIC(double logLikelihood, double modelDoF) {
      return 2 * (modelDoF + 1) - 2 * logLikelihood; // also count noise sigma as a parameter
    }
    public static double AICc(double logLikelihood, double modelDoF, double numObservations) {
      // noise sigma is counted as a parameter
      // TODO not all likelihoods have an additional parameter that has to be counted.
      return AIC(logLikelihood, modelDoF) + 2 * (modelDoF + 1) * (modelDoF + 2) / (numObservations - (modelDoF + 1) - 1);
    }

    public static double BIC(double logLikelihood, double modelDoF, double numObservations) {
      return (modelDoF + 1) * Math.Log(numObservations) - 2 * logLikelihood;
    }
  }
}

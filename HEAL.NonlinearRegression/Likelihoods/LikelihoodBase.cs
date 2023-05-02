using HEAL.Expressions;
using System;
using System.Linq.Expressions;

namespace HEAL.NonlinearRegression {
  // not necessary if we discover likelihood types automatically
  public enum LikelihoodEnum { Gaussian, Bernoulli } // TODO: Poisson, Cauchy, Multinomial, ...

  // A likelihood function together with its model.
  // The likelihood has parameters (which are usually the model parameters)
  public abstract class LikelihoodBase {
    // all our models are functions real^n -> real
    protected readonly double[,] x;
    protected readonly double[] y;
    protected readonly int numLikelihoodParams; // additional parameters of the likelihood function (not part of the model) e.g. sErr for Gaussian likelihood
    private int numModelParams;

    public double[,] X => x;
    public double[] Y => y;

    public virtual double Dispersion { get { return 1.0; } set { throw new NotSupportedException($"cannot set dispersion of {this.GetType().Name}"); } }

    protected LikelihoodBase(LikelihoodBase original) : this(original.modelExpr, original.x, original.y, original.numLikelihoodParams) { }
    protected LikelihoodBase(Expression<Expr.ParametricFunction> modelExpr, double[,] x, double[] y, int numLikelihoodParams) {
      this.x = x;
      this.y = y;
      this.NumberOfObservations = y.Length;
      this.numLikelihoodParams = numLikelihoodParams;

      ModelExpr = modelExpr;
    }

    private Expression<Expr.ParametricFunction> modelExpr;
    public Expression<Expr.ParametricFunction> ModelExpr {
      get => modelExpr;
      internal set {
        // updating the modelExpr also requires updating Jacobian and Hessian
        modelExpr = value;

        numModelParams = Expr.NumberOfParameters(modelExpr);
        ModelFunc = Expr.Broadcast(modelExpr).Compile();
        ModelJacobian = Expr.Jacobian(modelExpr, numModelParams).Compile();
        ModelHessian = Expr.Hessian(modelExpr, numModelParams).Compile();
      }
    }
    protected Expr.ParametricVectorFunction ModelFunc { get; private set; }
    protected Expr.ParametricJacobianFunction ModelJacobian { get; private set; }
    protected Expr.ParametricHessianFunction ModelHessian { get; private set; }


    public int NumberOfObservations { get; }
    public int NumberOfParameters => numLikelihoodParams + numModelParams;

    public abstract double BestNegLogLikelihood { get; } // the likelihood of a perfect model (with zero residuals)

    public abstract double NegLogLikelihood(double[] p);
    public abstract void NegLogLikelihoodGradient(double[] p, out double nll, double[]? nll_grad = null);

    public abstract double[,] FisherInformation(double[] p); // Hessian of the log likelihood

    public abstract LikelihoodBase Clone();

  }
}

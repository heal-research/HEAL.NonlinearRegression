using HEAL.Expressions;
using System.Linq.Expressions;

namespace HEAL.NonlinearRegression {

  // A likelihood function together with its model.
  // The likelihood has parameters (which are usually the model parameters)
  public abstract class LikelihoodBase {
    // all our models are functions real^n -> real
    protected readonly double[,] x;
    protected readonly double[] y;
    protected readonly int numLikelihoodParams; // additional parameters of the likelihood function (not part of the model) e.g. sErr for Gaussian likelihood
    private int numModelParams;

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
        var _func = Expr.Broadcast(modelExpr).Compile();
        var _jac = Expr.Jacobian(modelExpr, numModelParams).Compile();
        // var _hess = Expr.Hessian(modelExpr, numModelParams).Compile();
        ModelFunc = (double[] p, double[,] X, double[] f) => _func(p, X, f); // wrapper only necessary because return values are incompatible;
        ModelJacobian = (double[] p, double[,] X, double[] f, double[,] jac) => _jac(p, X, f, jac);
        // this.ModelHessian = (double[] p, double[,] X, double[,] hess) => _hess(p, X, hess);
      }
    }
    protected Function ModelFunc { get; private set; }
    protected Jacobian ModelJacobian { get; private set; }
    // protected Hessian ModelHessian { get; private set; }


    public int NumberOfObservations { get; }
    public int NumberOfParameters => numLikelihoodParams + numModelParams;

    public abstract double NegLogLikelihood(double[] p);
    public abstract void NegLogLikelihoodGradient(double[] p, out double nll, double[]? nll_grad = null);

    public abstract double[,] FisherInformation(double[] p); // Hessian of the log likelihood

    public abstract LikelihoodBase Clone();

  }
}

using HEAL.Expressions;
using System.Linq;
using System.Linq.Expressions;

namespace HEAL.NonlinearRegression {
  // TODO: not necessary if we discover likelihood types automatically
  public enum LikelihoodEnum { Gaussian, Bernoulli } // TODO: Poisson, Cauchy, Multinomial, ...

  // A likelihood function together with its model.
  // The likelihood has parameters (which are usually the model parameters)
  public abstract class LikelihoodBase {
    // all our models are functions real^n -> real
    protected readonly double[,] x;
    protected readonly double[][] xCol;
    protected readonly double[] y;
    protected readonly int numLikelihoodParams; // additional parameters of the likelihood function (not part of the model) e.g. sErr for Gaussian likelihood
    private int numModelParams;

    private ExpressionInterpreter interpreter;
    protected ExpressionInterpreter Interpreter {

      get {
        if (interpreter == null)
          interpreter = new ExpressionInterpreter(modelExpr, xCol, NumberOfObservations);

        return interpreter;
      }
    }

    private ExpressionInterpreter[] gradInterpreter;
    protected ExpressionInterpreter[] GradInterpreter {
      get {
        if (gradInterpreter == null)
          gradInterpreter = Gradient.Select(g => new ExpressionInterpreter(g, xCol, NumberOfObservations)).ToArray();

        return gradInterpreter;
      }
    }



    private Expression<Expr.ParametricFunction>[] symbolicGradient;
    protected Expression<Expr.ParametricFunction>[] Gradient {
      get {
        if (symbolicGradient == null)
          // TODO: use forward/reverse autodiff for Hessian
          symbolicGradient = Enumerable.Range(0, numModelParams).Select(pIdx => Expr.Derive(modelExpr, pIdx)).ToArray();

        return symbolicGradient;
      }
    }


    public double[,] X => x;
    public double[][] XCol => xCol; // column-oriented representation
    public double[] Y => y;

    protected LikelihoodBase(LikelihoodBase original) : this(original.modelExpr, original.x, original.y, original.numLikelihoodParams) { }
    protected LikelihoodBase(Expression<Expr.ParametricFunction> modelExpr, double[,] x, double[] y, int numLikelihoodParams) {
      this.x = x;
      this.xCol = Util.ToColumns(x);
      this.y = y;
      this.NumberOfObservations = y.Length;
      this.numLikelihoodParams = numLikelihoodParams;

      ModelExpr = modelExpr;
    }



    private Expression<Expr.ParametricFunction> modelExpr;
    public virtual Expression<Expr.ParametricFunction> ModelExpr {
      get { return modelExpr; }
      set {
        // updating the modelExpr also requires updating Jacobian and Hessian
        modelExpr = value;
        if (modelExpr != null) {
          numModelParams = Expr.NumberOfParameters(modelExpr);
        } else {
          numModelParams = 0;
        }
        interpreter = null;
        symbolicGradient = null;
        gradInterpreter = null;
      }
    }



    public int NumberOfObservations { get; }
    public int NumberOfParameters => numLikelihoodParams + numModelParams;

    public abstract double BestNegLogLikelihood(double[] p); // the likelihood of a perfect model (with zero residuals)

    public abstract double NegLogLikelihood(double[] p);
    public abstract void NegLogLikelihoodGradient(double[] p, out double nll, double[] nll_grad = null);

    public abstract double[,] FisherInformation(double[] p); // negative of Hessian of the log likelihood

    public abstract LikelihoodBase Clone();

    // creates a Gaussian likelihood as an approximation
    public ApproximateLikelihood LaplaceApproximation(double[] pOpt) {
      return new ApproximateLikelihood(x, y, modelExpr, NegLogLikelihood(pOpt), pOpt, FisherInformation(pOpt));
    }

  }
}

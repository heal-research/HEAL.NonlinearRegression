# HEAL.NonlinearRegression
C# implementation of nonlinear least squares fitting including calculation of t-profiles and pairwise profile plots (see [1]).
The t-profiles allow to calculate exact confidence intervals for nonlinear parameters and approximate pairwise confidence regions.

Implementation is based on:

`[1] Douglas Bates and Donald Watts, Nonlinear Regression and Its Applications, John Wiley and Sons, 1988`

# Building
```
git clone https://github.com/heal-research/HEAL.NonlinearRegression
cd HEAL.NonlinearRegression
dotnet build
```

Run the demo program:
```
dotnet run --project Demo
```

The demo runs three examples from [1] (PCB, BOD, Puromycin) and produces parameter estimates, predictions and a list of points on the 
the approximate pairwise confidence region contours.
```
Puromycin example
-----------------
p_opt: 2.12684e+002 6.41213e-002
Successful: True, NumIters: 16, NumFuncEvals: 40, NumJacEvals: 17
SSR 1.1954e+003 s 1.0934e+001
Para       Estimate      Std. error          Lower          Upper Correlation matrix
    0    2.1268e+002    6.9472e+000    1.9720e+002    2.2816e+002 1.00
    1    6.4121e-002    8.2809e-003    4.5670e-002    8.2572e-002 0.77 1.00

         yPred            low            high
   5.0566e+001    3.9499e+001    6.1633e+001
   5.0566e+001    3.9499e+001    6.1633e+001
   1.0281e+002    8.9049e+001    1.1657e+002
   1.0281e+002    8.9049e+001    1.1657e+002
   1.3436e+002    1.2249e+002    1.4624e+002
   1.3436e+002    1.2249e+002    1.4624e+002
   1.6468e+002    1.5457e+002    1.7480e+002
   1.6468e+002    1.5457e+002    1.7480e+002
   1.9083e+002    1.7767e+002    2.0400e+002
   1.9083e+002    1.7767e+002    2.0400e+002
   2.0097e+002    1.8508e+002    2.1686e+002
   2.0097e+002    1.8508e+002    2.1686e+002
```

# Usage
To call the library you have to provide a function for evaluating the output
of your model for a given set of parameters and another function that also produces the Jacobian matrix.

For example for the Puromycin model the functions F and Jac are:
```
  // substrate concentration
  var x = new double[] {
    0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56, 1.10, 1.10
  };

  // target
  var treated = new double[] {
    76, 47, 97, 107, 123, 139, 159, 152, 191, 201, 207, 200
  };


  // model: y = p1 x / (p2 + x)
  void F(double[] p, double[] fi) {
    for (int i = 0; i < m; i++) {
      fi[i] = p[0] * x[i] / (p[1] + x[i]);
    }
  }

  void Jac(double[] p, double[] fi, double[,] Jac) {
    F(p, fi);
    for (int i = 0; i < m; i++) {
      Jac[i, 0] = x[i] / (p[1] + x[i]);
      Jac[i, 1] = -p[0] * x[i] / Math.Pow(p[1] + x[i], 2);
    }
  }
```

Then you call the static fit method with a starting point for the parameters.
```
  var p = new double[] { 205, 0.08 };  // Bates and Watts page 41
  NonlinearRegression.FitLeastSquares(p, F, Jac, treated, out var report);
```

The fitting `report` contains statistics and allows to generate profile pair contours. 
```
    report.Statistics.ApproximateProfilePairContour(0, 1, alpha: 0.05, out _, out _, out var p1, out var p2);
```

# Dependencies
The implementation uses alglib (https://alglib.net) for linear algebra and Levenberg-Marquard nonlinear least squares fitting. 
Alglib is licensed under GPL2+ and includes code from other projects. Commercial licenses for alglib are available.

# License
The code is licensed under the conditions of the MIT license.

# HEAL.NonlinearRegression
C# implementation of nonlinear least squares fitting including calculation of t-profiles and pairwise profile plots (see [1]).
The t-profiles allow to calculate exact confidence intervals for nonlinear parameters and approximate pairwise confidence regions.

Implementation is based on:

`[1] Douglas Bates and Donald Watts, Nonlinear Regression and Its Applications, John Wiley and Sons, 1988`


[![Unit tests](https://github.com/heal-research/HEAL.NonlinearRegression/actions/workflows/build_and_test.yml/badge.svg?branch=main)](https://github.com/heal-research/HEAL.NonlinearRegression/actions/workflows/build_and_test.yml)

# Building
```
git clone https://github.com/heal-research/HEAL.NonlinearRegression
cd HEAL.NonlinearRegression
dotnet build
```

Run the tests for fitting nonlinear models:
```
dotnet test --filter "FullyQualifiedName~Fit"
```

```
Starting test execution, please wait...
A total of 1 test files matched the specified pattern.
p_opt: 1.10421e+002 1.03488e+002
Successful: True, NumIters: 2, NumFuncEvals: 10, NumJacEvals: 0
SSR: 9.5471e+003  s: 3.0898e+001 AICc: 19.0 BIC: 17.5 MDL: 15.1
Para       Estimate      Std. error     z Score          Lower          Upper Correlation matrix
    0    1.1042e+002    2.3371e+001   4.72e+000    5.8347e+001    1.6249e+002 1.00
    1    1.0349e+002    1.2024e+001   8.61e+000    7.6697e+001    1.3028e+002 -0.67 1.00

Optimized: ((110.42107672063618 * x0) + 103.48806186471386)


p_opt: 1.38378e+000 4.84833e-002 5.24299e-001 3.52511e-001 -6.84851e-002 -1.11809e+001
Successful: True, NumIters: 2, NumFuncEvals: 41, NumJacEvals: 0
Deviance: 7.8438e+002  Dispersion: 1.0000e+000 AICc: 808.7 BIC: 866.8 MDL: 456.0
Para       Estimate      Std. error     z Score          Lower          Upper Correlation matrix
    0    1.3838e+000    1.7042e-001   8.12e+000    1.0493e+000    1.7182e+000 1.00
    1    4.8483e-002    7.5999e-003   6.38e+000    3.3569e-002    6.3398e-002 -0.04 1.00
    2    5.2430e-001    9.7525e-002   5.38e+000    3.3291e-001    7.1569e-001 -0.08 0.02 1.00
    3    3.5251e-001    8.0993e-002   4.35e+000    1.9357e-001    5.1145e-001 -0.16 -0.08 -0.55 1.00
    4   -6.8485e-002    2.4032e-001  -2.85e-001   -5.4009e-001    4.0312e-001 -0.04 0.00 0.02 -0.10 1.00
    5   -1.1181e+001    1.0533e+000  -1.06e+001   -1.3248e+001   -9.1140e+000 -0.60 -0.38 -0.11 0.12 -0.62 1.00

Optimized: Logistic(((((((1.3837834757792504 * BI_RADS) + (0.04848326870703262 * Age)) + (0.5242993934295344 * Shape)) + (0.35251072256817134 * Margin)) + (-0.06848513367625395 * Density)) + -11.180915607397576))


p_opt: 6.41213e-002 2.12684e+002
Successful: True, NumIters: 3, NumFuncEvals: 44, NumJacEvals: 0
SSR: 1.1954e+003  s: 1.0934e+001 AICc: 19.0 BIC: 17.5 MDL: 21.5
Para       Estimate      Std. error     z Score          Lower          Upper Correlation matrix
    0    6.4121e-002    8.7112e-003   7.36e+000    4.4711e-002    8.3531e-002 1.00
    1    2.1268e+002    7.1607e+000   2.97e+001    1.9673e+002    2.2864e+002 0.78 1.00

Optimized: ((x0 / (0.06412128166090875 + x0)) * 212.68374312341493)

Passed!  - Failed:     0, Passed:     3, Skipped:     0, Total:     3, Duration: 274 ms - HEAL.NonlinearRegression.Console.Tests.dll (net6.0)
```

Run the tests for profile likelihood confidence intervals:
```
dotnet test --filter "(FullyQualifiedName~ProfilePuromycin|FullyQualifiedName~ProfileMammography)"
```

```
Starting test execution, please wait...
A total of 1 test files matched the specified pattern.

profile-based marginal confidence intervals (alpha=0.05)
p0    1.3838e+000    1.0586e+000    1.7270e+000
p1    4.8483e-002    3.3810e-002    6.3677e-002
p2    5.2430e-001    3.3325e-001    7.1665e-001
p3    3.5251e-001    1.9415e-001    5.1249e-001
p4   -6.8485e-002   -5.3640e-001    4.0944e-001
p5   -1.1181e+001   -1.3314e+001   -9.1744e+000


profile-based marginal confidence intervals (alpha=0.05)
p0    6.4121e-002    4.6920e-002    8.6157e-002
p1    2.1268e+002    1.9730e+002    2.2929e+002

Passed!  - Failed:     0, Passed:     2, Skipped:     0, Total:     2, Duration: 5 s - HEAL.NonlinearRegression.Console.Tests.dll (net6.0)
```

# Usage
To call the library you have to provide an expression for the model as well as a dataset to fit to.

```csharp
var x = new double[,] { { 0.02 }, { 0.02 }, { 0.06 }, { 0.06 }, { 0.11 }, { 0.11 }, { 0.22 }, { 0.22 }, { 0.56 }, { 0.56 }, { 1.10 }, { 1.10 } };
var y = new double[] {76, 47, 97, 107, 123, 139, 159, 152, 191, 201, 207, 200 };

var nlr = new NonlinearRegression();
nlr.Fit("0.1 * x0 / (1.0f + 0.1 * x0)", new[] { "x0" }, LikelihoodEnum.Gaussian, x, y);
var prediction = nlr.PredictWithIntervals(x, IntervalEnum.LaplaceApproximation);
System.Console.WriteLine($"pred: {prediction[0, 0]}, low: {prediction[0, 2]}, high: {prediction[0, 3]}");
```

# Dependencies
The implementation uses alglib (https://alglib.net) for linear algebra and nonlinear least squares fitting. 
Alglib is licensed under GPL2+ and includes code from other projects. Commercial licenses for alglib are available.

# License
The code is licensed under the conditions of the GPL version 3.

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

Run the tests for fitting nonlinear models:
```
dotnet test --filter "FullyQualifiedName~Fit"
```

Run the tests for profile likelihood confidence intervals:
```
dotnet test --filter "FullyQualifiedName~Profile"
```


# Usage
To call the library you have to provide an expression for the model as well as a dataset to fit to.



# Dependencies
The implementation uses alglib (https://alglib.net) for linear algebra and nonlinear least squares fitting. 
Alglib is licensed under GPL2+ and includes code from other projects. Commercial licenses for alglib are available.

# License
The code is licensed under the conditions of the GPL version 3.

﻿using NUnit.Framework;
using NUnit.Framework.Legacy;
using System.Globalization;

namespace HEAL.NonlinearRegression.Console.Tests {
  public class NLR_EndToEnd {
    [SetUp]
    public void Setup() {
      CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
      CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;
      Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
      Thread.CurrentThread.CurrentUICulture = CultureInfo.InvariantCulture;
    }

    #region nonlinear Puromycin
    [Test]
    public void FitPuromycin() {
      // result in R;
      // Parameter intervals are profile based
      // Formula: y ~ a * x0/(b + x0)
      // 
      // Parameters:
      //    Estimate Std. Error t value Pr(>|t|)    
      // a 2.127e+02  6.947e+00  30.615 3.24e-11 ***
      // b 6.412e-02  8.281e-03   7.743 1.57e-05 ***
      // ---
      // Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
      // 
      // Residual standard error: 10.93 on 10 degrees of freedom
      // 
      // Number of iterations to convergence: 5 
      // Achieved convergence tolerance: 8.824e-06

      //            2.5%        97.5%
      //  a 197.30212848 229.29006490
      //  b   0.04692517   0.08615995

      // The differences in standard error and confidence intervals are because
      // R ignores the Hessian of the model in the Laplace approximation.
      // The results are close enough.

      var expected = @"p_opt: 6.41213e-002 2.12684e+002
Successful: True, NumIters: 1, NumFuncEvals: 21, NumJacEvals: 21
SSR: 1.1954e+003  s: 1.0934e+001 RMSE: 1.0934e+001 AICc: 98.5 BIC: 96.9 DL: 61.23  DL (lattice): 59.14 neg. Evidence: 33.06
Para       Estimate      Std. error     z Score          Lower          Upper Correlation matrix
    0    6.4121e-002    8.7112e-003   7.36e+000    4.4711e-002    8.3531e-002 1.00
    1    2.1268e+002    7.1607e+000   2.97e+001    1.9673e+002    2.2864e+002 0.78 1.00

Optimized: x0 / (0.064121282 + x0) * 212.68374
";
      NlrFit("Puromycin.csv", "x0 / (0.06412128165180965 + x0) * 212.68374312341493", "0:11", "y", "Gaussian", expected);
    }

    [Test]
    public void ProfilePuromycin() {
      // R:
      //  confint(fit_nls, level = 0.95)
      // #           2.5%        97.5%
      // # a 197.30212848 229.29006490
      // # b   0.04692517   0.08615995
      var expected = @"profile-based marginal confidence intervals (alpha=0.05)
p0    6.4121e-002    4.6920e-002    8.6157e-002
p1    2.1268e+002    1.9730e+002    2.2929e+002
";
      NlrProfile("Puromycin.csv", "x0 / (0.06412128165180965 + x0) * 212.68374312341493", "0:11", "y", "Gaussian", expected);
    }

    [Test]
    public void EvaluatePuromycin() {
      var expected = @"SSR: 1195.45 MSE: 99.6207 RMSE: 9.98102 NMSE: 0.0387392 R2: 0.9613 MAE: 7.559  LogLik: -44.7294 AIC: 95.46 AICc: 98.46 BIC: 96.91 DL: 61.23 DL_lattice: 59.14 neg. Evidence: 33.06 DoF: 2 m: 12
";
      NlrEvaluate("Puromycin.csv", "x0 / (0.06412128165180965 + x0) * 212.68374312341493", "0:11", "y", "Gaussian", expected);
    }

    [Test]
    public void PredictPuromycinLaplace() {
      var expected = @"x0,y,residual,yPred,yPredLow,yPredHigh,isTrain,isTest
0.02,76,25.434,50.566,41.5437,59.5882,1,0
0.02,47,-3.56598,50.566,41.5437,59.5882,1,0
0.06,97,-5.81093,102.811,91.661,113.961,1,0
0.06,107,4.18907,102.811,91.661,113.961,1,0
0.11,123,-11.3616,134.362,124.866,143.857,1,0
0.11,139,4.63841,134.362,124.866,143.857,1,0
0.22,159,-5.68468,164.685,156.802,172.568,1,0
0.22,152,-12.6847,164.685,156.802,172.568,1,0
0.56,191,0.167065,190.833,180.466,201.2,1,0
0.56,201,10.1671,190.833,180.466,201.2,1,1
1.1,207,6.03115,200.969,188.335,213.603,1,1
1.1,200,-0.968852,200.969,188.335,213.603,1,1
";
      NlrPredict("Puromycin.csv", "x0 / (0.06412128165180965 + x0) * 212.68374312341493", "0:11", "0:11", "y", "Gaussian", "LaplaceApproximation", expected);
    }

    [Test]
    public void PredictPuromycinProfile() {
      var expected = @"x0,y,residual,yPred,yPredLow,yPredHigh,isTrain,isTest
0.02,76,25.434,50.566,42.2591,60.3665,1,0
0.02,47,-3.56598,50.566,42.2591,60.3665,1,0
0.06,97,-5.81093,102.811,91.7575,113.973,1,0
0.06,107,4.18907,102.811,91.7575,113.973,1,0
0.11,123,-11.3616,134.362,124.628,143.621,1,0
0.11,139,4.63841,134.362,124.628,143.621,1,0
0.22,159,-5.68468,164.685,156.776,172.55,1,0
0.22,152,-12.6847,164.685,156.776,172.55,1,0
0.56,191,0.167065,190.833,180.469,201.197,1,0
0.56,201,10.1671,190.833,180.469,201.197,1,1
1.1,207,6.03115,200.969,188.508,213.774,1,1
1.1,200,-0.968852,200.969,188.508,213.774,1,1
";
      NlrPredict("Puromycin.csv", "x0 / (0.06412128165180965 + x0) * 212.68374312341493", "0:11", "0:11", "y", "Gaussian", "TProfile", expected);
    }
    #endregion

    #region linear Puromycin (to compare LaplaceApproximation = tProfile for linear Gaussian models)
    [Test]
    public void FitLinearPuromycin() {
      var expected = @"p_opt: 1.10421e+002 1.03488e+002
Successful: True, NumIters: 1, NumFuncEvals: 11, NumJacEvals: 11
SSR: 9.5471e+003  s: 3.0898e+001 RMSE: 3.0898e+001 AICc: 123.4 BIC: 121.8 DL: 67.33  DL (lattice): 58.69 neg. Evidence: 41.93
Para       Estimate      Std. error     z Score          Lower          Upper Correlation matrix
    0    1.1042e+002    2.3371e+001   4.72e+000    5.8347e+001    1.6249e+002 1.00
    1    1.0349e+002    1.2024e+001   8.61e+000    7.6697e+001    1.3028e+002 -0.67 1.00

Optimized: 110.42108 * x0 + 103.48806
";

      /*
       * Formula: y ~ a * x0 + b

      Parameters:
        Estimate Std. Error t value Pr(>|t|)    
      a   110.42      23.37   4.725 0.000811 ***
      b   103.49      12.02   8.607 6.17e-06 ***
      ---
      Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

      Residual standard error: 30.9 on 10 degrees of freedom

      Number of iterations to convergence: 1 
      Achieved convergence tolerance: 5.903e-09

      Waiting for profiling to be done...
            2.5%    97.5%
      a 58.34723 162.4949
      b 76.69743 130.2787
       */

      NlrFit("Puromycin.csv", "((110.42107672063611 * x0) + 103.48806186471387)", "0:11", "y", "Gaussian", expected);
    }

    [Test]
    public void PredictLinearPuromycinLaplace() {
      // exactly the same as for t-Profile!
      var expected = @"x0,y,residual,yPred,yPredLow,yPredHigh,isTrain,isTest
0.02,76,-29.6965,105.696,79.5928,131.8,1,0
0.02,47,-58.6965,105.696,79.5928,131.8,1,0
0.06,97,-13.1133,110.113,85.3094,134.917,1,0
0.06,107,-3.11333,110.113,85.3094,134.917,1,0
0.11,123,7.36562,115.634,92.2949,138.974,1,0
0.11,139,23.3656,115.634,92.2949,138.974,1,0
0.22,159,31.2193,127.781,106.868,148.694,1,0
0.22,152,24.2193,127.781,106.868,148.694,1,0
0.56,191,25.6761,165.324,142.513,188.135,1,0
0.56,201,35.6761,165.324,142.513,188.135,1,1
1.1,207,-17.9512,224.951,180.898,269.005,1,1
1.1,200,-24.9512,224.951,180.898,269.005,1,1
";
      NlrPredict("Puromycin.csv", "((110.42107672063611 * x0) + 103.48806186471387)", "0:11", "0:11", "y", "Gaussian", "LaplaceApproximation", expected);
    }

    [Test]
    public void PredictLinearPuromycinProfile() {
      // exactly the same as for LaplaceApproximation!
      var expected = @"x0,y,residual,yPred,yPredLow,yPredHigh,isTrain,isTest
0.02,76,-29.6965,105.696,79.5928,131.8,1,0
0.02,47,-58.6965,105.696,79.5928,131.8,1,0
0.06,97,-13.1133,110.113,85.3094,134.917,1,0
0.06,107,-3.11333,110.113,85.3094,134.917,1,0
0.11,123,7.36562,115.634,92.2949,138.974,1,0
0.11,139,23.3656,115.634,92.2949,138.974,1,0
0.22,159,31.2193,127.781,106.868,148.694,1,0
0.22,152,24.2193,127.781,106.868,148.694,1,0
0.56,191,25.6761,165.324,142.513,188.135,1,0
0.56,201,35.6761,165.324,142.513,188.135,1,1
1.1,207,-17.9512,224.951,180.898,269.005,1,1
1.1,200,-24.9512,224.951,180.898,269.005,1,1
";
      NlrPredict("Puromycin.csv", "((110.42107672063611 * x0) + 103.48806186471387)", "0:11", "0:11", "y", "Gaussian", "TProfile", expected);
    }
    #endregion


    #region logistic regression mammography
    [Test]
    public void FitMammography() {
      // verified against R logistic regression (same fitting results)
      // Deviance Residuals: 
      //     Min       1Q   Median       3Q      Max  
      // -2.6441  -0.5574  -0.2157   0.5882   3.1270  
      // 
      // Coefficients:
      //              Estimate Std. Error z value Pr(>|z|)    
      // (Intercept) -11.18081    1.05325 -10.616  < 2e-16 ***
      // BI_RADS       1.38377    0.17042   8.120 4.67e-16 ***
      // Age           0.04848    0.00760   6.379 1.78e-10 ***
      // Shape         0.52430    0.09753   5.376 7.61e-08 ***
      // Margin        0.35251    0.08099   4.352 1.35e-05 ***
      // Density      -0.06850    0.24032  -0.285    0.776    
      // ---
      // Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
      // 
      // (Dispersion parameter for binomial family taken to be 1)
      // 
      //     Null deviance: 1326.98  on 960  degrees of freedom
      // Residual deviance:  784.38  on 955  degrees of freedom
      // AIC: 796.38

      // exactly the same result as in R (using full Hessian for FisherInformationMatrix)
      var expected = @"p_opt: 1.38378e+000 4.84833e-002 5.24299e-001 3.52511e-001 -6.84851e-002 -1.11809e+001
Successful: True, NumIters: 3, NumFuncEvals: 43, NumJacEvals: 43
Deviance: 7.8438e+002  AICc: 796.5 BIC: 825.6 DL: 458  DL (lattice): 456 neg. Evidence: 389.84
Para       Estimate      Std. error     z Score          Lower          Upper Correlation matrix
    0    1.3838e+000    1.7042e-001   8.12e+000    1.0493e+000    1.7182e+000 1.00
    1    4.8483e-002    7.5999e-003   6.38e+000    3.3569e-002    6.3398e-002 -0.04 1.00
    2    5.2430e-001    9.7525e-002   5.38e+000    3.3291e-001    7.1569e-001 -0.08 0.02 1.00
    3    3.5251e-001    8.0993e-002   4.35e+000    1.9357e-001    5.1145e-001 -0.16 -0.08 -0.55 1.00
    4   -6.8485e-002    2.4032e-001  -2.85e-001   -5.4009e-001    4.0312e-001 -0.04 0.00 0.02 -0.10 1.00
    5   -1.1181e+001    1.0533e+000  -1.06e+001   -1.3248e+001   -9.1140e+000 -0.60 -0.38 -0.11 0.12 -0.62 1.00

Optimized: logistic(1.3837835 * BI_RADS + 0.048483267 * Age + 0.5242994 * Shape + 0.35251072 * Margin + -0.068485134 * Density + -11.180916)
";

      NlrFit("mammography.csv",
        "Logistic(((((((1.383783475779263 * BI_RADS) + (0.04848326870704811 * Age)) + (0.5242993934294974 * Shape)) + (0.35251072256819216 * Margin)) + (-0.06848513367625006 * Density)) + -11.180915607397608))", 
        "0:960", 
        "Severity",
        "Bernoulli",
        expected);
    }

    [Test]
    public void EvaluateMammography() {

      var expected = @"Deviance: 784.375 LogLik: -392.188 AIC: 796.38 AICc: 796.46 BIC: 825.58 DL: 458.24  DL_lattice: 455.78 neg. Evidence: 389.84 DoF: 6 m: 961
";

      NlrEvaluate("mammography.csv",
        "Logistic(((((((1.383783475779263 * BI_RADS) + (0.04848326870704811 * Age)) + (0.5242993934294974 * Shape)) + (0.35251072256819216 * Margin)) + (-0.06848513367625006 * Density)) + -11.180915607397608))",
        "0:960",
        "Severity",
        "Bernoulli",
        expected);
    }

    [Test]
    public void CrossValidateMammography() {

      var expected = @"CV_score: 1.2400e+001 CV_stdev: 4.8117e-002 CV_se: 1.5216e-002
";

      NlrCrossValidate("mammography.csv",
        "Logistic(((((((1.383783475779263 * BI_RADS) + (0.04848326870704811 * Age)) + (0.5242993934294974 * Shape)) + (0.35251072256819216 * Margin)) + (-0.06848513367625006 * Density)) + -11.180915607397608))",
        "0:960",
        "Severity",
        "Bernoulli",
        expected);
    }

    [Test]
    public void ProfileMammography() {
      // comparions with R:
      // confint(fit_glm, level = 0.95)
      // Waiting for profiling to be done...
      //                    2.5 %      97.5 %
      // (Intercept) -13.31163450 -9.17694180
      // BI_RADS       1.05898689  1.72652748
      // Age           0.03382864  0.06365745
      // Shape         0.33349326  0.71640682
      // Margin        0.19434656  0.51229183
      // Density      -0.53580610  0.40883818
      var expected = @"profile-based marginal confidence intervals (alpha=0.05)
p0    1.3838e+000    1.0586e+000    1.7270e+000
p1    4.8483e-002    3.3810e-002    6.3677e-002
p2    5.2430e-001    3.3325e-001    7.1665e-001
p3    3.5251e-001    1.9415e-001    5.1249e-001
p4   -6.8485e-002   -5.3640e-001    4.0944e-001
p5   -1.1181e+001   -1.3314e+001   -9.1744e+000
";
      // -> very close match
      NlrProfile("mammography.csv",
        "Logistic(((((((1.38378347577933 * BI_RADS) + (0.04848326870687731 * Age)) + (0.5242993934295488 * Shape)) + (0.3525107225682143 * Margin)) + (-0.06848513367624065 * Density)) + -11.180915607397594))",
        "0:960",
        "Severity",
        "Bernoulli",
        expected);
    }

    [Test]
    public void PredictMammographyLaplace() {
      var expected = @"BI_RADS,Age,Shape,Margin,Density,y,residual,yPred,yPredLow,yPredHigh,isTrain,isTest
5,67,3,5,3,1,0.107515,0.892485,0.85419,0.930779,1,0
4,43,1,1,3,1,0.947332,0.0526678,0.0318619,0.0734736,1,0
5,58,4,5,3,1,0.0993626,0.900637,0.870021,0.931254,1,0
4,28,1,1,3,0,-0.0261631,0.0261631,0.012036,0.0402901,1,0
5,74,1,5,3,1,0.196682,0.803318,0.695273,0.911363,1,0
4,65,1,3,3,0,-0.246384,0.246384,0.167065,0.325702,1,0
4,70,3,3,3,0,-0.543148,0.543148,0.470361,0.615935,1,0
5,42,1,3,3,0,-0.299575,0.299575,0.188238,0.410911,1,0
5,57,1,5,3,1,0.358255,0.641745,0.492258,0.791232,1,0
5,60,3,5,1,1,0.128536,0.871464,0.757993,0.984935,1,0
5,76,1,4,3,1,0.240194,0.759806,0.649676,0.869936,1,0
3,42,2,1,3,1,0.978068,0.0219324,0.0104774,0.0333874,1,0
4,64,1,3,3,0,-0.237492,0.237492,0.160832,0.314153,1,0
4,36,3,1,2,0,-0.107941,0.107941,0.0484894,0.167393,1,0
4,60,2,1,2,0,-0.186541,0.186541,0.108367,0.264714,1,0
4,54,1,1,3,0,-0.0865639,0.0865639,0.0570232,0.116105,1,0
3,52,3,4,3,0,-0.150466,0.150466,0.0831465,0.217786,1,0
4,59,2,1,3,1,0.830562,0.169438,0.122467,0.216409,1,0
4,54,1,1,3,1,0.913436,0.0865639,0.0570232,0.116105,1,0
4,40,1,3,3,0,-0.0886626,0.0886626,0.0495763,0.127749,1,0
";
      NlrPredict("mammography.csv",
        "Logistic(((((((1.383783475779263 * BI_RADS) + (0.04848326870704811 * Age)) + (0.5242993934294974 * Shape)) + (0.35251072256819216 * Margin)) + (-0.06848513367625006 * Density)) + -11.180915607397608))",
        "0:960", "0:19", "Severity", "Bernoulli", "LaplaceApproximation", expected);
    }
    [Test]
    public void PredictMammographyProfile() {
      // verified against R with logistic regression
      // mammo_predictions = predict(fit_glm, mammography[1:10,], se.fit = TRUE) # type = "link" as default
      // z_crit = qnorm(0.975)
      // mammo_pred_low = mammo_predictions$fit-1 * z_crit * mammo_predictions$se.fit
      // mammo_pred_high = mammo_predictions$fit + 1 * z_crit * mammo_predictions$se.fit
      // 
      // data.frame(prob=predict(fit_glm, mammography[1:10,], se.fit = FALSE, type = "response"), 
      //           low=boot::inv.logit(mammo_pred_low), 
      //           high=boot::inv.logit(mammp_pred_high))
      //    prob        low       high
      //  1  0.89248381 0.84784386 0.92518287
      //  2  0.05266805 0.03536213 0.07776061
      //  3  0.90063675 0.86560368 0.92730468
      //  4  0.02616331 0.01520743 0.04465424
      //  5  0.80331579 0.67352705 0.88993877
      //  6  0.24638378 0.17586304 0.33373184
      //  7  0.54314781 0.47005149 0.61443203
      //  8  0.29957358 0.20112801 0.42082212
      //  9  0.64174304 0.48339643 0.77422326
      //  10 0.87146628 0.71141575 0.94910244
      var expected = @"BI_RADS,Age,Shape,Margin,Density,y,residual,yPred,yPredLow,yPredHigh,isTrain,isTest
5,67,3,5,3,1,0.107515,0.892485,0.848937,0.925925,1,0
4,43,1,1,3,1,0.947332,0.0526678,0.0348023,0.0767389,1,0
5,58,4,5,3,1,0.0993626,0.900637,0.86651,0.927942,1,0
4,28,1,1,3,0,-0.0261631,0.0261631,0.0148872,0.0438548,1,0
5,74,1,5,3,1,0.196682,0.803318,0.675313,0.891164,1,0
4,65,1,3,3,0,-0.246384,0.246384,0.175237,0.333312,1,0
4,70,3,3,3,0,-0.543148,0.543148,0.470008,0.61471,1,0
5,42,1,3,3,0,-0.299575,0.299575,0.200605,0.420816,1,0
5,57,1,5,3,1,0.358255,0.641745,0.484193,0.775557,1,0
5,60,3,5,1,1,0.128536,0.871464,0.711207,0.949378,1,0
5,76,1,4,3,1,0.240194,0.759806,0.635682,0.853934,1,0
3,42,2,1,3,1,0.978068,0.0219324,0.0127554,0.0362594,1,0
4,64,1,3,3,0,-0.237492,0.237492,0.168852,0.32174,1,0
4,36,3,1,2,0,-0.107941,0.107941,0.0604077,0.181415,1,0
4,60,2,1,2,0,-0.186541,0.186541,0.118983,0.275093,1,0
4,54,1,1,3,0,-0.0865639,0.0865639,0.0605089,0.11981,1,0
3,52,3,4,3,0,-0.150466,0.150466,0.0937662,0.228687,1,0
4,59,2,1,3,1,0.830562,0.169438,0.126462,0.22029,1,0
4,54,1,1,3,1,0.913436,0.0865639,0.0605089,0.11981,1,0
4,40,1,3,3,0,-0.0886626,0.0886626,0.0559771,0.135132,1,0
";
      NlrPredict("mammography.csv",
        "Logistic(((((((1.383783475779263 * BI_RADS) + (0.04848326870704811 * Age)) + (0.5242993934294974 * Shape)) + (0.35251072256819216 * Margin)) + (-0.06848513367625006 * Density)) + -11.180915607397608))",
        "0:960", "0:19", "Severity", "Bernoulli", "TProfile", expected);
    }

    [Test]
    public void RankMammography() {
      
      var expected = @"Num param: 6 rank: 6 log10_K(J): 2.89637 log10_K(J_rank): 2.89637
";
      NlrRank("mammography.csv",
        "Logistic(((((((1.383783475779263 * BI_RADS) + (0.04848326870704811 * Age)) + (0.5242993934294974 * Shape)) + (0.35251072256819216 * Margin)) + (-0.06848513367625006 * Density)) + -11.180915607397608))",
        "0:960", "Severity", "Bernoulli", expected);
    }

    [Test]
    public void VariableImporanceMammography() {

      var expected = @"Variable    SSR_ratio
BI_RADS     1.18
Age         1.04
Shape       1.04
Margin      1.03
Density     1.00
";
      NlrVariableImpact("mammography.csv",
        "Logistic(((((((1.383783475779263 * BI_RADS) + (0.04848326870704811 * Age)) + (0.5242993934294974 * Shape)) + (0.35251072256819216 * Margin)) + (-0.06848513367625006 * Density)) + -11.180915607397608))",
        "0:960", "Severity", "Bernoulli", expected);
    }
    [Test]
    public void NestedMammography() {

      var expected = @"
Deviance_Factor,numPar,AICc,dAICc,BIC,dBIC,Model
1.0000e+000,6,7.9646e+002,0.0000e+000,8.2558e+002,0.0000e+000,logistic(1.3837835 * BI_RADS + 0.048483267 * Age + 0.5242994 * Shape + 0.35251072 * Margin + -0.068485134 * Density + -11.180916)
1.0001e+000,5,7.9452e+002,-1.9442e+000,8.1880e+002,-6.7870e+000,logistic(BI_RADS * 1.3822883 + Age * 0.048492866 + Shape * 0.5248007 + Margin * 0.35034412 + -11.369226)
";
      NlrNested("mammography.csv",
        "Logistic(((((((1.383783475779263 * BI_RADS) + (0.04848326870704811 * Age)) + (0.5242993934294974 * Shape)) + (0.35251072256819216 * Margin)) + (-0.06848513367625006 * Density)) + -11.180915607397608))",
        "0:960", "Severity", "Bernoulli", expected);
    }
    [Test]
    public void SubtreesMammography() {

      var expected = @"SSR_factor  ΔDoF   ΔSSR        s2Extra     fRatio      p value    ΔAICc       ΔBIC        ΔDL         MSE         sub-expression
1.176e+000  1      2.106e+001  2.106e+001  1.6829e+002 5.551e-016 88.2        83.4        28.4        1.46e-001   p[0] 
1.176e+000  1      2.106e+001  2.106e+001  1.6829e+002 5.551e-016 88.2        83.4        28.4        1.46e-001   x[0] 
1.176e+000  1      2.106e+001  2.106e+001  1.6829e+002 5.551e-016 88.2        83.4        28.4        1.46e-001   (p[0] * x[0]) 
1.042e+000  1      5.017e+000  5.017e+000  4.0098e+001 3.712e-010 42.6        37.7        8.6         1.30e-001   p[1] 
1.042e+000  1      5.017e+000  5.017e+000  4.0098e+001 3.712e-010 42.6        37.7        8.6         1.30e-001   x[1] 
1.042e+000  1      5.017e+000  5.017e+000  4.0098e+001 3.712e-010 42.6        37.7        8.6         1.30e-001   (p[1] * x[1]) 
1.265e+000  2      3.162e+001  1.581e+001  1.2635e+002 5.551e-016 151.6       141.9       47.9        1.57e-001   ((p[0] * x[0]) + (p[1] * x[1])) 
1.039e+000  1      4.706e+000  4.706e+000  3.7615e+001 1.260e-009 26.7        21.9        3.1         1.29e-001   p[2] 
1.039e+000  1      4.706e+000  4.706e+000  3.7615e+001 1.260e-009 26.7        21.9        3.1         1.29e-001   x[2] 
1.039e+000  1      4.706e+000  4.706e+000  3.7615e+001 1.260e-009 26.7        21.9        3.1         1.29e-001   (p[2] * x[2]) 
1.383e+000  3      4.581e+001  1.527e+001  1.2204e+002 5.551e-016 208.5       194.0       65.6        1.72e-001   (((p[0] * x[0]) + (p[1] * x[1])) + (p[2] * x[2])) 
1.026e+000  1      3.071e+000  3.071e+000  2.4548e+001 8.569e-007 17.1        12.2        -4.6        1.28e-001   p[3] 
1.026e+000  1      3.071e+000  3.071e+000  2.4548e+001 8.569e-007 17.1        12.2        -4.6        1.28e-001   x[3] 
1.026e+000  1      3.071e+000  3.071e+000  2.4548e+001 8.569e-007 17.1        12.2        -4.6        1.28e-001   (p[3] * x[3]) 
1.994e+000  4      1.187e+002  2.968e+001  2.3724e+002 5.551e-016 531.6       512.2       217.8       2.48e-001   ((((p[0] * x[0]) + (p[1] * x[1])) + (p[2] * x[2])) + (p[3] * x[3])) 
1.000e+000  1      7.986e-003  7.986e-003  6.3829e-002 0.000e+000 -1.9        -6.8        -13.4       1.24e-001   p[4] 
1.000e+000  1      7.986e-003  7.986e-003  6.3826e-002 0.000e+000 -1.9        -6.8        -13.4       1.24e-001   x[4] 
1.000e+000  1      7.986e-003  7.986e-003  6.3826e-002 0.000e+000 -1.9        -6.8        -13.4       1.24e-001   (p[4] * x[4]) 
2.000e+000  5      1.195e+002  2.389e+001  1.9096e+002 5.551e-016 532.5       508.3       208.1       2.49e-001   (((((p[0] * x[0]) + (p[1] * x[1])) + (p[2] * x[2])) + (p[3] * x[3])) + (p[4] * x[4])) 
1.241e+000  1      2.874e+001  2.874e+001  2.2973e+002 5.551e-016 145.9       141.1       65.2        1.54e-001   p[5] 
2.000e+000  5      1.195e+002  2.389e+001  1.9096e+002 5.551e-016 532.5       508.3       208.1       2.49e-001   ((((((p[0] * x[0]) + (p[1] * x[1])) + (p[2] * x[2])) + (p[3] * x[3])) + (p[4] * x[4])) + p[5]) 
2.000e+000  5      1.195e+002  2.389e+001  1.9096e+002 5.551e-016 532.5       508.3       208.1       2.49e-001   Logistic(((((((p[0] * x[0]) + (p[1] * x[1])) + (p[2] * x[2])) + (p[3] * x[3])) + (p[4] * x[4])) + p[5])) 
";
      NlrSubtrees("mammography.csv",
        "Logistic(((((((1.383783475779263 * BI_RADS) + (0.04848326870704811 * Age)) + (0.5242993934294974 * Shape)) + (0.35251072256819216 * Margin)) + (-0.06848513367625006 * Density)) + -11.180915607397608))",
        "0:960", "Severity", "Bernoulli", expected);
    }
    #endregion

    internal void NlrFit(string dataFilename, string model, string trainRange, string target, string likelihood, string expected) {
      RunConsoleTest(() => Program.Main(new[] { "fit", "--dataset", dataFilename, "--model", model, "--train", trainRange, "--target", target, "--likelihood", likelihood}), expected);
    }
    internal void NlrPredict(string dataFilename, string model, string trainRange, string predRange, string target, string likelihood, string intervalType, string expected) {
      RunConsoleTest(() => Program.Main(new[] { "predict", "--dataset", dataFilename, "--model", model, "--target", target, "--likelihood", likelihood, "--train", trainRange, "--range", predRange, "--interval", intervalType }), expected);
    }
    internal void NlrProfile(string dataFilename, string model, string trainRange, string target, string likelihood, string expected) {
      RunConsoleTest(() => Program.Main(new[] { "profile", "--dataset", dataFilename, "--model", model, "--target", target, "--likelihood", likelihood, "--train", trainRange }), expected);
    }
    internal void NlrEvaluate(string dataFilename, string model, string range, string target, string likelihood, string expected) {
      RunConsoleTest(() => Program.Main(new[] { "evaluate", "--dataset", dataFilename, "--model", model, "--target", target, "--likelihood", likelihood, "--range", range }), expected);
    }
    internal void NlrCrossValidate(string dataFilename, string model, string range, string target, string likelihood, string expected) {
      RunConsoleTest(() => Program.Main(new[] { "crossvalidate", "--dataset", dataFilename, "--model", model, "--target", target, "--likelihood", likelihood, "--train", range, "--folds", "10", "--shuffle", "--seed","1234"}), expected);
    }
    internal void NlrRank(string dataFilename, string model, string trainRange, string target, string likelihood, string expected) {
      RunConsoleTest(() => Program.Main(new[] { "rank", "--dataset", dataFilename, "--model", model, "--target", target, "--likelihood", likelihood, "--train", trainRange}), expected);
    }

    internal void NlrNested(string dataFilename, string model, string trainRange, string target, string likelihood, string expected) {
      RunConsoleTest(() => Program.Main(new[] { "nested", "--dataset", dataFilename, "--model", model, "--target", target, "--likelihood", likelihood, "--train", trainRange }), expected);
    }

    internal void NlrVariableImpact(string dataFilename, string model, string trainRange, string target, string likelihood, string expected) {
      RunConsoleTest(() => Program.Main(new[] { "variableimpact", "--dataset", dataFilename, "--model", model, "--target", target, "--likelihood", likelihood, "--train", trainRange }), expected);
    }

    internal void NlrSubtrees(string dataFilename, string model, string trainRange, string target, string likelihood, string expected) {
      RunConsoleTest(() => Program.Main(new[] { "subtrees", "--dataset", dataFilename, "--model", model, "--target", target, "--likelihood", likelihood, "--train", trainRange }), expected);
    }
    internal void RunConsoleTest(Action action, string expected) {
      var randFilename = Path.GetRandomFileName();
      try {
        using (var memWriter = new StreamWriter(randFilename)) {
          var origOut = System.Console.Out;
          System.Console.SetOut(memWriter);
          try {
            action();
          } finally {
            System.Console.SetOut(origOut);
          }
        }
        var actual = File.ReadAllText(randFilename);
        System.Console.WriteLine(actual);
        ClassicAssert.AreEqual(expected.ReplaceLineEndings(), actual.ReplaceLineEndings());
      } finally {
        if (File.Exists(randFilename))
          File.Delete(randFilename);
      }
    }
  }
}

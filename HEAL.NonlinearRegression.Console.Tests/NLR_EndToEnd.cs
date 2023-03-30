﻿
using NUnit.Framework;
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

      // standard error and z-score are the same as in R
      // Laplace approximation lower and upper bounds are close to confidence intervals in R
      var expected = @"p_opt: 6.41213e-002 2.12684e+002
Successful: True, NumIters: 3, NumFuncEvals: 44, NumJacEvals: 0
SSR: 1.1954e+003  s: 1.0934e+001 AICc: 19.0 BIC: 17.5 MDL: 21.5
Para       Estimate      Std. error     z Score          Lower          Upper Correlation matrix
    0    6.4121e-002    8.7112e-003   7.36e+000    4.4711e-002    8.3531e-002 1.00
    1    2.1268e+002    7.1607e+000   2.97e+001    1.9673e+002    2.2864e+002 0.78 1.00

Optimized: ((x0 / (0.06412128166090875 + x0)) * 212.68374312341493)
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
      var expected = @"SSR: 1195.4488144393595 MSE: 99.62073453661328 RMSE: 9.981018712366653 NMSE: 0.03873916985979826 R2: 0.9612608301402017 LogLik: -4.999999999999999 AIC: 15.999999999999998 AICc: 19 BIC: 17.454719949363998 MDL: 21.516073714821182 MDL(freq): 19.710008327782482 DoF: 2
";
      NlrEvaluate("Puromycin.csv", "x0 / (0.06412128165180965 + x0) * 212.68374312341493", "0:11", "y", "Gaussian", expected);
    }

    [Test]
    public void PredictPuromycinLaplace() {
      var expected = @"x0,y,residual,yPred,yPredLow,yPredHigh,isTrain,isTest
0.02,76,25.434022182225043,50.56597781777496,41.95791290893474,59.17404272661518,1,0
0.02,47,-3.5659778177749573,50.56597781777496,41.95791290893474,59.17404272661518,1,0
0.06,97,-5.810931507033644,102.81093150703364,92.10677233386393,113.51509068020336,1,0
0.06,107,4.189068492966356,102.81093150703364,92.10677233386393,113.51509068020336,1,0
0.11,123,-11.361587052503495,134.3615870525035,125.1259054263397,143.5972686786673,1,0
0.11,139,4.638412947496505,134.3615870525035,125.1259054263397,143.5972686786673,1,0
0.22,159,-5.6846839970068,164.6846839970068,156.8174930318183,172.5518749621953,1,0
0.22,152,-12.6846839970068,164.6846839970068,156.8174930318183,172.5518749621953,1,0
0.56,191,0.16706472152927176,190.83293527847073,180.5942013515739,201.07166920536756,1,0
0.56,201,10.167064721529272,190.83293527847073,180.5942013515739,201.07166920536756,1,1
1.1,207,6.031148110302098,200.9688518896979,188.60791098183105,213.32979279756475,1,1
1.1,200,-0.9688518896979019,200.9688518896979,188.60791098183105,213.32979279756475,1,1
";
      NlrPredict("Puromycin.csv", "x0 / (0.06412128165180965 + x0) * 212.68374312341493", "0:11", "0:11", "y", "Gaussian", "LaplaceApproximation", expected);
    }

    [Test]
    public void PredictPuromycinProfile() {
      var expected = @"x0,y,residual,yPred,yPredLow,yPredHigh,isTrain,isTest
0.02,76,25.434022182225043,50.56597781777496,42.25914848662796,60.36645187030026,1,0
0.02,47,-3.5659778177749573,50.56597781777496,42.25914848662796,60.36645187030026,1,0
0.06,97,-5.810931507033644,102.81093150703364,91.75750014064471,113.97253545514351,1,0
0.06,107,4.189068492966356,102.81093150703364,91.75750014064471,113.97253545514351,1,0
0.11,123,-11.361587052503495,134.3615870525035,124.62818491340477,143.62124256533426,1,0
0.11,139,4.638412947496505,134.3615870525035,124.62818491340477,143.62124256533426,1,0
0.22,159,-5.6846839970068,164.6846839970068,156.7762431039124,172.54972638895924,1,0
0.22,152,-12.6846839970068,164.6846839970068,156.7762431039124,172.54972638895924,1,0
0.56,191,0.16706472152927176,190.83293527847073,180.46905231552648,201.19666313895144,1,0
0.56,201,10.167064721529272,190.83293527847073,180.46905231552648,201.19666313895144,1,1
1.1,207,6.031148110302098,200.9688518896979,188.5081808414797,213.77447974160089,1,1
1.1,200,-0.9688518896979019,200.9688518896979,188.5081808414797,213.77447974160089,1,1
";
      NlrPredict("Puromycin.csv", "x0 / (0.06412128165180965 + x0) * 212.68374312341493", "0:11", "0:11", "y", "Gaussian", "TProfile", expected);
    }
    #endregion

    #region linear Puromycin (to compare LaplaceApproximation = tProfile for linear Gaussian models)
    [Test]
    public void FitLinearPuromycin() {
      var expected = @"p_opt: 1.10421e+002 1.03488e+002
Successful: True, NumIters: 2, NumFuncEvals: 10, NumJacEvals: 0
SSR: 9.5471e+003  s: 3.0898e+001 AICc: 19.0 BIC: 17.5 MDL: 15.1
Para       Estimate      Std. error     z Score          Lower          Upper Correlation matrix
    0    1.1042e+002    2.3371e+001   4.72e+000    5.8347e+001    1.6249e+002 1.00
    1    1.0349e+002    1.2024e+001   8.61e+000    7.6697e+001    1.3028e+002 -0.67 1.00

Optimized: ((110.42107672063618 * x0) + 103.48806186471386)
";
      NlrFit("Puromycin.csv", "((110.42107672063611 * x0) + 103.48806186471387)", "0:11", "y", "Gaussian", expected);
    }

    [Test]
    public void PredictLinearPuromycinLaplace() {
      // exactly the same as for t-Profile!
      var expected = @"x0,y,residual,yPred,yPredLow,yPredHigh,isTrain,isTest
0.02,76,-29.696483399126592,105.69648339912659,79.5928197722846,131.80014702596858,1,0
0.02,47,-58.69648339912659,105.69648339912659,79.5928197722846,131.80014702596858,1,0
0.06,97,-13.113326467952035,110.11332646795204,85.30937235031085,134.91728058559323,1,0
0.06,107,-3.113326467952035,110.11332646795204,85.30937235031085,134.91728058559323,1,0
0.11,123,7.365619696016154,115.63438030398385,92.29487883245449,138.9738817755132,1,0
0.11,139,23.365619696016154,115.63438030398385,92.29487883245449,138.9738817755132,1,0
0.22,159,31.219301256746178,127.78069874325382,106.86779360020266,148.69360388630497,1,0
0.22,152,24.219301256746178,127.78069874325382,106.86779360020266,148.69360388630497,1,0
0.56,191,25.676135171729868,165.32386482827013,142.5131839174026,188.13454573913765,1,0
0.56,201,35.67613517172987,165.32386482827013,142.5131839174026,188.13454573913765,1,1
1.1,207,-17.95124625741363,224.95124625741363,180.89778117957925,269.00471133524803,1,1
1.1,200,-24.95124625741363,224.95124625741363,180.89778117957925,269.00471133524803,1,1
";
      NlrPredict("Puromycin.csv", "((110.42107672063611 * x0) + 103.48806186471387)", "0:11", "0:11", "y", "Gaussian", "LaplaceApproximation", expected);
    }

    [Test]
    public void PredictLinearPuromycinProfile() {
      // exactly the same as for LaplaceApproximation!
      var expected = @"x0,y,residual,yPred,yPredLow,yPredHigh,isTrain,isTest
0.02,76,-29.696483399126592,105.69648339912659,79.59281977228457,131.80014702596856,1,0
0.02,47,-58.69648339912659,105.69648339912659,79.59281977228457,131.80014702596856,1,0
0.06,97,-13.113326467952035,110.11332646795204,85.30937235031082,134.9172805855932,1,0
0.06,107,-3.113326467952035,110.11332646795204,85.30937235031082,134.9172805855932,1,0
0.11,123,7.365619696016154,115.63438030398385,92.29487883245447,138.97388177551318,1,0
0.11,139,23.365619696016154,115.63438030398385,92.29487883245447,138.97388177551318,1,0
0.22,159,31.219301256746178,127.78069874325382,106.86779360020265,148.69360388630497,1,0
0.22,152,24.219301256746178,127.78069874325382,106.86779360020265,148.69360388630497,1,0
0.56,191,25.676135171729868,165.32386482827013,142.5131839174026,188.13454573913762,1,0
0.56,201,35.67613517172987,165.32386482827013,142.5131839174026,188.13454573913762,1,1
1.1,207,-17.95124625741363,224.95124625741363,180.89778117957925,269.00471133524803,1,1
1.1,200,-24.95124625741363,224.95124625741363,180.89778117957925,269.00471133524803,1,1
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

      var expected = @"Deviance: 784.3750837455059 LogLik: -392.18754187275294 AIC: 808.3750837455059 AICc: 808.7041976695565 BIC: 866.7907766531494 MDL: 455.9073697707563 MDL(freq): 449.97561863153 DoF: 6
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

      var expected = @"CV_score: 6.9391e-001 CV_stdev: 5.3068e-003 CV_se: 1.6782e-003
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
p0    1.3859e+000    1.0586e+000    1.7270e+000
p1    4.8525e-002    3.3809e-002    6.3677e-002
p2    5.2442e-001    3.3324e-001    7.1666e-001
p3    3.5240e-001    1.9414e-001    5.1250e-001
p4   -6.6586e-002   -5.3641e-001    4.0946e-001
p5   -1.1198e+001   -1.3314e+001   -9.1743e+000
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
5,67,3,5,3,1,0.10751521187441859,0.8924847881255814,0.8531683937339792,0.9318011825171836,1,0
4,43,1,1,3,1,0.9473322449203974,0.05266775507960262,0.03158880082505203,0.07374670933415321,1,0
5,58,4,5,3,1,0.09936258583801727,0.9006374141619827,0.8709941848706296,0.9302806434533358,1,0
4,28,1,1,3,0,-0.02616305750483777,0.02616305750483777,0.011374100694390555,0.04095201431528498,1,0
5,74,1,5,3,1,0.19668219371154416,0.8033178062884558,0.686159200416027,0.9204764121608847,1,0
4,65,1,3,3,0,-0.24638372913929282,0.24638372913929282,0.1620569669143745,0.33071049136421116,1,0
4,70,3,3,3,0,-0.5431479146698254,0.5431479146698254,0.47639582549868503,0.6099000038409657,1,0
5,42,1,3,3,0,-0.299574513189939,0.299574513189939,0.19152151715954008,0.4076275092203379,1,0
5,57,1,5,3,1,0.35825508046733845,0.6417449195326616,0.48106112206791374,0.8024287169974094,1,0
5,60,3,5,1,1,0.12853611407770482,0.8714638859222952,0.7615750060107025,0.9813527658338879,1,0
5,76,1,4,3,1,0.24019385030590246,0.7598061496940975,0.643394571435092,0.876217727953103,1,0
3,42,2,1,3,1,0.9780676003581101,0.02193239964188982,0.012006290208276357,0.031858509075503284,1,0
4,64,1,3,3,0,-0.23749248435765508,0.23749248435765508,0.15588558247118373,0.3190993862441264,1,0
4,36,3,1,2,0,-0.10794120010644159,0.10794120010644159,0.044850529819888574,0.1710318703929946,1,0
4,60,2,1,2,0,-0.18654071132515063,0.18654071132515063,0.10806909550918267,0.26501232714111855,1,0
4,54,1,1,3,0,-0.08656390328372499,0.08656390328372499,0.05762353274647146,0.11550427382097851,1,0
3,52,3,4,3,0,-0.1504661598736393,0.1504661598736393,0.1037138505380333,0.1972184692092453,1,0
4,59,2,1,3,1,0.8305618978661792,0.1694381021338208,0.12380519871839601,0.21507100554924558,1,0
4,54,1,1,3,1,0.913436096716275,0.08656390328372499,0.05762353274647146,0.11550427382097851,1,0
4,40,1,3,3,0,-0.08866261736270638,0.08866261736270638,0.04612245664873757,0.13120277807667519,1,0
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
5,67,3,5,3,1,0.10751521187441859,0.8924847881255814,0.8489371199128735,0.9259246294717256,1,0
4,43,1,1,3,1,0.9473322449203974,0.05266775507960262,0.0348023341223409,0.07673885805343308,1,0
5,58,4,5,3,1,0.09936258583801727,0.9006374141619827,0.8665102304377208,0.9279421646894344,1,0
4,28,1,1,3,0,-0.02616305750483777,0.02616305750483777,0.014887193483585123,0.04385479332360548,1,0
5,74,1,5,3,1,0.19668219371154416,0.8033178062884558,0.6753125670911367,0.891163594930779,1,0
4,65,1,3,3,0,-0.24638372913929282,0.24638372913929282,0.17523693840741583,0.3333117080959789,1,0
4,70,3,3,3,0,-0.5431479146698254,0.5431479146698254,0.47000784231950987,0.614709748036738,1,0
5,42,1,3,3,0,-0.299574513189939,0.299574513189939,0.20060492458363446,0.42081623118546013,1,0
5,57,1,5,3,1,0.35825508046733845,0.6417449195326616,0.4841932033676725,0.7755572895945753,1,0
5,60,3,5,1,1,0.12853611407770482,0.8714638859222952,0.7112068105579976,0.9493778435222229,1,0
5,76,1,4,3,1,0.24019385030590246,0.7598061496940975,0.6356819454930842,0.8539336659656889,1,0
3,42,2,1,3,1,0.9780676003581101,0.02193239964188982,0.012755389589274071,0.03625943514208149,1,0
4,64,1,3,3,0,-0.23749248435765508,0.23749248435765508,0.16885205661150118,0.3217404771552057,1,0
4,36,3,1,2,0,-0.10794120010644159,0.10794120010644159,0.06040771565107439,0.18141548515598813,1,0
4,60,2,1,2,0,-0.18654071132515063,0.18654071132515063,0.11898340402395993,0.2750928472753196,1,0
4,54,1,1,3,0,-0.08656390328372499,0.08656390328372499,0.06050888004962476,0.11981019025184995,1,0
3,52,3,4,3,0,-0.1504661598736393,0.1504661598736393,0.09376620340061906,0.22868722603565672,1,0
4,59,2,1,3,1,0.8305618978661792,0.1694381021338208,0.1264615485565289,0.22029042459307405,1,0
4,54,1,1,3,1,0.913436096716275,0.08656390328372499,0.06050888004962476,0.11981019025184995,1,0
4,40,1,3,3,0,-0.08866261736270638,0.08866261736270638,0.05597707388930403,0.13513165333180607,1,0
";
      NlrPredict("mammography.csv",
        "Logistic(((((((1.383783475779263 * BI_RADS) + (0.04848326870704811 * Age)) + (0.5242993934294974 * Shape)) + (0.35251072256819216 * Margin)) + (-0.06848513367625006 * Density)) + -11.180915607397608))",
        "0:960", "0:19", "Severity", "Bernoulli", "TProfile", expected);
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
        Assert.AreEqual(expected, actual);
      } finally {
        if (File.Exists(randFilename))
          File.Delete(randFilename);
      }
    }
  }
}

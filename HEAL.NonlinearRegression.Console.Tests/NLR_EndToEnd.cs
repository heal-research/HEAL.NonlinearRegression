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
      var expected = @"SSR: 1195.4488144393595 MSE: 99.62073453661328 RMSE: 9.981018712366653 NMSE: 0.03873916985979826 R2: 0.9612608301402017 LogLik: -4.999999999999999 AIC: 15.999999999999998 AICc: 19 BIC: 17.454719949363998 MDL: 21.49568808877963 MDL(freq): 19.68962270174093 DoF: 2
";
      NlrEvaluate("Puromycin.csv", "x0 / (0.06412128165180965 + x0) * 212.68374312341493", "0:11", "y", "Gaussian", expected);
    }

    [Test]
    public void PredictPuromycinLaplace() {
      var expected = @"x0,y,residual,yPred,yPredLow,yPredHigh,isTrain,isTest
0.02,76,25.434022182225043,50.56597781777496,41.54374725630074,59.588208379249174,1,0
0.02,47,-3.5659778177749573,50.56597781777496,41.54374725630074,59.588208379249174,1,0
0.06,97,-5.810931507033644,102.81093150703364,91.6610232898636,113.96083972420368,1,0
0.06,107,4.189068492966356,102.81093150703364,91.6610232898636,113.96083972420368,1,0
0.11,123,-11.361587052503495,134.3615870525035,124.8660730759562,143.8571010290508,1,0
0.11,139,4.638412947496505,134.3615870525035,124.8660730759562,143.8571010290508,1,0
0.22,159,-5.6846839970068,164.6846839970068,156.801757233156,172.5676107608576,1,0
0.22,152,-12.6846839970068,164.6846839970068,156.801757233156,172.5676107608576,1,0
0.56,191,0.16706472152927176,190.83293527847073,180.4656415384385,201.20022901850297,1,0
0.56,201,10.167064721529272,190.83293527847073,180.4656415384385,201.20022901850297,1,1
1.1,207,6.031148110302098,200.9688518896979,188.33454413738727,213.60315964200854,1,1
1.1,200,-0.9688518896979019,200.9688518896979,188.33454413738727,213.60315964200854,1,1
";
      NlrPredict("Puromycin.csv", "x0 / (0.06412128165180965 + x0) * 212.68374312341493", "0:11", "0:11", "y", "Gaussian", "LaplaceApproximation", expected);
    }

    [Test]
    public void PredictPuromycinProfile() {
      var expected = @"x0,y,residual,yPred,yPredLow,yPredHigh,isTrain,isTest
0.02,76,25.434022182225043,50.56597781777496,42.25914848662748,60.36645187032445,1,0
0.02,47,-3.5659778177749573,50.56597781777496,42.25914848662748,60.36645187032445,1,0
0.06,97,-5.810931507033644,102.81093150703364,91.75750014065308,113.9725354551382,1,0
0.06,107,4.189068492966356,102.81093150703364,91.75750014065308,113.9725354551382,1,0
0.11,123,-11.361587052503495,134.3615870525035,124.62818491339584,143.62124256533073,1,0
0.11,139,4.638412947496505,134.3615870525035,124.62818491339584,143.62124256533073,1,0
0.22,159,-5.6846839970068,164.6846839970068,156.77624310391266,172.5497263889593,1,0
0.22,152,-12.6846839970068,164.6846839970068,156.77624310391266,172.5497263889593,1,0
0.56,191,0.16706472152927176,190.83293527847073,180.46905231552628,201.19666313895115,1,0
0.56,201,10.167064721529272,190.83293527847073,180.46905231552628,201.19666313895115,1,1
1.1,207,6.031148110302098,200.9688518896979,188.50818084147986,213.77447974160083,1,1
1.1,200,-0.9688518896979019,200.9688518896979,188.50818084147986,213.77447974160083,1,1
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

      var expected = @"Deviance: 784.3750837455059 LogLik: -392.18754187275294 AIC: 808.3750837455059 AICc: 808.7041976695565 BIC: 866.7907766531494 MDL: 455.9972040835651 MDL(freq): 450.0654529443388 DoF: 6
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

      var expected = @"CV_score: 6.9391e-001 CV_stdev: 5.3066e-003 CV_se: 1.6781e-003
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
5,67,3,5,3,1,0.10751520794558234,0.8924847920544177,0.8541902795906103,0.930779304518225,1,0
4,43,1,1,3,1,0.9473322436260281,0.05266775637397194,0.03186192391034221,0.07347358883760166,1,0
5,58,4,5,3,1,0.09936258277539556,0.9006374172246044,0.8700211646275459,0.931253669821663,1,0
4,28,1,1,3,0,-0.026163057940222542,0.026163057940222542,0.012036018086062842,0.040290097794382244,1,0
5,74,1,5,3,1,0.19668218614447008,0.8033178138555299,0.6952727987839018,0.911362828927158,1,0
4,65,1,3,3,0,-0.24638373666189764,0.24638373666189764,0.1670652826805335,0.3257021906432618,1,0
4,70,3,3,3,0,-0.5431479247562248,0.5431479247562248,0.47036107551434514,0.6159347739981044,1,0
5,42,1,3,3,0,-0.2995745189432531,0.2995745189432531,0.1882380321425159,0.4109110057439903,1,0
5,57,1,5,3,1,0.3582550717633153,0.6417449282366847,0.4922580035621686,0.7912318529112008,1,0
5,60,3,5,1,1,0.12853610998735443,0.8714638900126456,0.7579927489407225,0.9849350310845686,1,0
5,76,1,4,3,1,0.24019384149455036,0.7598061585054496,0.6496764323441024,0.8699358846667968,1,0
3,42,2,1,3,1,0.9780675998548116,0.021932400145188363,0.01047741452656988,0.03338738576380684,1,0
4,64,1,3,3,0,-0.23749249158744853,0.23749249158744853,0.16083233691914223,0.31415264625575484,1,0
4,36,3,1,2,0,-0.1079412019210533,0.1079412019210533,0.04848942594683379,0.16739297789527283,1,0
4,60,2,1,2,0,-0.18654071654823712,0.18654071654823712,0.10836742437409001,0.2647140087223842,1,0
4,54,1,1,3,0,-0.08656390584842007,0.08656390584842007,0.057023155936040675,0.11610465576079945,1,0
3,52,3,4,3,0,-0.15046616375121422,0.15046616375121422,0.0831464921990033,0.21778583530342513,1,0
4,59,2,1,3,1,0.8305618930844503,0.16943810691554967,0.12246673294355123,0.2164094808875481,1,0
4,54,1,1,3,1,0.9134360941515799,0.08656390584842007,0.057023155936040675,0.11610465576079945,1,0
4,40,1,3,3,0,-0.0886626194439229,0.0886626194439229,0.04957632877009238,0.1277489101177534,1,0
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
5,67,3,5,3,1,0.10751520794558234,0.8924847920544177,0.8489371199128876,0.9259246294716811,1,0
4,43,1,1,3,1,0.9473322436260281,0.05266775637397194,0.03480233412235196,0.07673885805342545,1,0
5,58,4,5,3,1,0.09936258277539556,0.9006374172246044,0.866510230437708,0.9279421646894738,1,0
4,28,1,1,3,0,-0.026163057940222542,0.026163057940222542,0.014887193483417936,0.043854793323600295,1,0
5,74,1,5,3,1,0.19668218614447008,0.8033178138555299,0.6753125670917147,0.8911635949278994,1,0
4,65,1,3,3,0,-0.24638373666189764,0.24638373666189764,0.17523693840844232,0.33331170809640054,1,0
4,70,3,3,3,0,-0.5431479247562248,0.5431479247562248,0.4700078423194731,0.6147097480368733,1,0
5,42,1,3,3,0,-0.2995745189432531,0.2995745189432531,0.20060492458221607,0.42081623118527506,1,0
5,57,1,5,3,1,0.3582550717633153,0.6417449282366847,0.48419320336839716,0.7755572895833935,1,0
5,60,3,5,1,1,0.12853610998735443,0.8714638900126456,0.7112068105282827,0.9493778435223577,1,0
5,76,1,4,3,1,0.24019384149455036,0.7598061585054496,0.6356819454924827,0.853933665962314,1,0
3,42,2,1,3,1,0.9780675998548116,0.021932400145188363,0.01275538958971232,0.03625943514202956,1,0
4,64,1,3,3,0,-0.23749249158744853,0.23749249158744853,0.1688520566124277,0.32174047715559007,1,0
4,36,3,1,2,0,-0.1079412019210533,0.1079412019210533,0.06040771565074944,0.18141548515458644,1,0
4,60,2,1,2,0,-0.18654071654823712,0.18654071654823712,0.11898340402406335,0.27509284727533206,1,0
4,54,1,1,3,0,-0.08656390584842007,0.08656390584842007,0.06050888004957876,0.11981019025186523,1,0
3,52,3,4,3,0,-0.15046616375121422,0.15046616375121422,0.09376620339797588,0.2286872260493082,1,0
4,59,2,1,3,1,0.8305618930844503,0.16943810691554967,0.12646154855642922,0.2202904245930607,1,0
4,54,1,1,3,1,0.9134360941515799,0.08656390584842007,0.06050888004957876,0.11981019025186523,1,0
4,40,1,3,3,0,-0.0886626194439229,0.0886626194439229,0.05597707388914907,0.13513165333166957,1,0
";
      NlrPredict("mammography.csv",
        "Logistic(((((((1.383783475779263 * BI_RADS) + (0.04848326870704811 * Age)) + (0.5242993934294974 * Shape)) + (0.35251072256819216 * Margin)) + (-0.06848513367625006 * Density)) + -11.180915607397608))",
        "0:960", "0:19", "Severity", "Bernoulli", "TProfile", expected);
    }

    [Test]
    public void RankMammography() {
      
      var expected = @"Num param: 6 rank: 6 log10_K(J): 2.896365501198681 log10_K(J_rank): 2.896365501198681
";
      NlrRank("mammography.csv",
        "Logistic(((((((1.383783475779263 * BI_RADS) + (0.04848326870704811 * Age)) + (0.5242993934294974 * Shape)) + (0.35251072256819216 * Margin)) + (-0.06848513367625006 * Density)) + -11.180915607397608))",
        "0:960", "Severity", "Bernoulli", expected);
    }

    [Test]
    public void VariableImporanceMammography() {

      var expected = @"variable    VarExpl    
BI_RADS     9.39       %
Age         4.64       %
Shape       2.99       %
Margin      1.99       %
Density     0.01       %
";
      NlrVariableImpact("mammography.csv",
        "Logistic(((((((1.383783475779263 * BI_RADS) + (0.04848326870704811 * Age)) + (0.5242993934294974 * Shape)) + (0.35251072256819216 * Margin)) + (-0.06848513367625006 * Density)) + -11.180915607397608))",
        "0:960", "Severity", "Bernoulli", expected);
    }
    [Test]
    public void NestedMammography() {

      var expected = @"
Deviance_Factor,numPar,AICc,dAICc,BIC,dBIC,Model
1.0000e+000,6,8.0870e+002,0.0000e+000,8.6679e+002,0.0000e+000,Logistic(((((((1.3837834757792504 * BI_RADS) + (0.04848326870703262 * Age)) + (0.5242993934295344 * Shape)) + (0.35251072256817134 * Margin)) + (-0.06848513367625395 * Density)) + -11.180915607397576))
1.0001e+000,5,8.0469e+002,-4.0165e+000,8.5314e+002,-1.3655e+001,Logistic((((((Age * 0.048492802511927176) + (Shape * 0.5247998194896707)) + (Margin * 0.35034484638723523)) + (BI_RADS * 1.382285850509489)) + -11.36921169209686))
";
      NlrNested("mammography.csv",
        "Logistic(((((((1.383783475779263 * BI_RADS) + (0.04848326870704811 * Age)) + (0.5242993934294974 * Shape)) + (0.35251072256819216 * Margin)) + (-0.06848513367625006 * Density)) + -11.180915607397608))",
        "0:960", "Severity", "Bernoulli", expected);
    }
    [Test]
    public void SubtreesMammography() {

      var expected = @"SSR_factor  deltaAIC    deltaBIC    Subtree
1.6880e+000 523.4       484.7       ((((p[0] * x[0]) + (p[1] * x[1])) + (p[2] * x[2])) + (p[3] * x[3]))
1.2735e+000 202.3       173.3       (((p[0] * x[0]) + (p[1] * x[1])) + (p[2] * x[2]))
1.1984e+000 147.5       128.2       ((p[0] * x[0]) + (p[1] * x[1]))
1.1150e+000 86.1        76.5        (p[0] * x[0])
1.1150e+000 86.1        76.5        x[0]
1.0569e+000 40.5        30.9        (p[1] * x[1])
1.0569e+000 40.5        30.9        x[1]
1.0367e+000 24.7        15.0        (p[2] * x[2])
1.0367e+000 24.7        15.0        x[2]
1.0244e+000 15.0        5.4         x[3]
1.0244e+000 15.0        5.4         (p[3] * x[3])
1.0001e+000 -4.0        -13.7       (p[4] * x[4])
1.0001e+000 -4.0        -13.7       x[4]
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
        Assert.AreEqual(expected, actual);
      } finally {
        if (File.Exists(randFilename))
          File.Delete(randFilename);
      }
    }
  }
}

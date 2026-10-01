#Keerthi Kesavan - 252431 INS (4th semester)
#Exercise 03: Fitting curves


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#the function given to us to generate the data
def MakeObservations(N):       # N number of data pairs

    Nrange = 20;               # set scale

    SigmaY = 200;

    X  = np.random.uniform(0,Nrange+1,N);      # uniformly distributed between 0 and 20

    Ymax = 0.5 * Nrange**3;                    

    MuY = 0.5 * X**3 - 0.5 * Ymax             

    Y  = np.random.normal(MuY, SigmaY );       # binomially distributed around mean R1

    Noutlier = np.floor(N/5).astype(int)       #  add 20% outliers

    ix = np.random.permutation( np.arange(0,N));

    Y[ ix[0:Noutlier] ] = np.random.uniform(low=-Ymax, high=Ymax, size=Noutlier)

    return X, Y;

#step 1: generate the data
np.random.seed(0) #so results are reproducible when you re-run it
N = 500
x_i, y_i = MakeObservations(N)

#step 2: define the two model functions
def linear_model(x, a, b):
    return x * b + a

def parabolic_model(x, a, b):
    return x**2 * b + a

#step 3: fit both models
popt1, pcov1 = curve_fit(linear_model, x_i, y_i)
popt2, pcov2 = curve_fit(parabolic_model, x_i, y_i)

a1, b1 = popt1
a2, b2 = popt2

#step 4: standard deviation of the parameter estimates
#the diagonal of the covariance matrix holds the variances of a and b, so sqrt gives std
perr1 = np.sqrt(np.diag(pcov1))
perr2 = np.sqrt(np.diag(pcov2))

print("Linear model:   a = {:.3f} +/- {:.3f}, b = {:.3f} +/- {:.3f}".format(a1, perr1[0], b1, perr1[1]))
print("Parabolic model: a = {:.3f} +/- {:.3f}, b = {:.3f} +/- {:.3f}".format(a2, perr2[0], b2, perr2[1]))

#step 5: predictions from each model
y_pred_linear = linear_model(x_i, a1, b1)
y_pred_parabolic = parabolic_model(x_i, a2, b2)

#step 6: standard deviation of observations from model predictions (residual std)
resid_linear = y_i - y_pred_linear
resid_parabolic = y_i - y_pred_parabolic

std_linear = np.std(resid_linear, ddof=2)      #ddof=2 because we estimated 2 parameters (a,b)
std_parabolic = np.std(resid_parabolic, ddof=2)

print("Std of residuals, linear model:    {:.3f}".format(std_linear))
print("Std of residuals, parabolic model: {:.3f}".format(std_parabolic))

#step 7: total, explained, residual variation
y_mean = np.mean(y_i)
SS_total = np.sum((y_i - y_mean)**2)

SS_resid_linear = np.sum(resid_linear**2)
SS_explained_linear = SS_total - SS_resid_linear

SS_resid_parabolic = np.sum(resid_parabolic**2)
SS_explained_parabolic = SS_total - SS_resid_parabolic

ratio_linear = SS_explained_linear / SS_resid_linear
ratio_parabolic = SS_explained_parabolic / SS_resid_parabolic

print("\nTotal variation (SS_total): {:.1f}".format(SS_total))
print("Linear    - explained: {:.1f}, residual: {:.1f}, ratio: {:.3f}".format(SS_explained_linear, SS_resid_linear, ratio_linear))
print("Parabolic - explained: {:.1f}, residual: {:.1f}, ratio: {:.3f}".format(SS_explained_parabolic, SS_resid_parabolic, ratio_parabolic))

#step 8: plot
x_smooth = np.linspace(np.min(x_i), np.max(x_i), 300)

plt.figure(figsize=(8,6))
plt.scatter(x_i, y_i, alpha=0.5, label='observations')
plt.plot(x_smooth, linear_model(x_smooth, a1, b1), 'r-', label='linear fit')
plt.plot(x_smooth, parabolic_model(x_smooth, a2, b2), 'g-', label='parabolic fit')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve fitting: line vs parabola')
plt.legend()
plt.tight_layout()
plt.savefig('/Users/keerthikesavan/Desktop/Course_Modules/Stats/Exercises_Solved_Stats/Exercise3_CurveFittingN500_plot2.png', dpi=120)
print("\nplot saved")

"""
For N=200
Linear model:   a = -2030.398 +/- 202.189, b = 135.165 +/- 16.734
Parabolic model: a = -1643.736 +/- 144.858, b = 7.081 +/- 0.747
Std of residuals, linear model:    1410.963
Std of residuals, parabolic model: 1349.105

Total variation (SS_total): 524070460.0
Linear    - explained: 129888692.5, residual: 394181767.5, ratio: 0.330
Parabolic - explained: 163693900.8, residual: 360376559.2, ratio: 0.454
"""

""" 
For N=30
Linear model:   a = -2615.530 +/- 520.860, b = 195.481 +/- 38.446
Parabolic model: a = -1868.991 +/- 379.850, b = 8.930 +/- 1.682
Std of residuals, linear model:    1238.965
Std of residuals, parabolic model: 1212.799

Total variation (SS_total): 82665871.9
Linear    - explained: 39684917.9, residual: 42980953.9, ratio: 0.923
Parabolic - explained: 41481197.7, residual: 41184674.2, ratio: 1.007

For N=500
Linear model:   a = -2433.088 +/- 109.960, b = 165.257 +/- 9.094
Parabolic model: a = -1891.887 +/- 78.120, b = 8.085 +/- 0.394
Std of residuals, linear model:    1244.579
Std of residuals, parabolic model: 1181.009

Total variation (SS_total): 1282952556.7
Linear    - explained: 511561866.5, residual: 771390690.2, ratio: 0.663
Parabolic - explained: 588350686.6, residual: 694601870.2, ratio: 0.847
"""
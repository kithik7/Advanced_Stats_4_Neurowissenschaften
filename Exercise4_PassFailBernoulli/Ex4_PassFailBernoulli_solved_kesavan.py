#Exercise 04: Pass-Fail Bernoulli GLM (logistic regression)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#the function that was already given for this exercise, generates the class data
def PassFail( Nstudent ):

    TrueBeta = np.array([-1, 0.1, 0.4]);

    StudyHours = np.random.randint(low=0, high=41, size=Nstudent );

    Attendance = np.random.randint(low=0, high=13, size=Nstudent );

    Intercept = np.ones(Nstudent);

    X = np.concatenate((Intercept.reshape(-1,1), StudyHours.reshape(-1, 1), Attendance.reshape(-1, 1)), axis=1);

    TrueBeta[0] = -TrueBeta[1] * np.mean(StudyHours) - TrueBeta[2] * np.mean(Attendance)

    logit = X @ TrueBeta

    Pi = 1 / (1 + np.exp(-logit))

    Pass = np.random.binomial(n=1, p=Pi, size=Nstudent);

    return Pass, Pi, X, TrueBeta


#step 1: generate the class of N=100 students using the given function
np.random.seed(0) #reproducible

N = 100
R, pi_true, X, TrueBeta = PassFail(N)

#pull S and A back out of X for convenience later (column 0 is the intercept
#column of 1s, column 1 is StudyHours, column 2 is Attendance)
S = X[:, 1]
A = X[:, 2]

#step 2: negative log likelihood function
#lambda_i = X @ beta gives the log odds for every student at once
#log P(y_i | lambda_i) = y_i * lambda_i - log(1 + e^lambda_i)
#so i must sum that over all students to get the joint log likelihood, then negate it
#because scipy's minimize function by default tends to minimize, and I want to maximize the likelihood
def neg_log_likelihood(beta, X, y):
    lam = X @ beta                                   #log odds for each student
    ll = y * lam - np.log(1 + np.exp(lam))           #log P(y_i | lambda_i) per student
    return -np.sum(ll)                               #negative sum = what we minimize

#step 3: minimize the negative log likelihood to find beta
beta_guess = np.array([0.0, 0.0, 0.0])   #start with a guess

result = minimize(neg_log_likelihood, beta_guess, args=(X, R), method='BFGS')
beta_fit = result.x

mu_fit, alpha1_fit, alpha2_fit = beta_fit

print("True parameters:   mu = {:.3f}, alpha1 = {:.3f}, alpha2 = {:.3f}".format(TrueBeta[0], TrueBeta[1], TrueBeta[2]))
print("Fitted parameters: mu = {:.3f}, alpha1 = {:.3f}, alpha2 = {:.3f}".format(mu_fit, alpha1_fit, alpha2_fit))

#step 4: use the fitted model to predict pass probability for each student
lambda_fit = X @ beta_fit
pi_fit = 1 / (1 + np.exp(-lambda_fit))

#step 5: sort students by predicted pi, split into 10 groups of similar size
sort_ix = np.argsort(pi_fit)          #indices that would sort pi_fit ascending
groups = np.array_split(sort_ix, 10)  #split the sorted indices into 10 roughly-equal chunks

#step 6: per group, average pi is the predictor and R, S, A which are the observables
group_pi = []
group_R = []
group_S = []
group_A = []

for g in groups:
    group_pi.append(np.mean(pi_fit[g]))
    group_R.append(np.mean(R[g]))
    group_S.append(np.mean(S[g]))
    group_A.append(np.mean(A[g]))

group_pi = np.array(group_pi)
group_R = np.array(group_R)
group_S = np.array(group_S)
group_A = np.array(group_A)

print("\nGroup-averaged predictor (pi) vs observables:")
for i in range(10):
    print("Group {}: pi={:.3f}  R={:.3f}  S={:.2f}  A={:.2f}".format(i+1, group_pi[i], group_R[i], group_S[i], group_A[i]))

#step 7: plot the observable averages as a function of the predictor average
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

axs[0].plot(group_pi, group_R, 'o-')
axs[0].plot([0,1],[0,1],'k--', alpha=0.4, label='ideal (R = pi)')
axs[0].set_xlabel('average predicted pi (per group)')
axs[0].set_ylabel('average observed R (pass rate)')
axs[0].set_title('Pass rate vs predicted probability')
axs[0].legend()

axs[1].plot(group_pi, group_S, 'o-', color='green')
axs[1].set_xlabel('average predicted pi (per group)')
axs[1].set_ylabel('average study hours S')
axs[1].set_title('Study hours vs predicted probability')

axs[2].plot(group_pi, group_A, 'o-', color='orange')
axs[2].set_xlabel('average predicted pi (per group)')
axs[2].set_ylabel('average lectures attended A')
axs[2].set_title('Lecture attendance vs predicted probability')

plt.tight_layout()
plt.savefig('/Users/keerthikesavan/Desktop/course_modules/Stats/Exercises_Solved_Stats/Exercise4_PassFailBernoulli/Exercise4_grouped_plot.png', dpi=120)
print("\nplot saved")

"""
1) the beta parameters that minimize the likelihood are : 
Fitted parameters: mu = -3.654, alpha1 = 0.091, alpha2 = 0.354


Group-averaged predictor (pi) vs observables:
Group 1: pi=0.061  R=0.200  S=3.40  A=1.50
Group 2: pi=0.140  R=0.000  S=12.40  A=2.00
Group 3: pi=0.217  R=0.200  S=6.50  A=5.00
Group 4: pi=0.329  R=0.200  S=22.60  A=2.50
Group 5: pi=0.463  R=0.500  S=16.40  A=5.70
Group 6: pi=0.625  R=0.700  S=17.90  A=7.20
Group 7: pi=0.756  R=0.700  S=22.40  A=7.80
Group 8: pi=0.846  R=1.000  S=24.40  A=8.90
Group 9: pi=0.906  R=0.800  S=31.90  A=8.60
Group 10: pi=0.957  R=1.000  S=33.70  A=10.70
"""

"""
Interpretatio section
interpret a hypothetical result: (given alpha = [-5, 0.1, 0.4])

How does log-odds change per additional study hour?
alpha1 = 0.1, so log-odds increase by 0.1 for every additional hour of
independent study and it holds attendance fixed.

How does log-odds change per additional lecture attended?
alpha2 = 0.4, so log-odds increase by 0.4 for every additional lecture
attended, holding study hours fixed. Attendance has 4 times the effect
of a single study hour on the log-odds scale.

Starting from pi = 0.5 (lambda = 0), what is pi after two more study hours?
Two more study hours adds 2 * 0.1 = 0.2 to lambda.
From the table in the exercise , lambda = 0.2 gives pi = 0.55.

Starting from pi = 0.5 (lambda = 0), what is pi after two more attended lectures?
Two more lectures adds 2 * 0.4 = 0.8 to lambda.
From the table, lambda = 0.8 gives pi = 0.69.

In this regard, two extra lectures move the pass probability further away from 0.50 to 0.69 than 
than two extra study hours (0.50 to 0.55), this adds up with attendance having
the larger coefficient.
"""
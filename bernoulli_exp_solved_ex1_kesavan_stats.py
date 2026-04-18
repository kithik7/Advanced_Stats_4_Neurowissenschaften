#Bernoulli experiment is a random experiment with only two possible outcomes
#success which has a probbaility p and failure which has a probability 1-p. The obtained distribution is a discrete in nature 
#therefore, we use a probability mass function (PMF) to describe the such a distribution. 
#The PMF of a Bernoulli distribution is as follows: 
# f(x) = p dirac_delta(x - 1) + (1 - p) dirac_delta(x)
#where a dirac delta is just a function that is zero everywhere except at a single point, where it is infinitely high.
#the PMF for a bernoulli distribution places the probability mass at exactly two points - 1 and 0 
# f (x) is p if p is 1 and f(x) is 1-p if p is 0. 

""" 
The mean and variance for such a distribution is <x> = p and Var(X) = p(1-p) , the second moment <x^2> = p because
x is either 0 or 1 and squaring both these values does not nchange them, if the outcome is so certain, there is really 
no variability , the maximum variabiality is at 0.5 when probability is 50-50
"""

""" 
Remember that the mean decides the height or peak and the variance which is the spread and 
when we take the standard deviation it decides the width of the distribution, if std is made 
really small, at 0 there is no spread but the peak gets taller because area under curve must remain 1, so all probabilties
are concentrated at x = 0 , this limiting shape is the dirac delta function. 

the dirac_delta function(a) is lim -infinity to positive infinity, dirac(x-a) f(x)dx = f(a)
ie, it integrates any function f(x) multiplied by dirac(x-a) and picks out the value of f at a single point x = a 

it is basically a selector function because it ignores everything else and only focuses on the value at x = a. it is used
in a bernoulli experiment because probability exists at 2 points - 0 and 1 and p x dirac(x-1)is the spike of height p at
x = 1 and (1-p) x dirac(x) is the spike of height (1-p) at x = 0.

It encodes probability p and x = 1 and (1-p) and x = 0. and it stays 0 everywhere else
"""

""" 
The setup: like a russian nesting doll of functions:
Function 1: one bernoulli trial where it takes p and returns 0 or 1

Function 2: calls function 1 exactly 5 times (n=5) and we know binomial distribution is the summation of N number of 
bernoulli trials. so function 2 adds up the results and returns a number between 0 and 5 and this number is the total 
successes in 5 trials. 

Function 3 : calls function 2 exactly 1000 times and we put all the results in a list

Function 4: takes this list and computes the PMF and cumulative PMF, we plot both and it would overlay a third 
plot which is the PMF of a binomial distribution (also derived from PMF of a bernoulli distribution) to verify
"""
#note that when I deal with randomness, I generate noise in the sense that I might not always get 10 heads if I flip a coin
#ten times, i might get 8 sometimes, and the more repetitions I do, this noise is reduced because I would then approach
#the expected value of 10 heads and this is a true distribution
#the binomial PMF is mathematically given by:
#P(X=k) = (n choose k) * p^k * (1-p)^(n-k), which describes the probability of getting exactly k successes in n trials
#by running the bernoulli experiment multiple times I can obtain a distribution of outcomes that approximates this PMF
#the approximation of this truth becomes more accurate as the number of trials increases but it can never be identical


#Exercise 01
#Use python to program a bernoulli experiment with parameter 0<=p<=1
#ie. obtain a random outcome x = 1 with probability p or outcome x = 0 with probability 1-p

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def bern_trial(p): #p is the probability of success
    r = np.random.rand() #generate a random number between 0 and 1 
    if r < p:
        return 1
    else:
        return 0

#set of bernoulli experiments - function 2 
def bern_set (p, n): #takes p probability and n number of trials as input 
    total = 0 #starting counter at 0 so we can accrue successes 
    for i in range(n): #repeat n times oky
        total = total + bern_trial(p) #call fn 1 get output 0 or 1 and add it to total 
    return total 

#repetition of 5 set of bern_trials a 1000 times 
def bern_reps(p,n,reps):
    results = []
    for i in range(reps):
        results.append(bern_set(p,n)) #call fn 2 get output which is a number between 0 and 5 and add to results
    return results 

#function four is plotting 
def bern_plot(p,n,reps):
    results = bern_reps(p , n , reps)
    counts = np.bincount(results , minlength = 6) #numpy function to count how often each outcome occurs 
    proportions = counts / reps #proportion = number of counts divided by the total number of repetitions
    #the sum of all these values (the count is values between 0-5) should equal 1 because the min and max proportion 
    #is that it never occured which is 0 or it occured each time which is 1 and all values in between are fractions of 1000

#MY PROPORTIONS ARE AN APPROXIMATION OF TRUE PROBABILITY WHICH IS DESCRIBED BY THE BINOMIAL PMF

    #x axis is counts and y axis is proportions 
    outcomes = np.arange(0 , 6)

    theoretical_PMF = stats.binom.pmf(outcomes, n, p) #binomial PMF

    cumulative = np.cumsum(proportions) #cumulative PMF is the cumulative sum of the proportions

    theoretical_CDF = stats.binom.cdf(outcomes, n, p) #cumulative PMF of the binomial distribution

    # left plot — PMF
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].bar(outcomes, proportions, label='simulated', alpha=0.7)
    axs[0].plot(outcomes, theoretical_PMF, 'ro-', label='theoretical PMF')
    axs[0].set_xlabel('number of successes')
    axs[0].set_ylabel('probability')
    axs[0].set_title('PMF: n={}, p={}'.format(n, p))
    axs[0].legend()
    
    # right plot — cumulative
    axs[1].bar(outcomes, cumulative, label='simulated', alpha=0.7)
    axs[1].plot(outcomes, theoretical_CDF, 'ro-', label='theoretical CDF')
    axs[1].set_xlabel('number of successes')
    axs[1].set_ylabel('cumulative probability')
    axs[1].set_title('CDF: n={}, p={}'.format(n, p))
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

bern_plot(p = 1 , n=5 , reps=1000)

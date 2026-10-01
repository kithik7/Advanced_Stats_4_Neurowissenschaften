import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import t 
from scipy.stats import norm

#np.array takes list as its input and converts it into a numpy array so i can compute and do 
#arithmetic on the whole array unlike a plain list 
x_A = np.array([14.9 , 5.6 , 5.1 , 1.0 , 11.6 , 11.5 , 7.7, 11.6, 8.7, 9.2, 8.5, 9.8])
x_B = np.array([12.5 , 25.9, 6.2, 2.7, 6.3, 6.5, 9.8, 11.2, 10.9, 14.7, 13.9, 15.8, 20.9, 18.1])

m_A = np.mean(x_A)
m_B = np.mean(x_B)

#compute bessels corrected for variance s²A and s²b
s_A = np.var(x_A, ddof=1) #ddof is delta degrees of freedom
s_B = np.var(x_B, ddof=1)


#compute standard error of both means
se_AB = np.sqrt(s_A/len(x_A) + s_B/len(x_B))


#compute t-statistic 
t_statistic = (m_A - m_B)/se_AB

#degrees of freedom
nA = 12
nB = 14
df = nA + nB - 2

#calculate p value 
p_value = 2 * (1 - t.cdf(abs(t_statistic), df))

#print results 
print(f'Mean Group A: {m_A:.3f}, Mean Group B: {m_B:.3f}')
print(f'Standard Error: {se_AB:.3f}')
print(f'T-Statistic: {t_statistic:.3f}')
print(f'Degrees of Freedom: {df}')
print(f'P-Value: {p_value:.3f}')  #3f is to round off to 3 decimal places 

#conclusion 
if p_value < 0.05: 
    print('Reject H0 : (p<.05) , significant')
else:
    print('Fail to Reject H0 : (p>.05) , not significant')

#mean group A = 8.767 , m_B = 12.529 , SE = 1.995 , T_statistic = -1.886
#df = 24 , p_value = 0.071, failed to reject hypothesis, not significant 

#Plotting this 

""" I need to plot the t distribution with df = 24, standard normal for comparison 
the observed t_statistic marked on the x-axis, does it fall near the middle or in one of 
the tails? , along x axis add signficance thresholds with p = 0.025 and p = 0.975
I have to get the x position of the thresholds from ppf"""

#create x axis 
x = np.linspace(-5 , 5 , 1000) #1000 evenly spaced values between -5 and +5

#compute t distribution and standard normal distribution 

t_pdf = t.pdf(x, df)
normal_pdf = norm.pdf(x)

#find significance thresholds with percentile point function 
#they are x positions where tails begin at alpha 0.05

threshold_025 = t.ppf(0.025, df)
threshold_975 = t.ppf(0.975, df)

plt.plot( x, t_pdf, label = 'T-distribution')
plt.plot(x,normal_pdf, label = 'Normal distribution')
plt.axvline(x = t_statistic, color = 'red', linestyle = '--', label = 'Observed t-statistic')
plt.axvline(x = threshold_025, color = 'teal', linestyle = '--', label = 'p=0.025 threshold')
plt.axvline(x = threshold_975, color = 'cornflowerblue', linestyle = '--', label = 'p=0.975 threshold')
plt.title('T-distribution with df=24 and Standard Normal Distribution')
plt.legend()
plt.show()

#observed t statistic line falls inside the left threshold
#does not cross the boundary and is consistent with the significant value 
#if this line had fallen outside the left threshold further into the tail then 
#we could have rejected H0 

#the t distribution in blue has slightly heavier tails as it should 
#compared to the normal distribution, the difference is small because 24 as df is large, 
#the t distribution is closer to the normal distribution with this df, if it were smaller 
#like df = 3 or 5, the difference would be more dramatic 




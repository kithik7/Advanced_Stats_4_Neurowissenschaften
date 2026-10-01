#Keerthi Kesavan - 252431 INS (4th semester)
#Exercise 07: Bootstrapping and Linear Discriminant Analysis

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist

np.random.seed(0)

#step 1: the two samples given in e learning

SampleA = np.array([[-0.26436244,  2.06425101],
           [-2.37980509, -1.20815215],
           [-2.58008385, -2.0985715 ],
           [-0.99182483,  0.40532395],
           [-2.35298289, -1.19921921],
           [-0.088609  ,  1.46397554],
           [-1.36481771,  1.20517033],
           [-0.80231029,  1.39103881],
           [-0.82031769,  1.50740471],
           [-0.95578733,  0.4699077 ],
           [-0.36882667,  2.28769372],
           [ 0.30486814,  2.2283791 ],
           [-1.84375443, -0.63733283],
           [-1.61632482,  1.41378173],
           [ 0.00501658,  1.85630433],
           [-1.38378393, -0.03321356],
           [-1.05251477,  1.69128332],
           [ 0.31473932,  1.56610092],
           [-1.50749516,  0.14893287],
           [-1.46838163, -0.13495142],
           [-1.15601402,  1.45990912],
           [-1.76948011,  1.38154245],
           [-2.27032472, -0.2013027 ],
           [-0.74971675,  1.20055567],
           [-1.83270594, -1.56912642],
           [-0.83260013,  1.35034954],
           [-1.69623965,  0.23150245],
           [ 1.54081113,  2.62715136],
           [-0.03781514,  3.21980173],
           [ 0.33647524,  1.38302971]])

SampleB = np.array([[ 0.50480269, -0.8595243 ],
           [ 2.12631376, -0.40588157],
           [ 0.96102095, -0.89540001],
           [ 0.21542733, -1.48592055],
           [ 0.34640588,  0.10311699],
           [ 0.70205594,  0.18206178],
           [-0.0031605 , -2.54795728],
           [ 1.07651253, -0.31112093],
           [-0.13244489, -1.2784308 ],
           [ 0.04531333, -0.59725611],
           [ 0.25350397, -1.74880276],
           [ 0.38669165, -1.57837649],
           [ 0.44758641, -0.22856069],
           [-0.36450952, -0.77027268],
           [ 0.36668203, -0.55905819],
           [ 1.39896103, -0.79891693],
           [ 0.20277905, -0.67569133],
           [-0.65652876, -0.25235505],
           [ 1.53706646, -0.03531933],
           [ 1.1792673 , -0.5066273 ],
           [ 0.95181083,  0.15476961],
           [-0.50245373, -1.84571815],
           [ 0.34148316, -1.61394333],
           [ 1.33762833, -0.71637223],
           [-0.71742679,  0.01838429],
           [ 1.88080539, -0.3811363 ],
           [ 1.60577076,  0.68325755],
           [-0.1470361 , -0.84635457],
           [ 0.26021979, -0.35310571],
           [-0.33442215, -0.72556742]])

nA = len(SampleA)
nB = len(SampleB)

#step 2: scatter plot of the raw samples
plt.figure(figsize=(6,6))
plt.scatter(SampleA[:,0], SampleA[:,1], color='steelblue', label='Sample A')
plt.scatter(SampleB[:,0], SampleB[:,1], color='orange', label='Sample B')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Raw samples')
plt.legend()
plt.axis('equal')
#(the discriminating line w gets added to this same plot further below,
# after we compute it)


#step 3: parametric univariate test, done separately for x and y
#this is a WELCH-style unequal-variance t-test: separate variances for
#each group, no pooling. that is exactly what combining SE_A^2 + SE_B^2
#(rather than a single pooled variance) means.
def welch_ttest(sample1, sample2):
    n1 = len(sample1)
    n2 = len(sample2)

    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)

    var1 = np.var(sample1, ddof=1)     #ddof=1 applies Bessel's correction, /(n-1)
    var2 = np.var(sample2, ddof=1)

    SE1 = np.sqrt(var1 / n1)
    SE2 = np.sqrt(var2 / n2)

    t_stat = (mean1 - mean2) / np.sqrt(SE1**2 + SE2**2)

    #Welch-Satterthwaite degrees of freedom, the standard df formula that
    #matches this exact "sum of two separate SE^2" test statistic
    df = (SE1**2 + SE2**2)**2 / ( (SE1**2)**2/(n1-1) + (SE2**2)**2/(n2-1) )

    p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df))   #two-sided p-value

    return t_stat, df, p_value, mean1, mean2, SE1, SE2

tx, df_x, px, meanAx, meanBx, SEAx, SEBx = welch_ttest(SampleA[:,0], SampleB[:,0])
ty, df_y, py, meanAy, meanBy, SEAy, SEBy = welch_ttest(SampleA[:,1], SampleB[:,1])

print("--- Parametric, one dimension at a time ---")
print("x: meanA={:.3f}, meanB={:.3f}, t={:.3f}, df={:.1f}, p={:.6f}".format(meanAx, meanBx, tx, df_x, px))
print("y: meanA={:.3f}, meanB={:.3f}, t={:.3f}, df={:.1f}, p={:.6f}".format(meanAy, meanBy, ty, df_y, py))


#step 4: LDA, combining both dimensions
mA = np.mean(SampleA, axis=0)     #centroid of A, shape (2,)
mB = np.mean(SampleB, axis=0)

#covariance matrices, as seen in the exercise (dividing by n, the MLE
# not n-1). his differs from the ddof=1 convention used
# above for the univariate test 

def covariance_matrix(sample, mean):
    diffs = sample - mean            #shape (n, 2)
    n = len(sample)
    cov = (diffs.T @ diffs) / n      #shape (2,2)
    return cov

CA = covariance_matrix(SampleA, mA)
CB = covariance_matrix(SampleB, mB)

#most discriminating direction: y = (CA+CB)^-1 (mA - mB), then I normalize
y_dir = np.linalg.inv(CA + CB) @ (mA - mB)
w = y_dir / np.linalg.norm(y_dir)

#project centroids and then compute variances of each group's projected points
m1 = mA @ w
m2 = mB @ w
sigma1_sq = w @ CA @ w
sigma2_sq = w @ CB @ w

SE1 = np.sqrt(sigma1_sq / nA)
SE2 = np.sqrt(sigma2_sq / nB)

txy = (m1 - m2) / np.sqrt(SE1**2 + SE2**2)
df_xy = (SE1**2 + SE2**2)**2 / ( (SE1**2)**2/(nA-1) + (SE2**2)**2/(nB-1) )
pxy = 2 * (1 - t_dist.cdf(abs(txy), df_xy))

print("\nParametric, (LDA)")
print("w (discriminating direction) = [{:.4f}, {:.4f}]".format(w[0], w[1]))
print("projected means: m1={:.3f}, m2={:.3f}".format(m1, m2))
print("t={:.3f}, df={:.1f}, p={:.8f}".format(txy, df_xy, pxy))

#now add the discriminating line to the raw-data scatter plot
#this line passes through the midpoint of the two centroids, direction w
midpoint = (mA + mB) / 2
line_len = 4
line_pts = np.array([midpoint - line_len*w, midpoint + line_len*w])
plt.plot(line_pts[:,0], line_pts[:,1], 'k-', linewidth=2, label='discriminating direction')
plt.legend()
plt.savefig('/Users/keerthikesavan/Desktop/course_modules/Stats/Exercises_Solved_Stats/Exercise7_Bootstrap_LDA/ex07_raw_scatter.png', dpi=120)
plt.close()


#step 5: bootstrap
Nboot = 200

centroidsA_boot = np.zeros((Nboot, 2))
centroidsB_boot = np.zeros((Nboot, 2))

for i in range(Nboot):
    #resample 30 rows WITH replacement from each sample, then take the mean
    resampleA = SampleA[np.random.choice(nA, size=nA, replace=True)]
    resampleB = SampleB[np.random.choice(nB, size=nB, replace=True)]
    centroidsA_boot[i] = np.mean(resampleA, axis=0)
    centroidsB_boot[i] = np.mean(resampleB, axis=0)

#scatter plot of the bootstrapped centroids, same discriminating line overlaid
plt.figure(figsize=(6,6))
plt.scatter(centroidsA_boot[:,0], centroidsA_boot[:,1], color='steelblue', alpha=0.5, s=15, label='bootstrap centroids A')
plt.scatter(centroidsB_boot[:,0], centroidsB_boot[:,1], color='orange', alpha=0.5, s=15, label='bootstrap centroids B')
plt.plot(line_pts[:,0], line_pts[:,1], 'k-', linewidth=2, label='discriminating direction')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bootstrapped centroids (Nboot={})'.format(Nboot))
plt.legend()
plt.axis('equal')
plt.savefig('/Users/keerthikesavan/Desktop/course_modules/Stats/Exercises_Solved_Stats/Exercise7_Bootstrap_LDA/ex07_bootstrap_scatter.png', dpi=120)
plt.close()

#project every bootstrapped centroid onto w (using the SAME w computed from
#the full original data, since the discriminating dir  is a property the original data 
#it is not computed again for each bootstrap draw 

projA_boot = centroidsA_boot @ w    #shape (200,)
projB_boot = centroidsB_boot @ w    #shape (200,)

#all pairwise differences: 200 x 200 = 40000 values
pairwise_diffs = projA_boot[:, None] - projB_boot[None, :]   #shape (200,200)
pairwise_diffs = pairwise_diffs.flatten()                     #shape (40000,)

#flip sign if most differences are negative, so the convention is consistent
if np.mean(pairwise_diffs < 0) > 0.5:
    pairwise_diffs = -pairwise_diffs

p_bootstrap = np.mean(pairwise_diffs < 0)

print("\nBootstrap, non-parametric")
print("number of pairwise differences: {}".format(len(pairwise_diffs)))
print("bootstrap p-value (wrong sign): {:.6f}".format(p_bootstrap))
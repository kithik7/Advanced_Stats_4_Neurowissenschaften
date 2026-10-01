import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import skew 

np.random.seed(0) #same random output each time it runs 

#step 1 : generate WT and KO samples
# Wt is a plain stanadard normal and KO is mostly standard normaly but a fraction of its values are swapped for outcomes from a shifted high response distribution N(3,0.5)

def make_samples(N, p_high):
    WT = np.random.normal(0, 1, N)

    n_high = int(round(N * p_high))     #how many KO obv are high resp
    n_base = N - n_high                 #the rest = ordinary 

    KO_base = np.random.normal(0, 1, n_base)
    KO_high = np.random.normal(3, 0.5, n_high)
    KO = np.concatenate([KO_base, KO_high])
    np.random.shuffle(KO)               
    #so high resp are not placed at the end of this array

    return WT, KO


#step 2: the two self-made statistics
def T_skew_stat(WT, KO):
    return skew(KO) - skew(WT)

def T_tail_stat(WT, KO):
    pooled = np.concatenate([WT, KO])
    q80 = np.percentile(pooled, 80)
    frac_KO = np.mean(KO > q80)
    frac_WT = np.mean(WT > q80)
    return frac_KO - frac_WT

#step 3 : the permutation test 

"""pools everything together and would shuffle the WT/KO labels. 
It also recomputes both statistics on the relabeled groups. Repeating or iterating this a lot would build up the
null distribution which is what these statistics would look like if group identity didn't matter.

"""
def permutation_test(WT, KO, Nperm=10000):
    N = len(WT)   #WT and KO are the same size in this exercise
    pooled = np.concatenate([WT, KO])

    T_skew_perm = np.zeros(Nperm)
    T_tail_perm = np.zeros(Nperm)

    for i in range(Nperm):
        shuffled = np.random.permutation(pooled)   #shuffle  pooled values
        WT_perm = shuffled[:N]                     #first N goes to  fake WT group
        KO_perm = shuffled[N:]                      #remaining N goes to a fake KO group
        T_skew_perm[i] = T_skew_stat(WT_perm, KO_perm)
        T_tail_perm[i] = T_tail_stat(WT_perm, KO_perm)

    return T_skew_perm, T_tail_perm

"""
step 4: bootstrap resampling
this resamples both WT and KO separately, with replacement
it would keep their group ID intact and tells us how much the statistic would be affected if samples
were to draw a slightly different sample from the same population """

def bootstrap_cis(WT, KO, Nboot=2000):
    N = len(WT)

    T_skew_boot = np.zeros(Nboot)
    T_tail_boot = np.zeros(Nboot)

    for i in range(Nboot):
        WT_boot = np.random.choice(WT, size=N, replace=True)
        KO_boot = np.random.choice(KO, size=N, replace=True)
        T_skew_boot[i] = T_skew_stat(WT_boot, KO_boot)
        T_tail_boot[i] = T_tail_stat(WT_boot, KO_boot)

    return T_skew_boot, T_tail_boot


#run the whole analysis (visualization and both stats) for one specific p_high this is the main run
def analysis_pipeline(p_high, save_prefix, Nperm=10000, Nboot=2000):

    N = 1000
    WT, KO = make_samples(N, p_high)

    
    #visualization with a scatter and box plot (2 out of all the other possible ways to visualize data)
    fig, axs = plt.subplots(1, 2, figsize=(12,4))

    #scatter plot one column of x-pos/per group, jitter to avoid vertical stacking 
    x_WT = np.random.normal(0, 0.04, N)      #tiny random jitter around x=0
    x_KO = np.random.normal(1, 0.04, N)      #tiny random jitter around x=1

    axs[0].scatter(x_WT, WT, alpha=0.3, s=10, label='WT')
    axs[0].scatter(x_KO, KO, alpha=0.3, s=10, label='KO')
    axs[0].set_xticks([0, 1])
    axs[0].set_xticklabels(['WT', 'KO'])
    axs[0].set_ylabel('value')
    axs[0].set_title('Strip plot, p_high={}'.format(p_high))
    axs[0].legend()

    #boxplot
    axs[1].boxplot([WT, KO], tick_labels=['WT', 'KO'])
    axs[1].set_ylabel('value')
    axs[1].set_title('Boxplot, p_high={}'.format(p_high))

    plt.tight_layout()
    plt.savefig(save_prefix + '_visualize.png', dpi=120)
    plt.close()

    #observed statistics
    T_skew_obs = T_skew_stat(WT, KO)
    T_tail_obs = T_tail_stat(WT, KO)

    #permutation test
    T_skew_perm, T_tail_perm = permutation_test(WT, KO, Nperm=Nperm)

    p_skew_one_sided = np.mean(T_skew_perm >= T_skew_obs)
    p_tail_one_sided = np.mean(T_tail_perm >= T_tail_obs)

    fig, axs = plt.subplots(1, 2, figsize=(12,4))
    axs[0].hist(T_skew_perm, bins=50, color='gray', alpha=0.7)
    axs[0].axvline(T_skew_obs, color='red', linewidth=2, label='observed T_skew')
    axs[0].set_title('Null distribution: T_skew')
    axs[0].set_xlabel('T_skew (permuted)')
    axs[0].legend()

    axs[1].hist(T_tail_perm, bins=50, color='gray', alpha=0.7)
    axs[1].axvline(T_tail_obs, color='red', linewidth=2, label='observed T_tail')
    axs[1].set_title('Null distribution: T_tail')
    axs[1].set_xlabel('T_tail (permuted)')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(save_prefix + '_permutation.png', dpi=120)
    plt.close()

    #bootstrap CIs
    T_skew_boot, T_tail_boot = bootstrap_cis(WT, KO, Nboot=Nboot)

    CI_skew = np.percentile(T_skew_boot, [2.5, 97.5])
    CI_tail = np.percentile(T_tail_boot, [2.5, 97.5])

    fig, axs = plt.subplots(1, 2, figsize=(12,4))
    axs[0].hist(T_skew_boot, bins=50, color='steelblue', alpha=0.7)
    axs[0].axvline(T_skew_obs, color='red', linewidth=2, label='observed')
    axs[0].axvline(CI_skew[0], color='black', linestyle='--', label='95% CI')
    axs[0].axvline(CI_skew[1], color='black', linestyle='--')
    axs[0].set_title('Bootstrap distribution: T_skew')
    axs[0].legend()

    axs[1].hist(T_tail_boot, bins=50, color='steelblue', alpha=0.7)
    axs[1].axvline(T_tail_obs, color='red', linewidth=2, label='observed')
    axs[1].axvline(CI_tail[0], color='black', linestyle='--', label='95% CI')
    axs[1].axvline(CI_tail[1], color='black', linestyle='--')
    axs[1].set_title('Bootstrap distribution: T_tail')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(save_prefix + '_bootstrap.png', dpi=120)
    plt.close()

    print("p_high = {}".format(p_high))
    print("T_skew observed = {:.4f}, one-sided p = {:.4f}, 95% CI = [{:.4f}, {:.4f}]".format(
        T_skew_obs, p_skew_one_sided, CI_skew[0], CI_skew[1]))
    print("T_tail  observed = {:.4f}, one-sided p = {:.4f}, 95% CI = [{:.4f}, {:.4f}]".format(
        T_tail_obs, p_tail_one_sided, CI_tail[0], CI_tail[1]))
    print()

    return {
        'p_high': p_high,
        'T_skew': T_skew_obs, 'p_skew': p_skew_one_sided, 'CI_skew': CI_skew,
        'T_tail': T_tail_obs, 'p_tail': p_tail_one_sided, 'CI_tail': CI_tail
    }


#run the full analysis for p_high val = 0.10 
result_main = analysis_pipeline(p_high=0.10, save_prefix='/Users/keerthikesavan/Desktop/course_modules/Stats/Exercises_Solved_Stats/Exercise6_Bootstrap_Aleshin/ex06_main', Nperm=10000, Nboot=2000)

# repeat w different effect sizes
p_high_values = [0.02, 0.05, 0.10, 0.20]
results_table = []

for p in p_high_values:
    res = analysis_pipeline(p_high=p, save_prefix='/Users/keerthikesavan/Desktop/course_modules/Stats/Exercises_Solved_Stats/Exercise6_Bootstrap_Aleshin/ex06_phigh_{}'.format(str(p).replace('.', '')),
                             Nperm=3000, Nboot=1000)
    results_table.append(res)
print("\n results table for different high response KO observations:")
print("{:<8} {:>10} {:>10} {:>20} {:>10} {:>10} {:>20}".format(
    'p_high', 'T_skew', 'p_skew', '95% CI T_skew', 'T_tail', 'p_tail', '95% CI T_tail'))
for res in results_table:
    print("{:<8} {:>10.4f} {:>10.4f} [{:>7.4f},{:>7.4f}] {:>10.4f} {:>10.4f} [{:>7.4f},{:>7.4f}]".format(
        res['p_high'], res['T_skew'], res['p_skew'], res['CI_skew'][0], res['CI_skew'][1],
        res['T_tail'], res['p_tail'], res['CI_tail'][0], res['CI_tail'][1]))
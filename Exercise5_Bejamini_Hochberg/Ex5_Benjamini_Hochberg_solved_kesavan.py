#Keerthi Kesavan - 252431 INS (4th semester)
#Exercise 05: Benjamini-Hochberg FDR correction

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#MixedSamples is the pre defined functionfor this exercise, it basically generates the mixed samples
def MixedSamples( Nsample ):

    # Define distributions
    mu0 = 0
    mu1 = 3
    xi = np.linspace(-5, 5, 101)
    f0i = 1/np.sqrt(2*np.pi) * np.exp(-((xi - mu0)**2)/2)
    f1i = 1/np.sqrt(2*np.pi) * np.exp(-((xi - mu1)**2)/2)

    # Define mixture
    pi0 = 0.98
    pi1 = 1 - pi0

    # Intersection
    sthresh = np.log(pi0/pi1) / mu1 + mu1/2

    # Sample distributions
    N = Nsample
    N0 = round(N * pi0)
    N1 = round(N * pi1)

    s0 = np.random.normal(mu0, 1, N0)
    s1 = np.random.normal(mu1, 1, N1)

    # Combine samples and source labels
    samples = np.concatenate([s0, s1])
    sources = np.concatenate([np.zeros(N0), np.ones(N1)])

    # Generate shuffling indices
    indices = np.random.permutation(len(samples))

    # Apply to both arrays
    samples = samples[indices]
    sources = sources[indices]

    # Return results
    return samples, sources, sthresh


def run_bh_analysis(Nsample, save_prefix):

    #step 1: get samples, their true source (0=null, 1=non-null), and the true threshold
    samples, sources, sthresh = MixedSamples(Nsample)

    #step 2: sort samples from largest to smallest, and then carry the source labels along
    #in the same order so each source label still matches its sample
    sort_ix = np.argsort(samples)[::-1]     #argsort gives ascending order, [::-1] reverses to descending
    samples_sorted = samples[sort_ix]
    sources_sorted = sources[sort_ix]

    #cumulative probability vector: rank/N. rank 1 (the largest sample) gets 1/N,
    #which is the "smallest" cumulative probability, matching the exercise's instruction
    cum_prob = np.arange(1, Nsample + 1) / Nsample

    red_mask = sources_sorted == 1   #True where a sample is actually non-null

    #step 3: plot cumulative distribution, sample value (x) vs cumulative probability (y)
    plt.figure(figsize=(6,5))
    plt.plot(samples_sorted, cum_prob, 'o', markersize=3, color='steelblue', label='all samples')
    plt.plot(samples_sorted[red_mask], cum_prob[red_mask], 'o', markersize=4, color='red', label='non-null (true)')
    plt.xlabel('sample value')
    plt.ylabel('cumulative probability')
    plt.title('Cumulative distribution, N={}'.format(Nsample))
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_prefix + '_cumulative.png', dpi=120)

    #step 4: convert every sample value to a p-value
    #p-value = tail area of a STANDARD normal beyond that sample value
    #norm.sf(x) = 1 - CDF(x) = exactly this tail area
    p_values = norm.sf(samples_sorted)

    #true threshold converted to a p-value the same way, for the dashed line
    p_thresh = norm.sf(sthresh)

    #BH criterion line: p = q * (rank/N), a straight line through the origin with slope q
    q = 0.2
    bh_line = q * cum_prob

    #step 5: BH plot, cumulative probability (x) vs p-value (y), zoomed to [0, 0.05]
    plt.figure(figsize=(6,5))
    plt.plot(cum_prob, p_values, 'o', markersize=3, color='steelblue', label='all samples')
    plt.plot(cum_prob[red_mask], p_values[red_mask], 'o', markersize=4, color='red', label='non-null (true)')
    plt.axhline(p_thresh, color='black', linestyle='--', label='true threshold')
    plt.plot(cum_prob, bh_line, color='magenta', label='B-H criterion (q=0.2)')
    plt.xlim(0, 0.05)
    plt.ylim(0, 0.05)
    plt.xlabel('cumulative probability (rank / N)')
    plt.ylabel('p-value')
    plt.title('Benjamini-Hochberg plot, N={}'.format(Nsample))
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_prefix + '_BH.png', dpi=120)

    #step 6: find the B-H cutoff and compute the false discovery fraction
    #condition: p_(i) <= q * (i/N), samples are already in ascending-p / ascending-rank order
    below_bh = p_values <= bh_line

    if np.any(below_bh):
        k_max = np.max(np.where(below_bh)[0])   #largest index satisfying the condition
        rejected = np.zeros(Nsample, dtype=bool)
        rejected[:k_max + 1] = True             #reject everything from rank 1 up to k_max
    else:
        rejected = np.zeros(Nsample, dtype=bool)

    n_rejected = np.sum(rejected)
    n_false_discoveries = np.sum(rejected & (sources_sorted == 0))   #rejected but actually null

    if n_rejected > 0:
        fdr_observed = n_false_discoveries / n_rejected
    else:
        fdr_observed = float('nan')

    print("N = {} ".format(Nsample))
    print("true threshold sthresh = {:.3f}, p_thresh = {:.5f}".format(sthresh, p_thresh))
    print("number of samples rejected by B-H: {}".format(n_rejected))
    print("of those, actually null (false discoveries): {}".format(n_false_discoveries))
    print("observed false discovery fraction: {:.3f}  (target q = {})".format(fdr_observed, q))
    print()

    return fdr_observed


#run once at a moderate N
np.random.seed(0)
run_bh_analysis(Nsample=1000, save_prefix='/Users/keerthikesavan/Desktop/course_modules/Stats/Exercises_Solved_Stats/Exercise5_Bejamini_Hochberg/ex05_N1000')

#run again at a much larger N to check the false discovery fraction more reliably
np.random.seed(0)
run_bh_analysis(Nsample=20000, save_prefix='/Users/keerthikesavan/Desktop/course_modules/Stats/Exercises_Solved_Stats/Exercise5_Bejamini_Hochberg/ex05_N20000')
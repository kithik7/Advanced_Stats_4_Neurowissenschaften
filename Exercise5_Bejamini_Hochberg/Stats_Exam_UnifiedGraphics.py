import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

from matplotlib.patches import FancyArrowPatch, Ellipse

from scipy import stats

from scipy.stats import norm, expon, gamma, lognorm, poisson, binom, chi2

*# ─── PALETTE ────────────────────────────────────────────────────────────────*

BLUE   = '#3D7AB5'

GREEN  = '#4E9A6D'

RED    = '#B5413D'

ORANGE = '#C07A2E'

PURPLE = '#7B5EA7'

TEAL   = '#2E8B8B'

GRAY   = '#6B7280'

LGRAY  = '#D1D5DB'

BG     = 'white'

PANEL  = '#F7F8FC'

**def** style(ax, title='', xlabel='', ylabel=''):

    ax.set_facecolor(PANEL)

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_color(LGRAY)

    ax.spines['bottom'].set_color(LGRAY)

    ax.tick_params(colors=GRAY, labelsize=8)

    if title:  ax.set_title(title, fontsize=10, fontweight='bold', color='#1a1a2e', pad=8)

    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color=GRAY)

    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=GRAY)

    ax.grid(True, color=LGRAY, linewidth=0.5, alpha=0.7)

*# ════════════════════════════════════════════════════════════════════════════*

*# DIAGRAM 0: Normal distribution   mean, variance, SE, z-score*

*# ════════════════════════════════════════════════════════════════════════════*

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

fig.patch.set_facecolor(BG)

fig.suptitle('The Normal Distribution   Mean, Variance, Standard Error, Z-score',

             fontsize=12, fontweight='bold', color='#1a1a2e', y=1.01)

*# Panel 1: Bell curve with mean, std, variance*

ax = axes[0]

x = np.linspace(-4, 4, 300)

y = norm.pdf(x)

ax.plot(x, y, color=BLUE, linewidth=2.5)

ax.fill_between(x, y, where=(np.abs(x) <= 1), color=BLUE, alpha=0.15,

                label=**r**'68%: $\pm 1\sigma$')

ax.fill_between(x, y, where=(np.abs(x) <= 2), color=TEAL, alpha=0.08,

                label=**r**'95%: $\pm 2\sigma$')

ax.axvline(0, color=RED, linewidth=1.8, linestyle='--', label=**r**'mean $\mu=0$')

ax.annotate('', xy=(1, 0.22), xytext=(0, 0.22),

            arrowprops=dict(arrowstyle='<->', color=GREEN, lw=1.8))

ax.text(0.5, 0.24, **r**'$\sigma$', color=GREEN, fontsize=11, ha='center')

ax.annotate('', xy=(-1, 0.15), xytext=(1, 0.15),

            arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=1.8))

ax.text(0, 0.17, **r**'$2\sigma$ (spread)', color=ORANGE, fontsize=8.5, ha='center')

ax.text(0, -0.055,

**r**'$\text{Var}(x) = \langle x^2 \rangle - \langle x \rangle^2$'

        '\n' **r**'$\text{Std}(x) = \sqrt{\text{Var}(x)}$',

        ha='center', fontsize=8, color='#1a1a2e',

        bbox=dict(boxstyle='round', facecolor='#EEF2FF', edgecolor=BLUE, alpha=0.9))

ax.set_ylim(-0.1, 0.48)

ax.legend(fontsize=7.5, framealpha=0.8)

style(ax, title=**r**'$\mathcal{N}(0,1)$: Shape, Mean, Variance',

      xlabel='x', ylabel='density p(x)')

*# Panel 2: Standard error   sampling distribution of the mean*

ax = axes[1]

x = np.linspace(-4, 4, 300)

for n, col, lw, lab in [(1, BLUE, 2.5, 'n=1 (individual)'),

                         (5, GREEN, 2.0, 'n=5'),

                         (25, RED, 2.0, 'n=25'),

                         (100, PURPLE, 1.8, 'n=100')]:

    se = 1/np.sqrt(n)

    y = norm.pdf(x, 0, se)

    ax.plot(x, y, color=col, linewidth=lw, label=**f**'n={n}, SE={se**:.2f**}')

ax.text(0, -0.25,

**r**'$SE = \frac{\sigma}{\sqrt{n}}$' + '     ' +

**r**'$s^2 = \frac{1}{n-1}\sum(x_i - m)^2$',

        ha='center', fontsize=8.5, color='#1a1a2e',

        bbox=dict(boxstyle='round', facecolor='#EDFBF4', edgecolor=GREEN, alpha=0.9))

ax.set_ylim(-0.4, 4.5)

ax.legend(fontsize=7.5, framealpha=0.8)

style(ax, title='Standard Error: sampling distribution of mean',

      xlabel='sample mean m', ylabel='density')

*# Panel 3: Z-score transformation*

ax = axes[2]

x_raw = np.linspace(60, 140, 300)

mu, sigma = 100, 15

y_raw = norm.pdf(x_raw, mu, sigma)

ax2 = ax.twiny()

x_z = (x_raw - mu) / sigma

ax.plot(x_raw, y_raw, color=BLUE, linewidth=2.5, label='raw: N(100,15)')

ax.axvline(mu, color=RED, linestyle='--', linewidth=1.5, label=**f**'mean={mu}')

ax.fill_between(x_raw, y_raw,

                where=(x_raw >= mu-sigma) & (x_raw <= mu+sigma),

                color=BLUE, alpha=0.15)

ax2.set_xlim(ax.get_xlim())

ax2.set_xticks([mu-2\*sigma, mu-sigma, mu, mu+sigma, mu+2\*sigma])

ax2.set_xticklabels(['-2', '-1', '0', '+1', '+2'], fontsize=8, color=GREEN)

ax2.set_xlabel('z-score', fontsize=8.5, color=GREEN)

ax.text(100, -0.004,

**r**'$z_i = \frac{x_i - \mu}{\sigma}$' + '\nmaps any N(μ,σ) to N(0,1)',

        ha='center', fontsize=8.5, color='#1a1a2e',

        bbox=dict(boxstyle='round', facecolor='#FFF8EE', edgecolor=ORANGE, alpha=0.9))

ax.set_ylim(-0.008, 0.032)

ax.legend(fontsize=7.5, framealpha=0.8)

style(ax, title='Z-score: morphism N(μ,σ) → N(0,1)',

      xlabel='raw x (e.g. IQ)', ylabel='density')

plt.tight_layout()

plt.savefig('/Users/keerthikesavan/Desktop/Course_Modules/Stats/Exercises_Solved_Stats_2026/diagram_normal.png', dpi=140, bbox_inches='tight', facecolor=BG)

plt.close()

print("diagram_normal done")

*# ════════════════════════════════════════════════════════════════════════════*

*# DIAGRAM 1: Q1   Mean, Median, Mode, Variance*

*# ════════════════════════════════════════════════════════════════════════════*

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

fig.patch.set_facecolor(BG)

fig.suptitle('Q1: What does my data look like?   Central tendency and spread',

             fontsize=12, fontweight='bold', color='#1a1a2e')

*# Panel 1: symmetric vs skewed*

ax = axes[0]

x = np.linspace(-1, 8, 400)

*# symmetric*

y_sym = norm.pdf(x, 3, 0.8)

*# skewed (lognormal)*

y_skew = lognorm.pdf(x, 0.7, scale=np.exp(0.8))

y_skew = y_skew / y_skew\.max() \* y_sym.max() \* 0.85

ax.plot(x, y_sym, color=BLUE, linewidth=2.5, label='Symmetric: mode=median=mean')

ax.plot(x, y_skew, color=RED, linewidth=2.5, linestyle='--', label='Skewed: mode < median < mean')

*# symmetric markers*

ax.axvline(3, color=BLUE, linewidth=1.2, alpha=0.6)

ax.text(3, 0.52, 'all three\ncoincide', color=BLUE, fontsize=7.5, ha='center')

*# skewed markers*

mode_sk = x[np.argmax(y_skew)]

med_sk = 1.8

mean_sk = 2.5

for val, lbl, col in [(mode_sk, 'mode', RED),

                       (med_sk, 'median', ORANGE),

                       (mean_sk, 'mean\n(pulled by tail)', PURPLE)]:

    ax.axvline(val, color=col, linewidth=1.3, alpha=0.7, linestyle=':')

    ax.text(val, -0.04, lbl, color=col, fontsize=7.5, ha='center')

ax.set_ylim(-0.08, 0.58)

ax.legend(fontsize=8, framealpha=0.9, loc='upper right')

style(ax, title='Mode / Median / Mean', xlabel='x', ylabel='density')

*# Panel 2: Variance = average squared distance*

ax = axes[1]

np.random.seed(7)

data = np.array([2, 3, 3, 4, 4, 4, 5, 5, 6, 9])

mean_d = np.mean(data)

ax.scatter(data, np.zeros_like(data), color=BLUE, s=80, zorder=5, label='observations')

ax.axvline(mean_d, color=RED, linewidth=2, label=**f**'mean = {mean_d**:.1f**}')

for xi in data:

    ax.annotate('', xy=(xi, 0.3), xytext=(mean_d, 0.3),

                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2, alpha=0.7))

ax.text(mean_d, 0.55,

**r**'$\text{Var}(x) = \frac{1}{N}\sum(x_i - \bar{x})^2$' + '\n' +

**r**'$= \langle x^2 \rangle - \langle x \rangle^2$',

        ha='center', fontsize=9, color='#1a1a2e',

        bbox=dict(boxstyle='round', facecolor='#EDFBF4', edgecolor=GREEN))

ax.text(mean_d, -0.35,

**r**'Bessel: $s^2 = \frac{1}{n-1}\sum(x_i - m)^2$' + '\n' +

**r**'$SE = s/\sqrt{n}$',

        ha='center', fontsize=9, color='#1a1a2e',

        bbox=dict(boxstyle='round', facecolor='#FFF8EE', edgecolor=ORANGE))

ax.set_xlim(0, 11)

ax.set_ylim(-0.55, 0.75)

ax.set_yticks([])

ax.legend(fontsize=8, framealpha=0.9)

style(ax, title='Variance = average squared deviation from mean', xlabel='x')

plt.tight_layout()

plt.savefig('/Users/keerthikesavan/Desktop/Course_Modules/Stats/Exercises_Solved_Stats_2026/diagram_q1.png', dpi=140, bbox_inches='tight', facecolor=BG)

plt.close()

print("diagram_q1 done")

*# ════════════════════════════════════════════════════════════════════════════*

*# DIAGRAM 2: Q2   All 8 distributions*

*# ════════════════════════════════════════════════════════════════════════════*

fig, axes = plt.subplots(2, 4, figsize=(16, 7))

fig.patch.set_facecolor(BG)

fig.suptitle('Q2: The Eight Distributions   Each lives in a different space',

             fontsize=12, fontweight='bold', color='#1a1a2e')

cols = [BLUE, TEAL, GREEN, ORANGE, RED, PURPLE, '#2E8B57', '#B8860B']

dists = [

    ('Normal\nN(0,1)', 'ℝ', 'μ, σ'),

    ('Exponential\nθ=2', 'ℝ⁺', 'θ (scale)'),

    ('Gamma\nα=3, θ=2', 'ℝ⁺', 'α (shape), θ'),

    ('Lognormal\nμ=0, σ=0.5', 'ℝ⁺', 'μ, σ (log scale)'),

    ('Bernoulli\np=0.7', '{0,1}', 'p'),

    ('Binomial\nN=10, p=0.4', '{0..N}', 'N, p'),

    ('Poisson\nλ=4', 'ℤ⁺', 'λ'),

    ('Chi-square\nk=5', 'ℝ⁺', 'k (dof)'),

]

for idx, (ax, (title, space, params), col) in enumerate(zip(axes.flat, dists, cols)):

    ax.set_facecolor(PANEL)

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_color(LGRAY)

    ax.spines['bottom'].set_color(LGRAY)

    ax.tick_params(colors=GRAY, labelsize=7)

    if idx == 0:

        x = np.linspace(-4, 4, 200)

        ax.plot(x, norm.pdf(x), color=col, lw=2)

        ax.fill_between(x, norm.pdf(x), alpha=0.15, color=col)

    elif idx == 1:

        x = np.linspace(0, 12, 200)

        ax.plot(x, expon.pdf(x, scale=2), color=col, lw=2)

        ax.fill_between(x, expon.pdf(x, scale=2), alpha=0.15, color=col)

    elif idx == 2:

        x = np.linspace(0, 20, 200)

        ax.plot(x, gamma.pdf(x, 3, scale=2), color=col, lw=2)

        ax.fill_between(x, gamma.pdf(x, 3, scale=2), alpha=0.15, color=col)

    elif idx == 3:

        x = np.linspace(0.01, 6, 200)

        ax.plot(x, lognorm.pdf(x, 0.5), color=col, lw=2)

        ax.fill_between(x, lognorm.pdf(x, 0.5), alpha=0.15, color=col)

    elif idx == 4:

        ax.bar([0, 1], [0.3, 0.7], color=[LGRAY, col], edgecolor='white', width=0.4)

        ax.set_xticks([0, 1])

        ax.set_xticklabels(['0\n(fail)', '1\n(success)'], fontsize=7)

    elif idx == 5:

        k = np.arange(0, 11)

        ax.bar(k, binom.pmf(k, 10, 0.4), color=col, alpha=0.8, edgecolor='white')

    elif idx == 6:

        k = np.arange(0, 12)

        ax.bar(k, poisson.pmf(k, 4), color=col, alpha=0.8, edgecolor='white')

    elif idx == 7:

        x = np.linspace(0, 20, 200)

        ax.plot(x, chi2.pdf(x, 5), color=col, lw=2)

        ax.fill_between(x, chi2.pdf(x, 5), alpha=0.15, color=col)

    ax.set_title(title, fontsize=8.5, fontweight='bold', color='#1a1a2e', pad=4)

    ax.text(0.97, 0.92, **f**'Space: {space}', transform=ax.transAxes,

            fontsize=6.5, color=col, ha='right', fontweight='bold',

            bbox=dict(boxstyle='round', facecolor='white', edgecolor=col, alpha=0.8, pad=0.2))

    ax.text(0.97, 0.78, **f**'Params: {params}', transform=ax.transAxes,

            fontsize=6, color=GRAY, ha='right')

    ax.grid(True, color=LGRAY, linewidth=0.4, alpha=0.6)

plt.tight_layout()

plt.savefig('/Users/keerthikesavan/Desktop/Course_Modules/Stats/Exercises_Solved_Stats_2026/diagram_q2.png', dpi=140, bbox_inches='tight', facecolor=BG)

plt.close()

print("diagram_q2 done")

*# ════════════════════════════════════════════════════════════════════════════*

*# DIAGRAM 3: Q3   Hypothesis testing, test statistic, p-value*

*# ════════════════════════════════════════════════════════════════════════════*

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

fig.patch.set_facecolor(BG)

fig.suptitle('Q3: Is what I see surprising?   Hypothesis testing',

             fontsize=12, fontweight='bold', color='#1a1a2e')

*# Panel 1: null distribution and test statistic*

ax = axes[0]

x = np.linspace(-5, 5, 300)

y = norm.pdf(x)

ax.plot(x, y, color=BLUE, linewidth=2.5, label='null distribution')

t_obs = 2.3

ax.axvline(t_obs, color=RED, linewidth=2.5, label=**f**'t observed = {t_obs}')

ax.fill_between(x, y, where=(x >= t_obs), color=RED, alpha=0.25, label='p-value (one-sided)')

ax.fill_between(x, y, where=(x <= -t_obs), color=RED, alpha=0.25)

ax.text(t_obs + 0.2, 0.25, **f**'p = {2\*(1-norm.cdf(t_obs))**:.3f**}', color=RED,

        fontsize=9, fontweight='bold')

ax.text(0, -0.07,

**r**'$t = \frac{m_1 - m_2}{SE}$' + '     signal / noise',

        ha='center', fontsize=9,

        bbox=dict(boxstyle='round', facecolor='#EEF2FF', edgecolor=BLUE))

ax.legend(fontsize=7.5)

style(ax, title='Null distribution and p-value',

      xlabel='test statistic t', ylabel='density')

*# Panel 2: Type I and Type II errors*

ax = axes[1]

x = np.linspace(-5, 8, 400)

y_null = norm.pdf(x, 0, 1)

y_alt  = norm.pdf(x, 3, 1)

ax.plot(x, y_null, color=BLUE, lw=2, label=**r**'$H_0$: null (no effect)')

ax.plot(x, y_alt,  color=GREEN, lw=2, label=**r**'$H_1$: alternative (real effect)')

alpha_line = norm.ppf(0.95)

ax.axvline(alpha_line, color=GRAY, linewidth=1.5, linestyle='--',

           label=**f**'threshold α=0.05 → {alpha_line**:.2f**}')

ax.fill_between(x, y_null, where=(x >= alpha_line), color=RED,

                alpha=0.3, label='Type I: false positive (α)')

ax.fill_between(x, y_alt, where=(x <= alpha_line), color=ORANGE,

                alpha=0.3, label='Type II: false negative (β)')

ax.text(2.3, 0.12, 'POWER\n1-β', color=GREEN, fontsize=8, ha='center', fontweight='bold')

ax.legend(fontsize=7, framealpha=0.9)

style(ax, title='Type I (α) and Type II (β) errors',

      xlabel='test statistic', ylabel='density')

*# Panel 3: The four tests*

ax = axes[2]

ax.set_facecolor(PANEL)

ax.axis('off')

tests = [

    ('t-test', 'tests MEANS', **r**'$t = \frac{m_1-m_2}{\sqrt{s_1^2/n_1+s_2^2/n_2}}$',

**r**'$\nu = n_1+n_2-2$', BLUE),

    ('F-test', 'tests VARIANCES', **r**'$f = s_1^2 / s_2^2$',

**r**'$F(\nu_1, \nu_2)$ distribution', GREEN),

    ('χ² test', 'tests DISCRETE counts', **r**'$\chi^2 = \sum \frac{(n_i-e_i)^2}{e_i}$',

**r**'discrete, binned data', RED),

    ('KS test', 'tests CONTINUOUS distributions',

**r**'$D = \max_x |S\_{N_1}(x) - S\_{N_2}(x)|$',

     'no distribution assumed', ORANGE),

]

y_pos = 0.92

for name, what, formula, note, col in tests:

    ax.text(0.02, y_pos, name, fontsize=10, fontweight='bold', color=col,

            transform=ax.transAxes)

    ax.text(0.28, y_pos, what, fontsize=8.5, color='#1a1a2e',

            transform=ax.transAxes)

    ax.text(0.02, y_pos-0.07, formula, fontsize=8.5, color=col,

            transform=ax.transAxes, family='monospace')

    ax.text(0.02, y_pos-0.13, note, fontsize=7.5, color=GRAY,

            transform=ax.transAxes, style='italic')

*#     ax.axhline(y=(y_pos-0.16), xmin=0.01, xmax=0.99,*

    y_pos -= 0.23

ax.set_title('The four tests   same structure: signal/noise',

             fontsize=10, fontweight='bold', color='#1a1a2e', pad=8)

plt.tight_layout()

plt.savefig('/Users/keerthikesavan/Desktop/Course_Modules/Stats/Exercises_Solved_Stats_2026/diagram_q3.png', dpi=140, bbox_inches='tight', facecolor=BG)

plt.close()

print("diagram_q3 done")

*# ════════════════════════════════════════════════════════════════════════════*

*# DIAGRAM 4: Q4   Pairwise association*

*# ════════════════════════════════════════════════════════════════════════════*

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

fig.patch.set_facecolor(BG)

fig.suptitle('Q4: Do two variables move together?   Pairwise association',

             fontsize=12, fontweight='bold', color='#1a1a2e')

*# Panel 1: Pearson r as cosine angle*

ax = axes[0]

np.random.seed(42)

for r_val, col, lbl in [(0.9, BLUE, 'r=0.9 (strong)'),

                          (0.4, GREEN, 'r=0.4 (weak)'),

                          (0.0, RED, 'r=0 (none)')]:

    cov = np.array([[1, r_val],[r_val, 1]])

    data = np.random.multivariate_normal([0,0], cov, 80)

    ax.scatter(data[:,0], data[:,1], color=col, alpha=0.4, s=15, label=lbl)

ax.set_aspect('equal')

ax.text(0, -3.2,

**r**'$r = \frac{\text{Cov}(x,y)}{\sigma_x \sigma_y} = \langle z(x) \cdot z(y) \rangle$'

        '\n= cosine of angle between data vectors',

        ha='center', fontsize=8.5,

        bbox=dict(boxstyle='round', facecolor='#EEF2FF', edgecolor=BLUE))

ax.legend(fontsize=7.5)

style(ax, title='Pearson r   linear association', xlabel='x', ylabel='y')

*# Panel 2: Spearman (ranks)*

ax = axes[1]

np.random.seed(7)

x_d = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)

y_d = x_d\*\*2 + np.random.normal(0, 3, 10)

ranks_x = stats.rankdata(x_d)

ranks_y = stats.rankdata(y_d)

ax.scatter(x_d, y_d, color=TEAL, s=60, zorder=5, label='raw values')

ax2_twin = ax.twinx().twiny()

ax2_twin.scatter(ranks_x, ranks_y, color=ORANGE, s=60, marker='s',

                 zorder=6, label='ranks', alpha=0.7)

ax2_twin.set_xlabel('rank of x', fontsize=7.5, color=ORANGE)

ax2_twin.set_ylabel('rank of y', fontsize=7.5, color=ORANGE)

ax2_twin.tick_params(colors=ORANGE, labelsize=7)

r_s, \_ = stats.spearmanr(x_d, y_d)

ax.text(5.5, 5,

**r**'$r_s = 1 - \frac{6\sum D_i^2}{N(N^2-1)}$' + **f**'\n$D_i = R_i - S_i$\n'

**f**'Spearman r = {r_s**:.3f**}',

        ha='center', fontsize=8.5, color='#1a1a2e',

        bbox=dict(boxstyle='round', facecolor='#FFF8EE', edgecolor=ORANGE))

ax.legend(fontsize=7.5, loc='upper left')

style(ax, title='Spearman   rank-order (monotonic, robust)', xlabel='x', ylabel='y')

*# Panel 3: Mutual information (entropy)*

ax = axes[2]

ax.set_facecolor(PANEL)

ax.axis('off')

ax.set_title('Mutual Information   most general', fontsize=10,

             fontweight='bold', color='#1a1a2e', pad=8)

lines = [

    (**r**'$H(x) = -\sum_i p_i \ln p_i$', 'Shannon entropy of x', PURPLE),

    (**r**'$H(y) = -\sum_j p_j \ln p_j$', 'Shannon entropy of y', PURPLE),

    (**r**'$H(x,y) = -\sum\_{ij} p\_{ij} \ln p\_{ij}$', 'Joint entropy', TEAL),

    (**r**'$U(x,y) = \frac{2[H(x)+H(y)-H(x,y)]}{H(x)+H(y)}$',

     'Normalised mutual info ∈ [0,1]', RED),

    (**r**'$= 0$: x,y independent', 'No shared information', GRAY),

    (**r**'$= 1$: x determines y', 'Perfect dependence', GREEN),

]

y_p = 0.88

for formula, desc, col in lines:

    ax.text(0.05, y_p, formula, fontsize=8.5, color=col,

            transform=ax.transAxes, family='monospace')

    ax.text(0.05, y_p-0.07, desc, fontsize=7.5, color=GRAY,

            transform=ax.transAxes, style='italic')

    y_p -= 0.16

ax.text(0.5, 0.02,

        'Fisher z-transform: $z=\\\frac{1}{2}\\\ln\\\frac{1+r}{1-r}$ maps $(-1,1)\\\to\\\mathbb{R}$',

        ha='center', fontsize=8, color=PURPLE, transform=ax.transAxes,

        bbox=dict(boxstyle='round', facecolor='#F3E5F5', edgecolor=PURPLE))

plt.tight_layout()

plt.savefig('/Users/keerthikesavan/Desktop/Course_Modules/Stats/Exercises_Solved_Stats_2026/diagram_q4.png', dpi=140, bbox_inches='tight', facecolor=BG)

plt.close()

print("diagram_q4 done")

*# ════════════════════════════════════════════════════════════════════════════*

*# DIAGRAM 5: Q5   Linear regression, SSE, MLE, BIC*

*# ════════════════════════════════════════════════════════════════════════════*

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

fig.patch.set_facecolor(BG)

fig.suptitle('Q5: Can I model y from x?   Linear regression, MLE, SSE, BIC',

             fontsize=12, fontweight='bold', color='#1a1a2e')

*# Panel 1: SSE = sum of squared residuals*

ax = axes[0]

np.random.seed(3)

x_r = np.linspace(0, 10, 15)

y_r = 2\*x_r + 3 + np.random.normal(0, 3, 15)

a_fit, b_fit = np.polyfit(x_r, y_r, 1)

y_pred = a_fit\*x_r + b_fit

ax.scatter(x_r, y_r, color=BLUE, s=60, zorder=5, label='observations')

x_line = np.linspace(0, 10, 100)

ax.plot(x_line, a_fit\*x_line + b_fit, color=RED, lw=2.5,

        label=**f**'fit: y = {b_fit**:.1f**} + {a_fit**:.1f**}x')

for xi, yi, ypi in zip(x_r, y_r, y_pred):

    ax.plot([xi, xi], [yi, ypi], color=GREEN, linewidth=1.3, alpha=0.7)

ax.scatter(x_r, y_pred, color=GREEN, s=20, zorder=4, alpha=0.6)

ax.text(5, -3,

**r**'$\text{SSE} = \sum_i (y_i - a - bx_i)^2$' + '\nminimise this → best fit',

        ha='center', fontsize=8.5,

        bbox=dict(boxstyle='round', facecolor='#EDFBF4', edgecolor=GREEN))

ax.legend(fontsize=7.5)

style(ax, title='SSE = sum of squared residuals (green lines)',

      xlabel='x', ylabel='y')

*# Panel 2: S sums and solution*

ax = axes[1]

ax.axis('off')

ax.set_facecolor(PANEL)

ax.set_title('S sums   from differentiating SSE', fontsize=10,

             fontweight='bold', color='#1a1a2e', pad=8)

content = [

    ('Why S sums exist:', **r**'Setting $\partial \text{SSE}/\partial a = 0$ and $\partial \text{SSE}/\partial b = 0$', GRAY),

    ('gives two equations with these sums:', '', GRAY),

    ('', '', ''),

    (**r**'$S = N$', 'count of observations', BLUE),

    (**r**'$S_x = \sum x_i$', 'sum of x values', BLUE),

    (**r**'$S_y = \sum y_i$', 'sum of y values', BLUE),

    (**r**'$S\_{xx} = \sum x_i^2$', 'sum of squared x', BLUE),

    (**r**'$S\_{xy} = \sum x_i y_i$', 'sum of x×y products', BLUE),

    ('', '', ''),

    (**r**'$t_i = x_i - \bar{x}$', 'centered x (deviation from mean)', TEAL),

    (**r**'$S\_{tt} = \sum t_i^2 \propto \text{Var}(x)$', 'spread of x', TEAL),

    (**r**'$S\_{ty} = \sum t_i y_i \propto \text{Cov}(x,y)$', 'covariance', TEAL),

    ('', '', ''),

    (**r**'$\hat{b} = S\_{ty}/S\_{tt}$', '= Cov(x,y)/Var(x)', RED),

    (**r**'$\hat{a} = \bar{y} - \hat{b}\bar{x}$', 'line through centroid', RED),

]

y_p = 0.96

for formula, desc, col in content:

    if formula:

        ax.text(0.03, y_p, formula, fontsize=8, color=col,

                transform=ax.transAxes, family='monospace')

        ax.text(0.5, y_p, desc, fontsize=7.5, color=GRAY,

                transform=ax.transAxes, style='italic')

    y_p -= 0.065

*# Panel 3: BIC model comparison*

ax = axes[2]

np.random.seed(5)

x_b = np.linspace(0, 10, 30)

y_b = 0.3\*x_b\*\*2 - 2\*x_b + 5 + np.random.normal(0, 1.5, 30)

*# fit line and parabola*

c1 = np.polyfit(x_b, y_b, 1)

c2 = np.polyfit(x_b, y_b, 2)

x_pl = np.linspace(0, 10, 200)

ax.scatter(x_b, y_b, color=BLUE, s=40, zorder=5, alpha=0.7, label='data')

ax.plot(x_pl, np.polyval(c1, x_pl), color=RED, lw=2,

        linestyle='--', label='linear (p=2)')

ax.plot(x_pl, np.polyval(c2, x_pl), color=GREEN, lw=2.5,

        label='parabolic (p=3, lower BIC)')

sse1 = np.sum((y_b - np.polyval(c1, x_b))\*\*2)

sse2 = np.sum((y_b - np.polyval(c2, x_b))\*\*2)

n = len(x_b)

bic1 = n\*np.log(sse1/n) + 2\*np.log(n)

bic2 = n\*np.log(sse2/n) + 3\*np.log(n)

ax.text(5, -2,

**f**'BIC linear = {bic1**:.1f**}\nBIC parabola = {bic2**:.1f**}\n'

**r**'$\text{BIC} = n\log\langle\sigma^2\_\epsilon\rangle + p\log(n)$'

        '\nLower BIC wins',

        ha='center', fontsize=8.5, color='#1a1a2e',

        bbox=dict(boxstyle='round', facecolor='#FFF8EE', edgecolor=ORANGE))

ax.legend(fontsize=7.5)

style(ax, title='BIC: penalise complexity, reward fit',

      xlabel='x', ylabel='y')

plt.tight_layout()

plt.savefig('/Users/keerthikesavan/Desktop/Course_Modules/Stats/Exercises_Solved_Stats_2026/diagram_q5.png', dpi=140, bbox_inches='tight', facecolor=BG)

plt.close()

print("diagram_q5 done")

*# ════════════════════════════════════════════════════════════════════════════*

*# DIAGRAM 6: Q6   Logistic regression*

*# ════════════════════════════════════════════════════════════════════════════*

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

fig.patch.set_facecolor(BG)

fig.suptitle('Q6: What if y is binary?   Logistic regression',

             fontsize=12, fontweight='bold', color='#1a1a2e')

*# Panel 1: Sigmoid and logit*

ax = axes[0]

lam = np.linspace(-6, 6, 300)

pi  = 1 / (1 + np.exp(-lam))

ax.plot(lam, pi, color=BLUE, lw=2.5, label=**r**'$\pi = \frac{1}{1+e^{-\lambda}}$ (sigmoid)')

ax.axhline(0.5, color=LGRAY, lw=1, linestyle=':')

ax.axvline(0, color=LGRAY, lw=1, linestyle=':')

ax.fill_between(lam, pi, 0.5, where=(lam > 0), color=GREEN, alpha=0.1)

ax.fill_between(lam, pi, 0.5, where=(lam < 0), color=RED, alpha=0.1)

ax.text(3, 0.3, **r**'$\lambda = \ln\frac{\pi}{1-\pi}$' + '\n(logit: inverse)',

        color=RED, fontsize=8.5,

        bbox=dict(boxstyle='round', facecolor='white', edgecolor=RED, alpha=0.9))

ax.set_ylim(-0.05, 1.1)

ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

ax.legend(fontsize=8)

style(ax, title='Sigmoid: maps ℝ → (0,1)',

      xlabel=**r**'log-odds $\lambda$', ylabel=**r**'probability $\pi$')

*# Panel 2: Design matrix*

ax = axes[1]

ax.axis('off')

ax.set_facecolor(PANEL)

ax.set_title('Design matrix X', fontsize=10, fontweight='bold',

             color='#1a1a2e', pad=8)

ax.text(0.5, 0.92, **r**'$\lambda_i = X_i \cdot \beta = \mu + \alpha_1 S_i + \alpha_2 A_i$',

        ha='center', fontsize=9, transform=ax.transAxes, color=BLUE,

        bbox=dict(boxstyle='round', facecolor='#EEF2FF', edgecolor=BLUE))

*# Draw matrix*

rows = [['1', 'S₁', 'A₁'],

        ['1', 'S₂', 'A₂'],

        ['⋮', '⋮', '⋮'],

        ['1', 'Sₙ', 'Aₙ']]

headers = ['intercept\n(always 1)', 'study\nhours S', 'attendance\nA']

col_colors = [RED, GREEN, PURPLE]

for j, (hdr, col) in enumerate(zip(headers, col_colors)):

    ax.text(0.2 + j\*0.22, 0.75, hdr, ha='center', fontsize=7.5,

            color=col, fontweight='bold', transform=ax.transAxes)

for i, row in enumerate(rows):

    for j, val in enumerate(row):

        bg = '#FFF0F0' if j == 0 else PANEL

        ax.text(0.2 + j\*0.22, 0.62 - i\*0.1, val, ha='center', fontsize=9,

                transform=ax.transAxes, color=col_colors[j],

                bbox=dict(boxstyle='round', facecolor=bg,

                          edgecolor=LGRAY, alpha=0.8, pad=0.3))

ax.text(0.5, 0.17,

**r**'First column = all 1s → estimates intercept $\beta_0$' + '\n' +

**r**'$S = X^TX$ must be invertible (full rank)' + '\n' +

**r**'No closed form → minimise $-\log L$ with BFGS',

        ha='center', fontsize=8, transform=ax.transAxes, color='#1a1a2e',

        bbox=dict(boxstyle='round', facecolor='#EDFBF4', edgecolor=GREEN))

*# Panel 3: Log likelihood*

ax = axes[2]

ax.axis('off')

ax.set_facecolor(PANEL)

ax.set_title('Negative log likelihood', fontsize=10, fontweight='bold',

             color='#1a1a2e', pad=8)

steps = [

    ('For each student i:', '', GRAY),

    (**r**'$\lambda_i = X_i \cdot \beta$', 'log-odds from model', BLUE),

    (**r**'$\pi_i = \frac{1}{1+e^{-\lambda_i}}$', 'predicted pass probability', TEAL),

    (**r**'$P(y_i|\pi_i) = \pi_i^{y_i}(1-\pi_i)^{1-y_i}$', 'Bernoulli likelihood', GREEN),

    ('Joint (all students):', '', GRAY),

    (**r**'$\log L = \sum_i [y_i\lambda_i - \log(1+e^{\lambda_i})]$', 'log likelihood', PURPLE),

    ('Minimise:', '', GRAY),

    (**r**'$-\log L = \sum_i [\log(1+e^{\lambda_i}) - y_i\lambda_i]$', 'what BFGS minimises', RED),

    ('', '', ''),

    (**r**'$\sigma^2 = n\pi(1-\pi)$', 'binomial variance', ORANGE),

    (**r**'$SE = \sqrt{\pi_i(1-\pi_i)/n_i}$', 'standard error of π', ORANGE),

]

y_p = 0.95

for formula, desc, col in steps:

    if formula:

        ax.text(0.03, y_p, formula, fontsize=8, color=col,

                transform=ax.transAxes, family='monospace')

        if desc:

            ax.text(0.03, y_p-0.055, desc, fontsize=7, color=GRAY,

                    transform=ax.transAxes, style='italic')

        y_p -= 0.1 if desc else 0.06

    else:

        y_p -= 0.03

plt.tight_layout()

plt.savefig('/Users/keerthikesavan/Desktop/Course_Modules/Stats/Exercises_Solved_Stats_2026/diagram_q6.png', dpi=140, bbox_inches='tight', facecolor=BG)

plt.close()

print("diagram_q6 done")

*# ════════════════════════════════════════════════════════════════════════════*

*# DIAGRAM 7: Q7   FDR / BH*

*# ════════════════════════════════════════════════════════════════════════════*

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

fig.patch.set_facecolor(BG)

fig.suptitle('Q7: What if I run many tests?   FDR correction (Benjamini-Hochberg)',

             fontsize=12, fontweight='bold', color='#1a1a2e')

*# Panel 1: BH plot*

ax = axes[0]

np.random.seed(2)

N = 20

q = 0.1

p_vals = np.sort(np.concatenate([

    np.random.uniform(0, 0.01, 5),

    np.random.uniform(0, 0.3, 15)

]))

ranks = np.arange(1, N+1)

bh_threshold = q \* ranks / N

significant = p_vals <= bh_threshold

k_max = np.max(np.where(significant)[0]) if significant.any() else -1

ax.plot(ranks/N, p_vals, 'o', color=BLUE, markersize=7, label='p-values', zorder=5)

ax.plot(ranks/N, bh_threshold, color=RED, lw=2,

        linestyle='--', label=**f**'BH line: slope q={q}')

ax.axhline(0.05, color=GRAY, lw=1, linestyle=':', alpha=0.5, label='α=0.05')

if k_max >= 0:

    ax.scatter(ranks[:k_max+1]/N, p_vals[:k_max+1],

               color=GREEN, s=80, zorder=6, label=**f**'significant (k≤{k_max+1})')

    ax.axvline(ranks[k_max]/N, color=GREEN, lw=1.5, linestyle=':', alpha=0.7)

ax.set_xlim(0, 0.5)

ax.set_ylim(0, 0.12)

ax.legend(fontsize=7.5)

style(ax, title='BH plot: find largest k where p_k ≤ q·(k/N)',

      xlabel='cumulative probability i/N', ylabel='p-value')

*# Panel 2: procedure steps*

ax = axes[1]

ax.axis('off')

ax.set_facecolor(PANEL)

ax.set_title('The BH procedure   4 steps', fontsize=10,

             fontweight='bold', color='#1a1a2e', pad=8)

steps_bh = [

    ('Step 1', 'Run all N tests. Collect p-values.', BLUE),

    ('Step 2', 'Sort p-values smallest to largest.\nAssign rank i = 1, 2, ..., N', TEAL),

    ('Step 3', 'For each rank i, compute BH threshold:\n'

**r**'$p_i^{thresh} = q \cdot \frac{i}{N}$', GREEN),

    ('Step 4', 'Find LARGEST k where $p_k \\\leq q \\\cdot k/N$\n'

     'Declare ALL ranks 1 to k significant\n(as a block, even if some are above threshold)', RED),

    ('Note', 'Bonferroni: $\\\alpha\_{corr} = \\\alpha/N$\n'

     'controls P(any false positive)   very conservative\n'

     'BH controls FDR: fraction of false among significant', ORANGE),

]

y_p = 0.93

for label, text, col in steps_bh:

    ax.text(0.03, y_p, label, fontsize=9, color=col, fontweight='bold',

            transform=ax.transAxes)

    ax.text(0.22, y_p, text, fontsize=8, color='#1a1a2e',

            transform=ax.transAxes)

    y_p -= 0.18

plt.tight_layout()

plt.savefig('/Users/keerthikesavan/Desktop/Course_Modules/Stats/Exercises_Solved_Stats_2026/diagram_q7.png', dpi=140, bbox_inches='tight', facecolor=BG)

plt.close()

print("diagram_q7 done")

*# ════════════════════════════════════════════════════════════════════════════*

*# DIAGRAM 8: Q8   Bootstrap and permutation*

*# ════════════════════════════════════════════════════════════════════════════*

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

fig.patch.set_facecolor(BG)

fig.suptitle('Q8: No distribution assumed   Permutation testing and Bootstrap',

             fontsize=12, fontweight='bold', color='#1a1a2e')

*# Panel 1: Permutation null distribution*

ax = axes[0]

np.random.seed(10)

wt  = np.random.normal(1.5, 0.8, 15)

ko  = np.random.normal(2.5, 0.8, 15)

t_obs = np.mean(ko) - np.mean(wt)

pooled = np.concatenate([wt, ko])

perm_stats = []

for \_ in range(3000):

    sh = np.random.permutation(pooled)

    perm_stats.append(np.mean(sh[15:]) - np.mean(sh[:15]))

perm_stats = np.array(perm_stats)

ax.hist(perm_stats, bins=40, color=BLUE, alpha=0.6, density=True,

        label='null distribution\n(permuted)')

ax.axvline(t_obs, color=RED, lw=2.5, label=**f**'T observed = {t_obs**:.2f**}')

p_val = np.mean(perm_stats >= t_obs)

ax.fill_between(np.sort(perm_stats),

                np.zeros_like(perm_stats),

                where=(np.sort(perm_stats) >= t_obs),

                color=RED, alpha=0.3)

ax.text(t_obs\*0.3, 0.6, **f**'p = {p_val**:.3f**}', color=RED, fontsize=10, fontweight='bold')

ax.text(0, -0.18,

**r**'$p = r / n_P$   where   $r = \sum [T_i \geq T_1]$',

        ha='center', fontsize=8.5, transform=ax.transAxes,

        bbox=dict(boxstyle='round', facecolor='#EEF2FF', edgecolor=BLUE))

ax.legend(fontsize=7.5)

style(ax, title='Permutation: shuffle labels WITHOUT replacement',

      xlabel='T = mean(KO) - mean(WT)', ylabel='density')

*# Panel 2: Bootstrap CI*

ax = axes[1]

np.random.seed(5)

data_b = np.concatenate([wt, ko])

theta_obs = np.mean(data_b)

boot_means = [np.mean(np.random.choice(data_b, len(data_b), replace=True))

              for \_ in range(2000)]

boot_means = np.array(boot_means)

ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

ax.hist(boot_means, bins=40, color=GREEN, alpha=0.6, density=True,

        label='bootstrap distribution')

ax.axvline(theta_obs, color=RED, lw=2.5, label=**f**'observed = {theta_obs**:.2f**}')

ax.axvline(ci_lo, color=ORANGE, lw=2, linestyle='--', label=**f**'95% CI: [{ci_lo**:.2f**}, {ci_hi**:.2f**}]')

ax.axvline(ci_hi, color=ORANGE, lw=2, linestyle='--')

ax.fill_between(np.sort(boot_means),

                np.zeros_like(boot_means),

                where=((np.sort(boot_means) >= ci_lo) & (np.sort(boot_means) <= ci_hi)),

                color=GREEN, alpha=0.2)

ax.text(0, -0.18,

**r**'$CI\_{95\\%} = [\theta^\*\_{(0.025B)},\ \theta^\*\_{(0.975B)}]$'

        '    sort replicates, take percentiles',

        ha='center', fontsize=8, transform=ax.transAxes,

        bbox=dict(boxstyle='round', facecolor='#EDFBF4', edgecolor=GREEN))

ax.legend(fontsize=7.5)

style(ax, title='Bootstrap: resample WITH replacement',

      xlabel='statistic θ\*', ylabel='density')

*# Panel 3: comparison table*

ax = axes[2]

ax.axis('off')

ax.set_facecolor(PANEL)

ax.set_title('Permutation vs Bootstrap', fontsize=10,

             fontweight='bold', color='#1a1a2e', pad=8)

rows_t = [

    ('', 'PERMUTATION', 'BOOTSTRAP', GRAY),

    ('Replacement', 'WITHOUT', 'WITH', BLUE),

    ('Purpose', 'test null\nhypothesis', 'estimate CI\nand SE', TEAL),

    ('What varies', 'group labels', 'which obs\nappear', GREEN),

    ('Output', 'p-value', 'SE, CI', RED),

    ('Preserves', 'actual values', 'group structure', ORANGE),

    ('When to use', 'normality violated\nor self-made stat', 'any stat\nno formula exists', PURPLE),

]

y_p = 0.92

for label, perm, boot, col in rows_t:

    if label == '':

        ax.text(0.35, y_p, perm, ha='center', fontsize=8.5, color=RED,

                fontweight='bold', transform=ax.transAxes)

        ax.text(0.72, y_p, boot, ha='center', fontsize=8.5, color=GREEN,

                fontweight='bold', transform=ax.transAxes)

    else:

        ax.text(0.02, y_p, label, fontsize=7.5, color=col,

                fontweight='bold', transform=ax.transAxes)

        ax.text(0.35, y_p, perm, ha='center', fontsize=7.5, color='#1a1a2e',

                transform=ax.transAxes)

        ax.text(0.72, y_p, boot, ha='center', fontsize=7.5, color='#1a1a2e',

                transform=ax.transAxes)

*#         ax.axhline(y_p-0.045, xmin=0.01, xmax=0.99,*

    y_p -= 0.12

plt.tight_layout()

plt.savefig('/Users/keerthikesavan/Desktop/Course_Modules/Stats/Exercises_Solved_Stats_2026/diagram_q8.png', dpi=140, bbox_inches='tight', facecolor=BG)

plt.close()

print("diagram_q8 done")

*# ════════════════════════════════════════════════════════════════════════════*

*# DIAGRAM 9: Q9   LDA*

*# ════════════════════════════════════════════════════════════════════════════*

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

fig.patch.set_facecolor(BG)

fig.suptitle('Q9: Many dimensions   Linear Discriminant Analysis',

             fontsize=12, fontweight='bold', color='#1a1a2e')

*# Panel 1: two group clouds with LDA direction*

ax = axes[0]

np.random.seed(42)

cov = np.array([[1.2, 0.3],[0.3, 0.8]])

g1 = np.random.multivariate_normal([0, 0], cov, 50)

g2 = np.random.multivariate_normal([3, 2], cov, 50)

ax.scatter(g1[:,0], g1[:,1], color=BLUE, alpha=0.5, s=25, label='Group 1')

ax.scatter(g2[:,0], g2[:,1], color=RED, alpha=0.5, s=25, label='Group 2')

m1 = g1.mean(axis=0)

m2 = g2.mean(axis=0)

ax.plot(\*m1, '\*', color=BLUE, markersize=15, markeredgecolor='white', zorder=6)

ax.plot(\*m2, '\*', color=RED,  markersize=15, markeredgecolor='white', zorder=6)

ax.text(m1[0]-0.3, m1[1]-0.5, **r**'$m_1$', color=BLUE, fontsize=10, fontweight='bold')

ax.text(m2[0]+0.1, m2[1]-0.5, **r**'$m_2$', color=RED, fontsize=10, fontweight='bold')

*# LDA direction*

C1 = np.cov(g1.T)

C2 = np.cov(g2.T)

w = np.linalg.inv(C1+C2) @ (m1-m2)

w = w / np.linalg.norm(w)

mid = (m1+m2)/2

t_vals = np.linspace(-4, 4, 100)

lda_x = mid[0] + t_vals\*w[0]

lda_y = mid[1] + t_vals\*w[1]

ax.plot(lda_x, lda_y, color=GREEN, lw=2.5, linestyle='--',

        label=**r**'$w \propto (C_1+C_2)^{-1}(m_1-m_2)$')

ax.annotate('', xy=mid+1.5\*w, xytext=mid,

            arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))

ax.set_aspect('equal')

ax.legend(fontsize=7.5, loc='upper right')

style(ax, title='LDA: find direction maximising separation',

      xlabel='dimension 1', ylabel='dimension 2')

*# Panel 2: projection onto w*

ax = axes[1]

proj1 = g1 @ w

proj2 = g2 @ w

ax.hist(proj1, bins=15, color=BLUE, alpha=0.6, density=True, label='Group 1 projected')

ax.hist(proj2, bins=15, color=RED, alpha=0.6, density=True, label='Group 2 projected')

ax.axvline(proj1.mean(), color=BLUE, lw=2, linestyle='--',

           label=**f**'m₁·w = {proj1.mean()**:.2f**}')

ax.axvline(proj2.mean(), color=RED, lw=2, linestyle='--',

           label=**f**'m₂·w = {proj2.mean()**:.2f**}')

s1 = np.std(proj1)

s2 = np.std(proj2)

dp = abs(proj1.mean()-proj2.mean()) / np.sqrt(s1\*\*2+s2\*\*2)

ax.text(0.5, 0.85, **f**"d' = {dp**:.2f**}",

        ha='center', fontsize=11, fontweight='bold', color=GREEN,

        transform=ax.transAxes,

        bbox=dict(boxstyle='round', facecolor='#EDFBF4', edgecolor=GREEN))

ax.text(0, -0.22,

**r**"$d' = \frac{m_1 - m_2}{\sqrt{\sigma_1^2 + \sigma_2^2}}$" +

        '   signal / noise   (= multivariate t-statistic)',

        ha='center', fontsize=8.5, transform=ax.transAxes,

        bbox=dict(boxstyle='round', facecolor='#EEF2FF', edgecolor=BLUE))

ax.legend(fontsize=7.5)

style(ax, title=**r**"Projected distributions   maximise d'",

      xlabel='projection onto w', ylabel='density')

*# Panel 3: formulas*

ax = axes[2]

ax.axis('off')

ax.set_facecolor(PANEL)

ax.set_title('LDA   3 steps + formulas', fontsize=10,

             fontweight='bold', color='#1a1a2e', pad=8)

steps_lda = [

    ('STEP 1: Centroids', **r**'$m_k = \frac{1}{n_k}\sum\_{x \in X_k} x$',

     'mean vector of each group', BLUE),

    ('STEP 1: Covariance matrices',

**r**'$C_k = \frac{1}{n_k}\sum\_{x \in X_k}(x-m_k)(x-m_k)^T$',

     'multivariate variance\n(diagonal=variance, off-diag=covariance)', TEAL),

    ('STEP 2: Project onto w',

**r**'$\tilde{x} = x \cdot w = x^Tw$',

     'dot product = projection\n= how far along w', GREEN),

    ('STEP 3: Objective',

**r**"$d' = \frac{m_1-m_2}{\sqrt{w^TC_1w + w^TC_2w}}$",

     'signal / noise (same as t-statistic)', ORANGE),

    ('OPTIMAL w',

**r**'$w \propto (C_1+C_2)^{-1}(m_1-m_2)$',

     '(C₁+C₂)⁻¹ corrects for covariance\nthen normalise: w = w/‖w‖', RED),

]

y_p = 0.95

for step, formula, desc, col in steps_lda:

    ax.text(0.03, y_p, step, fontsize=8, color=col, fontweight='bold',

            transform=ax.transAxes)

    ax.text(0.03, y_p-0.07, formula, fontsize=8.5, color=col,

            transform=ax.transAxes, family='monospace')

    ax.text(0.03, y_p-0.125, desc, fontsize=7, color=GRAY,

            transform=ax.transAxes, style='italic')

*#     ax.axhline(y_p-0.155, xmin=0.01, xmax=0.99,*

    y_p -= 0.185

plt.tight_layout()

plt.savefig('/Users/keerthikesavan/Desktop/Course_Modules/Stats/Exercises_Solved_Stats_2026/diagram_q9.png', dpi=140, bbox_inches='tight', facecolor=BG)

plt.close()

print("diagram_q9 done")

*# ════════════════════════════════════════════════════════════════════════════*

*# DIAGRAM: Mappings / Morphisms*

*# ════════════════════════════════════════════════════════════════════════════*

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

fig.patch.set_facecolor(BG)

fig.suptitle('The Morphisms   Transformations between spaces',

             fontsize=12, fontweight='bold', color='#1a1a2e')

*# 1. ln: R+ → R*

ax = axes[0,0]

x = np.linspace(0.01, 10, 300)

ax.plot(x, np.log(x), color=BLUE, lw=2.5)

ax.axhline(0, color=LGRAY, lw=1)

ax.axvline(1, color=RED, lw=1.5, linestyle='--', label='ln(1) = 0')

ax.fill_between(x, np.log(x), 0, where=(x > 1), color=GREEN, alpha=0.1)

ax.fill_between(x, np.log(x), 0, where=(x < 1), color=RED, alpha=0.1)

ax.text(5, -1.5, **r**'$\ln(a \times b) = \ln a + \ln b$' +

        '\nmultiplication → addition', fontsize=8.5,

        bbox=dict(boxstyle='round', facecolor='#EEF2FF', edgecolor=BLUE))

ax.legend(fontsize=8)

style(ax, title=**r**'$\ln$: $\mathbb{R}^+ \to \mathbb{R}$',

      xlabel='x (positive)', ylabel='ln(x)')

*# 2. Logit: (0,1) → R*

ax = axes[0,1]

pi = np.linspace(0.01, 0.99, 300)

logit = np.log(pi/(1-pi))

ax.plot(pi, logit, color=RED, lw=2.5)

ax.axhline(0, color=LGRAY, lw=1)

ax.axvline(0.5, color=BLUE, lw=1.5, linestyle='--', label='π=0.5 → logit=0')

ax.fill_between(pi, logit, 0, where=(pi > 0.5), color=GREEN, alpha=0.1)

ax.fill_between(pi, logit, 0, where=(pi < 0.5), color=RED, alpha=0.1)

ax.set_ylim(-5, 5)

ax.text(0.5, -3.5, **r**'$\lambda = \ln\frac{\pi}{1-\pi}$' +

        '\n(0,1) → ℝ: enables linear model', ha='center', fontsize=8.5,

        bbox=dict(boxstyle='round', facecolor='#FFF0F0', edgecolor=RED))

ax.legend(fontsize=8)

style(ax, title=**r**'Logit: $(0,1) \to \mathbb{R}$',

      xlabel='probability π', ylabel='log-odds λ')

*# 3. Sigmoid: R → (0,1)*

ax = axes[0,2]

lam = np.linspace(-6, 6, 300)

sig = 1/(1+np.exp(-lam))

ax.plot(lam, sig, color=GREEN, lw=2.5)

ax.axhline(0.5, color=LGRAY, lw=1)

ax.axhline(1, color=LGRAY, lw=1, linestyle=':')

ax.axhline(0, color=LGRAY, lw=1, linestyle=':')

ax.fill_between(lam, sig, 0.5, where=(lam > 0), color=GREEN, alpha=0.1)

ax.text(0, 0.1, **r**'$\pi = \frac{1}{1+e^{-\lambda}}$' +

        '\nℝ → (0,1): inverse of logit', ha='center', fontsize=8.5,

        bbox=dict(boxstyle='round', facecolor='#EDFBF4', edgecolor=GREEN))

style(ax, title=**r**'Sigmoid: $\mathbb{R} \to (0,1)$',

      xlabel='log-odds λ', ylabel='probability π')

*# 4. Squaring: R → R+*

ax = axes[1,0]

x = np.linspace(-3, 3, 200)

ax.plot(x, x\*\*2, color=PURPLE, lw=2.5)

ax.axhline(0, color=LGRAY, lw=1)

ax.fill_between(x, x\*\*2, color=PURPLE, alpha=0.1)

ax.text(0, 5, **r**'$x^2 \geq 0$ always' + '\nNormal → Chi-square\n'

**r**'$z \sim N(0,1) \Rightarrow z^2 \sim \chi^2_1$',

        ha='center', fontsize=8.5,

        bbox=dict(boxstyle='round', facecolor='#F3E5F5', edgecolor=PURPLE))

style(ax, title=**r**'Squaring: $\mathbb{R} \to \mathbb{R}^+$',

      xlabel='x (can be negative)', ylabel='x² (always positive)')

*# 5. Z-score: any normal → N(0,1)*

ax = axes[1,1]

x = np.linspace(-15, 20, 400)

for mu_s, sig_s, col, lab in [

    (0, 1, BLUE, 'N(0,1)'),

    (5, 2, GREEN, 'N(5,2)'),

    (10, 3, RED, 'N(10,3)'),

]:

    y = norm.pdf(x, mu_s, sig_s)

    ax.plot(x, y, color=col, lw=2, label=lab)

    ax.axvline(mu_s, color=col, lw=1, linestyle=':', alpha=0.5)

ax.text(3, 0.27, **r**'$z = \frac{x-\mu}{\sigma}$' +

        '\nmaps all to N(0,1)', fontsize=8.5,

        bbox=dict(boxstyle='round', facecolor='#EEF2FF', edgecolor=BLUE))

ax.legend(fontsize=8)

style(ax, title=**r**'Z-score: $(\mathbb{R},\mu,\sigma) \to N(0,1)$',

      xlabel='x', ylabel='density')

*# 6. Fisher z: (-1,1) → R*

ax = axes[1,2]

r_vals = np.linspace(-0.99, 0.99, 300)

z_fish = 0.5 \* np.log((1+r_vals)/(1-r_vals))

ax.plot(r_vals, z_fish, color=TEAL, lw=2.5)

ax.axhline(0, color=LGRAY, lw=1)

ax.axvline(0, color=LGRAY, lw=1)

*# SE bands*

ax.fill_between(r_vals, z_fish - 1/np.sqrt(50-3),

                z_fish + 1/np.sqrt(50-3),

                color=TEAL, alpha=0.15, label='±1 SE (n=50)')

ax.text(0, -2.5,

**r**'$z = \frac{1}{2}\ln\frac{1+r}{1-r}$' +

**r**'   $SE_z = \frac{1}{\sqrt{n-3}}$' +

        '\nflattens correlation manifold\nenables hypothesis testing on r',

        ha='center', fontsize=8,

        bbox=dict(boxstyle='round', facecolor='#E0F7FA', edgecolor=TEAL))

ax.set_ylim(-4, 4)

ax.legend(fontsize=8)

style(ax, title=**r**"Fisher z: $(-1,1) \to \mathbb{R}$",

      xlabel='Pearson r', ylabel="Fisher z")

plt.tight_layout()

plt.savefig('/Users/keerthikesavan/Desktop/Course_Modules/Stats/Exercises_Solved_Stats_2026/diagram_morphisms.png', dpi=140, bbox_inches='tight', facecolor=BG)

plt.close()

print("diagram_morphisms done")

print("\nAll diagrams saved to /Users/keerthikesavan/Desktop/Course_Modules/Stats/Exercises_Solved_Stats_2026/")
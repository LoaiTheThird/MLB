import math, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split


# ------------------------------------------------------------------------
# helper functions
# ------------------------------------------------------------------------
rng = np.random.default_rng(0)


def kl_div(p, q):
    """Binary KL divergence KL(p||q)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def kl_inv(p, C, iters=50):
    """
    Inverse binary-KL: smallest q ≥ p such that KL(p||q) = C.
    """
    lo, hi = p, 1 - 1e-12
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if kl_div(p, mid) > C:
            hi = mid
        else:
            lo = mid
    return hi


def alternating_minimization(L_vals, n_r, delta=0.05,
                             tol=1e-6, max_iter=1000):
    """
    Alternating minimisation of the PAC-Bayes-λ bound (Thm 6)
    to obtain posterior ρ and trade-off λ.
    """
    m = len(L_vals)
    π = np.full(m, 1 / m)

    # shift for numerical stability
    x = L_vals - L_vals.min()

    lam = 0.5
    for _ in range(max_iter):
        # --- ρ-update -----------------------------------------------------
        logits = -lam * n_r * x
        logits -= logits.max()                    # softmax stabiliser
        ρ = π * np.exp(logits)
        ρ /= ρ.sum()

        KL = (ρ * np.log(ρ / π)).sum()
        E_L = (ρ * L_vals).sum()
        ln_term = math.log(2 * math.sqrt(n_r) / delta)

        # --- λ-update -----------------------------------------------------
        lam_new = 2 / (math.sqrt(2 * n_r * E_L / (KL + ln_term)) + 1)

        if abs(lam - lam_new) < tol:
            lam = lam_new
            break
        lam = lam_new

    return ρ, lam


def gamma_seed_jaakkola(X, y):
    """
    Vectorised Jaakkola heuristic for the RBF kernel bandwidth γ.   # >>>
    """
    # pairwise squared Euclidean distances
    diff = X[:, None, :] - X[None, :, :]
    d2 = (diff ** 2).sum(-1)

    # for each xi take the nearest point of the *opposite* class
    opp_mask = y[:, None] != y[None, :]
    d2[~opp_mask] = np.inf
    G = np.min(d2, axis=1)

    return 1 / (2 * np.median(np.sqrt(G)) ** 2)


# ------------------------------------------------------------------------
# main experiment
# ------------------------------------------------------------------------
N_REP = 1        # set to 10‒20 to get mean ± std bands   # >>>

def run_once(rep_seed=0):
    rng = np.random.default_rng(rep_seed)

    # --------------------------------------------------------------------
    # data
    ion = fetch_openml(name='ionosphere', version=1,
                       as_frame=False, parser='liac-arff')
    X, y_raw = ion.data, ion.target
    y = np.where(y_raw == 'g', 1, -1)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, train_size=200, stratify=y, random_state=15 + rep_seed)

    n_tr, d = X_tr.shape
    r = d + 1
    n_r = n_tr - r

    # --------------------------------------------------------------------
    # baseline: CV-tuned RBF SVM
    γ_seed = gamma_seed_jaakkola(X_tr, y_tr)                       # >>>
    γ_grid = γ_seed * 10.0 ** np.array([-4, -2, 0, 2, 4])
    C_grid = np.logspace(-3, 3, 7)
    param_grid = {'C': C_grid, 'gamma': γ_grid}

    t0 = time.time()
    cv = GridSearchCV(SVC(kernel='rbf'), param_grid,
                      cv=5, n_jobs=1)                              # >>>
    cv.fit(X_tr, y_tr)
    baseline_time = time.time() - t0
    baseline_err = 1 - cv.score(X_te, y_te)

    # --------------------------------------------------------------------
    # PAC-Bayes aggregation
    ms = np.unique(np.logspace(0, np.log10(n_tr), 21, dtype=int))
    pb_err, pb_time, pb_bound = [], [], []

    for m in ms:
        t1 = time.time()
        val_err = []
        classifiers = []

        # --- build m weak SVMs -----------------------------------------
        for _ in range(m):
            idx = rng.choice(n_tr, size=r, replace=False)
            clf = SVC(kernel='rbf', C=1,
                      gamma=rng.choice(γ_grid))
            clf.fit(X_tr[idx], y_tr[idx])

            classifiers.append(clf)

            mask = np.ones(n_tr, bool)
            mask[idx] = False
            val_err.append(1 - clf.score(X_tr[mask], y_tr[mask]))

        val_err = np.array(val_err)

        # --- posterior ρ and λ -----------------------------------------
        ρ, λ = alternating_minimization(val_err, n_r)

        # majority-vote test error
        votes = sum(w * clf.predict(X_te) for w, clf in zip(ρ, classifiers))
        y_pred = np.where(votes > 0, 1, -1)
        pb_err.append((y_pred != y_te).mean())

        # PAC-Bayes-kl **with proper factors**                          # >>>
        π = np.full(m, 1 / m)
        KL = (ρ * np.log(ρ / π)).sum()
        L_hat = (ρ * val_err).sum()

        numer = KL + math.log(2 * math.sqrt(n_r) / 0.05)
        denom = n_r * λ * (1 - λ / 2)
        C_term = numer / denom

        q = kl_inv(L_hat / (1 - λ / 2), C_term)
        pb_bound.append(q)
        pb_time.append(time.time() - t1)

    return (ms, np.array(pb_err), np.array(pb_bound),
            np.array(pb_time), baseline_err, baseline_time)


# ------------------------------------------------------------------------
# run repetitions
# ------------------------------------------------------------------------
all_err, all_bnd, all_tim = [], [], []
for rep in range(N_REP):
    out = run_once(rep)
    all_err.append(out[1])
    all_bnd.append(out[2])
    all_tim.append(out[3])

ms = out[0]
baseline_err, baseline_time = out[4], out[5]

all_err, all_bnd, all_tim = map(np.vstack, (all_err, all_bnd, all_tim))

err_mean, err_std = all_err.mean(0), all_err.std(0)
bnd_mean, bnd_std = all_bnd.mean(0), all_bnd.std(0)
tim_mean, tim_std = all_tim.mean(0), all_tim.std(0)

# ------------------------------------------------------------------------
# plotting
# ------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(8, 5))

# test-loss curves --------------------------------------------------------
ax1.plot(ms, err_mean, '-k', label='Our Method')
if N_REP > 1:                                                         # >>>
    ax1.fill_between(ms, err_mean - err_std, err_mean + err_std,
                     alpha=0.15, color='k')

ax1.plot(ms, bnd_mean, '-b', label='PAC-Bayes-kl bound')
if N_REP > 1:
    ax1.fill_between(ms, bnd_mean - bnd_std, bnd_mean + bnd_std,
                     alpha=0.15, color='b')

ax1.axhline(baseline_err, color='red', label='CV SVM')
ax1.set_xscale('log')
ax1.set_xlabel('number of weak classifiers $m$')
ax1.set_ylabel('test loss')
ax1.set_ylim(0, 1)

# runtime curves ----------------------------------------------------------
ax2 = ax1.twinx()
ax2.plot(ms, tim_mean, '--k', label='PB aggregation time')
if N_REP > 1:
    ax2.fill_between(ms, tim_mean - tim_std, tim_mean + tim_std,
                     alpha=0.15, color='grey')
ax2.axhline(baseline_time, color='red', linestyle='--',
            label='CV SVM time')
ax2.set_ylabel('training time (s)')
ax2.set_ylim(0, 1)

# legend ------------------------------------------------------------------
lines, labels = [], []
for ax in (ax1, ax2):
    L, lab = ax.get_legend_handles_labels()
    lines.extend(L); labels.extend(lab)

ax1.legend(lines, labels, ncol=2, loc='upper left', frameon=False)
plt.title(f'Ionosphere dataset  (n = 200,  r = d + 1 = {int(out[0][0])})')
plt.tight_layout()
plt.show()

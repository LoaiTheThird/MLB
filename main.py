import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split


def kl_div(p, q):
    """Binary KL divergence: KL(p||q)"""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return p * math.log(p/q) + (1 - p) * math.log((1 - p)/(1 - q))


def kl_inv(p, C, iters=50):
    """Inverse binary KL: smallest q >= p such that kl_div(p,q) = C"""
    lo, hi = p, 1 - 1e-12
    for _ in range(iters):
        mid = (lo + hi) / 2
        if kl_div(p, mid) > C:
            hi = mid
        else:
            lo = mid
    return hi


def alternating_minimization(L_vals, n_r, delta=0.05, tol=1e-6, max_iter=1000):
    """
    Alternating minimization of the PAC-Bayes-λ bound to find posterior rho and trade-off λ.
    Returns posterior rho and λ.
    """
    m = len(L_vals)
    pi = np.ones(m) / m
    # numerical shift
    min_L = np.min(L_vals)
    x = L_vals - min_L

    lam = 0.5
    for _ in range(max_iter):
        # update rho
        expn = -lam * n_r * x
        expn -= np.max(expn)
        weights = pi * np.exp(expn)
        rho = weights / np.sum(weights)
        # KL and expected loss
        KL = np.sum(rho * np.log(rho / pi))
        E_L = np.dot(rho, L_vals)
        # update λ
        ln_term = math.log(2 * math.sqrt(n_r) / delta)
        lam_new = 2 / (math.sqrt( 2*n_r*E_L / (KL + ln_term) ) + 1 )
        if abs(lam - lam_new) < tol:
            lam = lam_new
            break
        lam = lam_new

    return rho, lam


def compute_gamma_seed(X_tr, y_tr):
    """Jaakkola heuristic for RBF gamma"""
    G = []
    for i, xi in enumerate(X_tr):
        dists = np.linalg.norm(X_tr[y_tr != y_tr[i]] - xi, axis=1)
        G.append(np.min(dists))
    return 1 / (2 * (np.median(G) ** 2))


def main():
    # Load Ionosphere (no pandas required)
    ion = fetch_openml(name='ionosphere', version=1, as_frame=False, parser='liac-arff')
    X, y_raw = ion.data, ion.target
    y = np.where(y_raw == 'g', 1, -1)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=200, stratify=y, random_state=42)
    n_train, d = X_train.shape
    n_test = X_test.shape[0]
    r = d + 1
    n_r = n_train - r

    # Baseline: CV-tuned RBF SVM
    gamma_seed = compute_gamma_seed(X_train, y_train)
    gamma_grid = gamma_seed * np.power(10.0, [-4, -2, 0, 2, 4])
    C_grid = np.logspace(-3, 3, 7)
    param_grid = {'C': C_grid, 'gamma': gamma_grid}
    t0 = time.time()
    cv = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, n_jobs=-1)
    cv.fit(X_train, y_train)
    baseline_time = time.time() - t0
    baseline_error = 1 - cv.score(X_test, y_test)

    # PAC-Bayesian aggregation
    ms = np.unique(np.logspace(0, np.log10(n_train), 20, dtype=int))
    pb_errors, kl_bounds, pb_times = [], [], []
    delta = 0.05

    for m in ms:
        t1 = time.time()
        classifiers, val_errors = [], []
        # train m weak SVMs on subsets of size r
        for _ in range(m):
            idx = np.random.choice(n_train, size=r, replace=False)
            clf = SVC(kernel='rbf', C=1, gamma=np.random.choice(gamma_grid))
            clf.fit(X_train[idx], y_train[idx])
            classifiers.append(clf)
            mask = np.ones(n_train, dtype=bool)
            mask[idx] = False
            val_errors.append(1 - clf.score(X_train[mask], y_train[mask]))
        val_errors = np.array(val_errors)

        # posterior and λ
        rho, lam = alternating_minimization(val_errors, n_r, delta=delta)
        # test error of majority vote
        votes = sum(w * clf.predict(X_test) for w, clf in zip(rho, classifiers))
        pb_errors.append(np.mean((votes > 0).astype(int)*2-1 != y_test))
        # PAC-Bayes-kl bound on randomized classifier
        pi = np.ones(m) / m
        KL = np.sum(rho * np.log(rho / pi))
        E_L = np.dot(rho, val_errors)
        Cterm = (KL + math.log(2 * math.sqrt(n_r) / delta)) / n_r
        kl_bounds.append(kl_inv(E_L, Cterm))
        pb_times.append(time.time() - t1)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(8, 5))
    l1, = ax1.plot(ms, pb_errors, '-k', label='Our Method')
    l2, = ax1.plot(ms, kl_bounds, '-b', label='PAC-Bayes-kl bound')
    l3 = ax1.axhline(baseline_error, color='red', linestyle='-', label='CV SVM')
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of weak classifiers $m$')
    ax1.set_ylabel('Test loss')
    ax1.set_ylim(0, 0.45)
    ax1.set_yticks(np.linspace(0, 0.45, 5))

    ax2 = ax1.twinx()
    l4, = ax2.plot(ms, pb_times, '--k', label='PB aggregation time')
    l5 = ax2.axhline(baseline_time, color='red', linestyle='--', label='CV SVM time')
    ax2.set_ylabel('Training time (s)')
    ax2.set_ylim(0, 0.8)
    ax2.set_xlim(ax1.get_xlim())

    lines = [l3, l1, l2, l4, l5]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc='upper left', ncol=2, frameon=False)

    plt.title(f'Ionosphere dataset (n={n_train + n_test}, r={r})')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

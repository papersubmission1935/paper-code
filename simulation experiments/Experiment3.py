import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_iid(
    n, d, p_list, theta_assign, gamma0, gamma1,
    seed, outcome_type
):
    np.random.seed(seed)
    X = np.random.binomial(1, p_list, size=(n, d))
    delta = X.dot(theta_assign)
    eps0 = np.random.gumbel(size=n)
    eps1 = np.random.gumbel(size=n)
    U0, U1 = eps0, delta + eps1
    T = (U1 > U0).astype(int)

    # potential outcomes
    if outcome_type == 'polynomial':
        Y0_full = X.dot(theta_assign) + (X.dot(gamma0))**2 + np.random.normal(size=n)
        Y1_full = X.dot(theta_assign) + (X.dot(gamma1))**2 + np.random.normal(size=n)
    elif outcome_type == 'logistic_x':
        Y0_full = sigmoid(X.dot(theta_assign)) + sigmoid(X.dot(gamma0)) + np.random.normal(size=n)
        Y1_full = sigmoid(X.dot(theta_assign)) + sigmoid(X.dot(gamma1)) + np.random.normal(size=n)
    elif outcome_type == 'sine_x':
        Y0_full = np.sin(X.dot(theta_assign)) + np.sin(X.dot(gamma0)) + np.random.normal(size=n)
        Y1_full = np.sin(X.dot(theta_assign)) + np.sin(X.dot(gamma1)) + np.random.normal(size=n)
    else:
        raise ValueError("Unknown outcome_type")

    Y_obs = T * Y1_full + (1 - T) * Y0_full
    return X, T, Y_obs, Y0_full, Y1_full

def estimate_all(n_reps=200):
    n, d = 1000, 4
    p_list = np.array([0.8, 0.2, 0.8, 0.2])
    theta_assign = np.array([3., -3., 3., -3.])
    outcomes = ['polynomial', 'logistic_x', 'sine_x']
    methods = ['naive', 'IPW', 'AIPW', 'MATCH']
    summary = []

    for outcome in outcomes:
        # storage arrays
        true_ate = np.zeros(n_reps)
        true_Y1 = np.zeros(n_reps)
        true_Y0 = np.zeros(n_reps)
        est = {m: { 'ATE': np.zeros(n_reps),
                    'Y1': np.zeros(n_reps),
                    'Y0': np.zeros(n_reps) }
               for m in methods}

        # random gamma fixed per outcome to isolate outcome effect
        gamma0 = np.random.rand(d)
        gamma1 = np.random.rand(d)

        for rep in range(n_reps):
            X, T, Y_obs, Y0_full, Y1_full = simulate_iid(
                n, d, p_list, theta_assign, gamma0, gamma1,
                rep, outcome
            )
            true_Y1[rep] = Y1_full.mean()
            true_Y0[rep] = Y0_full.mean()
            true_ate[rep] = true_Y1[rep] - true_Y0[rep]

            # Naive
            y1_naive = Y_obs[T == 1].mean()
            y0_naive = Y_obs[T == 0].mean()
            est['naive']['Y1'][rep] = y1_naive
            est['naive']['Y0'][rep] = y0_naive
            est['naive']['ATE'][rep] = y1_naive - y0_naive

            # IPW
            ps = LogisticRegression(max_iter=500).fit(X, T)
            e_hat = np.clip(ps.predict_proba(X)[:,1], 1e-3, 1-1e-3)
            ipw_y1 = np.mean(T * Y_obs / e_hat)
            ipw_y0 = np.mean((1 - T) * Y_obs / (1 - e_hat))
            est['IPW']['Y1'][rep] = ipw_y1
            est['IPW']['Y0'][rep] = ipw_y0
            est['IPW']['ATE'][rep] = ipw_y1 - ipw_y0

            # AIPW
            rf0 = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=rep)
            rf1 = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=rep)
            rf0.fit(X[T == 0], Y_obs[T == 0])
            rf1.fit(X[T == 1], Y_obs[T == 1])
            mu0 = rf0.predict(X)
            mu1 = rf1.predict(X)
            aipw_y1 = np.mean(mu1 + T * (Y_obs - mu1) / e_hat)
            aipw_y0 = np.mean(mu0 + (1 - T) * (Y_obs - mu0) / (1 - e_hat))
            est['AIPW']['Y1'][rep] = aipw_y1
            est['AIPW']['Y0'][rep] = aipw_y0
            est['AIPW']['ATE'][rep] = aipw_y1 - aipw_y0

            # Matching (ATT)
            nbrs = NearestNeighbors(n_neighbors=1).fit(e_hat[T == 0].reshape(-1,1))
            idx_treated = np.where(T == 1)[0]
            ps_treated = e_hat[idx_treated].reshape(-1,1)
            _, idx_ctrl = nbrs.kneighbors(ps_treated)
            matched_ctrl = np.where(T == 0)[0][idx_ctrl.flatten()]
            match_y1 = Y_obs[idx_treated].mean()
            match_y0 = Y_obs[matched_ctrl].mean()
            est['MATCH']['Y1'][rep] = match_y1
            est['MATCH']['Y0'][rep] = match_y0
            est['MATCH']['ATE'][rep] = match_y1 - match_y0

        # summarize biases and variances
        for m in methods:
            summary.append({
                'outcome': outcome,
                'method': m,
                'ATE_bias': est[m]['ATE'].mean() - true_ate.mean(),
                'Y1_bias':  est[m]['Y1'].mean()  - true_Y1.mean(),
                'Y0_bias':  est[m]['Y0'].mean()  - true_Y0.mean(),
                'ATE_var':  est[m]['ATE'].var()
            })

    return pd.DataFrame(summary)

# Run and display
df_summary = estimate_all()

print(df_summary) 
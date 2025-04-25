import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_two_stage_no_noise(
    n, d, p_list, theta_assign, gamma0, gamma1,
    stage1_size, p_low, p_high, seed, outcome_type
):
    np.random.seed(seed)
    X = np.random.binomial(1, p_list, size=(n, d))
    # Two-stage assignment with Gumbel noise
    delta = X.dot(theta_assign)
    eps0 = np.random.gumbel(size=n)
    eps1 = np.random.gumbel(size=n)
    U0, U1 = eps0, delta + eps1
    T1 = (U1[:stage1_size] > U0[:stage1_size]).astype(int)
    # Stage1 propensity model
    ps1 = LogisticRegression(max_iter=500).fit(X[:stage1_size], T1)
    theta_hat = ps1.coef_[0]
    # Generate outcomes based only on X
    if outcome_type == 'polynomial':
        Y0_full = X.dot(theta_assign) + (X.dot(gamma0))**2 + np.random.normal(size=n)
        Y1_full = X.dot(theta_assign) + (X.dot(gamma1))**2 + np.random.normal(size=n)
    elif outcome_type == 'logistic_x':
        Y0_full = sigmoid(X.dot(theta_assign))+sigmoid(X.dot(gamma0)) + np.random.normal(size=n)
        Y1_full = sigmoid(X.dot(theta_assign))+sigmoid(X.dot(gamma1)) + np.random.normal(size=n)
    elif outcome_type == 'sine_x':
        Y0_full = np.sin(X.dot(theta_assign))+np.sin(X.dot(gamma0))  + np.random.normal(size=n)
        Y1_full = np.sin(X.dot(theta_assign))+np.sin(X.dot(gamma1))  + np.random.normal(size=n)
    else:
        raise ValueError("Unknown outcome_type")
    # Observed stage1 outcomes
    Y_obs1 = T1 * Y1_full[:stage1_size] + (1 - T1) * Y0_full[:stage1_size]
    # Stage2 assignment with bonus
    X2 = X[stage1_size:]
    delta_hat2 = X2.dot(theta_hat)
    log_odds_low = np.log(p_low / (1 - p_low))
    log_odds_high = np.log(p_high / (1 - p_high))
    s = np.where(sigmoid(delta_hat2) < p_low,
                 log_odds_low - delta_hat2,
                 np.where(sigmoid(delta_hat2) > p_high,
                          log_odds_high - delta_hat2,
                          0.0))
    eps0_2 = np.random.gumbel(size=n-stage1_size)
    eps1_2 = np.random.gumbel(size=n-stage1_size)
    U0_2, U1_2 = eps0_2, delta[stage1_size:] + s + eps1_2
    T2 = (U1_2 > U0_2).astype(int)
    Y_obs2 = T2 * Y1_full[stage1_size:] + (1 - T2) * Y0_full[stage1_size:]
    # Combine
    T = np.concatenate([T1, T2])
    Y_obs = np.concatenate([Y_obs1, Y_obs2])
    return X, T, Y_obs, ps1, theta_hat, s, Y0_full, Y1_full

def fit_offset_logistic(X, s, T, lr=0.2, n_iter=500):
    theta = np.zeros(X.shape[1])
    for _ in range(n_iter):
        eta = X.dot(theta) + s
        p = sigmoid(eta)
        theta += lr * ((T - p) @ X) / len(T)
    return theta

# Simulation settings
n_reps = 200
n, d = 1000, 4
stage1_size = 300
p_list = np.array([0.8,0.2,0.8,0.2])
theta_assign = np.array([3.0, -3.0, 3.0, -3.0])
gamma0 = np.random.rand(d)
gamma1 = np.random.rand(d)
p_low, p_high = 0.4, 0.6

outcomes = ['polynomial', 'logistic_x', 'sine_x']
methods = ['naive', 'IPW_offset', 'AIPW_offset']

results = []
for outcome in outcomes:
    # Separate arrays for each estimator
    naive_ate = np.zeros(n_reps)
    naive_Y1 = np.zeros(n_reps)
    naive_Y0 = np.zeros(n_reps)
    ipw_ate = np.zeros(n_reps)
    ipw_Y1 = np.zeros(n_reps)
    ipw_Y0 = np.zeros(n_reps)
    aipw_ate = np.zeros(n_reps)
    aipw_Y1 = np.zeros(n_reps)
    aipw_Y0 = np.zeros(n_reps)
    true_ate = np.zeros(n_reps)
    true_Y1 = np.zeros(n_reps)
    true_Y0 = np.zeros(n_reps)

    for rep in range(n_reps):
        X, T, Y_obs, ps1, theta_hat, s, Y0, Y1 = simulate_two_stage_no_noise(
            n, d, p_list, theta_assign, gamma0, gamma1,
            stage1_size, p_low, p_high, rep, outcome
        )
        true_ate[rep] = np.mean(Y1 - Y0)
        true_Y1[rep] = np.mean(Y1)
        true_Y0[rep] = np.mean(Y0)

        # Naive
        naive_ate[rep] = np.mean(Y_obs[T==1]) - np.mean(Y_obs[T==0])
        naive_Y1[rep] = np.mean(Y_obs[T==1])
        naive_Y0[rep] = np.mean(Y_obs[T==0])

        # IPW with offset
        X2 = X[stage1_size:]
        T2 = T[stage1_size:]
        theta_off = fit_offset_logistic(X2, s, T2)
        e1 = np.clip(ps1.predict_proba(X[:stage1_size])[:,1],1e-3,1-1e-3)
        e2 = np.clip(sigmoid(X2.dot(theta_off) + s),1e-3,1-1e-3)
        e = np.concatenate([e1, e2])
        ipw_ate[rep] = np.mean(T * Y_obs / e - (1-T) * Y_obs / (1-e))
        ipw_Y1[rep] = np.mean(T * Y_obs / e)
        ipw_Y0[rep] = np.mean((1-T) * Y_obs / (1-e))

        # AIPW
        rf0 = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=rep)
        rf1 = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=rep)
        rf0.fit(X2[T2==0], Y_obs[stage1_size:][T2==0])
        rf1.fit(X2[T2==1], Y_obs[stage1_size:][T2==1])
        mu0 = rf0.predict(X)
        mu1 = rf1.predict(X)
        aipw_Y1[rep] = np.mean(mu1 + T * (Y_obs - mu1) / e)
        aipw_Y0[rep] = np.mean(mu0 + (1-T) * (Y_obs - mu0) / (1-e))
        aipw_ate[rep] = np.mean(mu1 - mu0 + T * (Y_obs - mu1) / e - (1-T) * (Y_obs - mu0) / (1-e))

    results.append(pd.DataFrame({
        'outcome': outcome,
        'method': methods,
        'ATE bias': [naive_ate.mean() - true_ate.mean(),
                     ipw_ate.mean() - true_ate.mean(),
                     aipw_ate.mean() - true_ate.mean()],
        'Y1 bias': [naive_Y1.mean() - true_Y1.mean(),
                    ipw_Y1.mean() - true_Y1.mean(),
                    aipw_Y1.mean() - true_Y1.mean()],
        'Y0 bias': [naive_Y0.mean() - true_Y0.mean(),
                    ipw_Y0.mean() - true_Y0.mean(),
                    aipw_Y0.mean() - true_Y0.mean()],
        'variance': [naive_ate.var(), ipw_ate.var(), aipw_ate.var()]
    }))

summary = pd.concat(results, ignore_index=True)

print(summary) 
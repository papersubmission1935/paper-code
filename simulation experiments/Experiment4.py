
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_one_stage(
    n, d, p_list, theta_assign, gamma0, gamma1,
    seed, outcome_type
):
    np.random.seed(seed)
    # 1. covariates X
    X = np.random.binomial(1, p_list, size=(n, d))
    # 2. latent utilities with Gumbel noise
    delta = X.dot(theta_assign)
    eps0  = np.random.gumbel(size=n)
    eps1  = np.random.gumbel(size=n)
    U0, U1 = eps0, delta + eps1
    # 3. treatment assignment (single stage)
    T = (U1 > U0).astype(int)
    # 4. potential outcomes (noise added)
    if outcome_type == 'continuous':
        Y0_full = U0 + (X.dot(gamma0))**2 + X.dot(gamma0) + np.random.normal(size=n)
        Y1_full = U1 + (X.dot(gamma1))**2 + X.dot(gamma1) + np.random.normal(size=n)
    elif outcome_type == 'logistic':
        Y0_full = sigmoid(U0) + sigmoid(X.dot(gamma0)) + np.random.normal(size=n)
        Y1_full = sigmoid(U1) + sigmoid(X.dot(gamma1)) + np.random.normal(size=n)
    elif outcome_type == 'sine':
        Y0_full = np.sin(U0) + np.sin(X.dot(gamma0)) + np.random.normal(size=n)
        Y1_full = np.sin(U1) + np.sin(X.dot(gamma1)) + np.random.normal(size=n)
    else:
        raise ValueError("Unknown outcome_type")
    # 5. observe
    Y_obs = T * Y1_full + (1 - T) * Y0_full

    return X, T, Y_obs, Y0_full, Y1_full

# simulation setting 
n_reps       = 100
n, d         = 1000, 4
p_list       = np.array([0.8, 0.2, 0.8, 0.2])
theta_assign = np.array([3.0, -3.0, 3.0, -3.0])

outcomes     = ['continuous', 'logistic', 'sine']
methods      = ['naive', 'IPW', 'AIPW']
all_runs     = []
n_gamma_runs = 20

for run in range(n_gamma_runs):
    gamma0 = np.random.rand(d)
    gamma1 = np.random.rand(d)

    run_results = []
    for outcome in outcomes:
        
        true_ate    = np.zeros(n_reps)
        true_Y1     = np.zeros(n_reps)
        true_Y0     = np.zeros(n_reps)
        naive_ate   = np.zeros(n_reps)
        naive_Y1    = np.zeros(n_reps)
        naive_Y0    = np.zeros(n_reps)
        ipw_ate     = np.zeros(n_reps)
        ipw_Y1      = np.zeros(n_reps)
        ipw_Y0      = np.zeros(n_reps)
        aipw_ate    = np.zeros(n_reps)
        aipw_Y1     = np.zeros(n_reps)
        aipw_Y0     = np.zeros(n_reps)

        for rep in range(n_reps):
            X, T, Y_obs, Y0_full, Y1_full = simulate_one_stage(
                n, d, p_list, theta_assign, gamma0, gamma1,
                rep, outcome
            )
            # True value 
            true_Y1[rep] = Y1_full.mean()
            true_Y0[rep] = Y0_full.mean()
            true_ate[rep] = true_Y1[rep] - true_Y0[rep]

            # 1. Naive
            y1n = Y_obs[T==1].mean()
            y0n = Y_obs[T==0].mean()
            naive_Y1[rep] = y1n
            naive_Y0[rep] = y0n
            naive_ate[rep] = y1n - y0n

            # 2. IPW
            ps = LogisticRegression(max_iter=500).fit(X, T)
            e_hat = np.clip(ps.predict_proba(X)[:,1], 1e-3, 1-1e-3)
            y1_ipw = np.mean(T * Y_obs / e_hat)
            y0_ipw = np.mean((1-T) * Y_obs / (1-e_hat))
            ipw_Y1[rep]  = y1_ipw
            ipw_Y0[rep]  = y0_ipw
            ipw_ate[rep] = y1_ipw - y0_ipw

            # 3. AIPW
            rf0 = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=rep)
            rf1 = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=rep)
            rf0.fit(X[T==0], Y_obs[T==0])
            rf1.fit(X[T==1], Y_obs[T==1])
            mu0 = rf0.predict(X)
            mu1 = rf1.predict(X)
            y1_aipw = np.mean(mu1 + T * (Y_obs - mu1) / e_hat)
            y0_aipw = np.mean(mu0 + (1-T) * (Y_obs - mu0) / (1-e_hat))
            aipw_Y1[rep]  = y1_aipw
            aipw_Y0[rep]  = y0_aipw
            aipw_ate[rep] = y1_aipw - y0_aipw

        # Collect results for the current run
        df = pd.DataFrame({
            'outcome': outcome,
            'method': methods,
            'ATE_bias': [
                naive_ate.mean() - true_ate.mean(),
                ipw_ate.mean()   - true_ate.mean(),
                aipw_ate.mean()  - true_ate.mean()
            ],
            'Y1_bias': [
                naive_Y1.mean() - true_Y1.mean(),
                ipw_Y1.mean()   - true_Y1.mean(),
                aipw_Y1.mean()  - true_Y1.mean()
            ],
            'Y0_bias': [
                naive_Y0.mean() - true_Y0.mean(),
                ipw_Y0.mean()   - true_Y0.mean(),
                aipw_Y0.mean()  - true_Y0.mean()
            ],
            'ATE_var': [
                naive_ate.var(),
                ipw_ate.var(),
                aipw_ate.var()
            ]
        })
        run_results.append(df)

    all_runs.append(pd.concat(run_results, ignore_index=True))

# Take average 
combined   = pd.concat(all_runs, keys=range(n_gamma_runs))
avg_summary = (
    combined
    .groupby(['outcome','method'])
    .mean()
    .reset_index()
)

print(avg_summary)
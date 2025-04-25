import pandas as pd
import numpy as np
#import ace_tools as tools
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# Load data
df = pd.read_csv("No-incentive-data.csv", encoding="utf-8")
cols = df.columns.tolist()
pref_col = next(c for c in cols if 'preferred movie genre' in c.lower())
choice_col = next(c for c in cols if c.startswith('6.If you can choose'))

# Preprocess
df[pref_col] = df[pref_col].str.strip().str.title()
df[choice_col] = df[choice_col].str.strip().str.title()
df['pref_binary'] = (df[pref_col]=='Sci-Fi').astype(int)
df['T'] = (df[choice_col]=='Sci-Fi').astype(int)
# compute means
t_idx = cols.index(choice_col)
sci_pref = cols[t_idx+1:t_idx+7]
fan_pref = cols[t_idx+7:t_idx+13]
fan_pref2 = cols[t_idx+13:t_idx+19]
sci_pref2 = cols[t_idx+19:t_idx+25]
df['sci_mean'] = df.apply(lambda r: pd.to_numeric(r[sci_pref], errors='coerce').mean() if r['pref_binary']==1 else pd.to_numeric(r[sci_pref2], errors='coerce').mean(), axis=1)
df['fan_mean'] = df.apply(lambda r: pd.to_numeric(r[fan_pref], errors='coerce').mean() if r['pref_binary']==1 else pd.to_numeric(r[fan_pref2], errors='coerce').mean(), axis=1)
df = df.dropna(subset=['sci_mean','fan_mean'])

# features
age_col = next(c for c in cols if 'age range' in c.lower())
sex_col = next(c for c in cols if 'sex' in c.lower())
enjoy_col = next(c for c in cols if 'enjoy imaginative or emotional' in c.lower())
curious_col = next(c for c in cols if 'curious are you about scientific' in c.lower())
df[age_col] = df[age_col].str.strip()
df[sex_col] = df[sex_col].str.strip().str.title()
X = pd.concat([
    pd.get_dummies(df[[age_col, sex_col]], drop_first=True),
    df[[enjoy_col, curious_col, 'pref_binary']]
], axis=1).fillna(0)

def estimators(df, X):
  # True and Naive
  Y1_true = df['sci_mean'].mean()
  Y0_true = df['fan_mean'].mean()
  Y1_naive = df.loc[df['T']==1,'sci_mean'].mean()
  Y0_naive = df.loc[df['T']==0,'fan_mean'].mean()

  logit = LogisticRegression(max_iter=1000).fit(X, df['T'])
  df['e_hat_logit'] = logit.predict_proba(X)[:,1]
  df['strata3'] = pd.qcut(df['e_hat_logit'],3,labels=False,duplicates='drop')
  rf1 = RandomForestRegressor(random_state=42).fit(X.loc[df['T']==1], df.loc[df['T']==1,'sci_mean'])
  rf0 = RandomForestRegressor(random_state=42).fit(X.loc[df['T']==0], df.loc[df['T']==0,'fan_mean'])
  N = len(df)

  # PSM Nearest ATT
  nn = NearestNeighbors(n_neighbors=1).fit(df['e_hat_logit'].values.reshape(-1,1)[df['T']==0])
  dist, ind = nn.kneighbors(df['e_hat_logit'].values.reshape(-1,1)[df['T']==1])
  t_idx_arr = df.index[df['T']==1]
  c_idx_arr = df.index[df['T']==0][ind.flatten()]
  Y1_att = df.loc[t_idx_arr,'sci_mean'].mean()
  Y0_att = rf0.predict(X.loc[t_idx_arr]).mean()

  # IPW/AIPW with logistic PS
  e = df['e_hat_logit']
  ipw_s_logit = np.mean(df['T']*df['sci_mean']/e)
  ipw_f_logit = np.mean((1-df['T'])*df['fan_mean']/(1-e))
  mu1 = rf1.predict(X); mu0 = rf0.predict(X)
  aipw_s_logit = np.mean(mu1 + df['T']*(df['sci_mean']-mu1)/e)
  aipw_f_logit = np.mean(mu0 + (1-df['T'])*(df['fan_mean']-mu0)/(1-e))

  # IPW/AIPW with RF PS
  rf_ps = RandomForestClassifier(random_state=42).fit(X, df['T'])
  e_rf = rf_ps.predict_proba(X)[:,1]
  df['e_hat_rf'] = e_rf
  ipw_s_rf = np.mean(df['T']*df['sci_mean']/e_rf)
  ipw_f_rf = np.mean((1-df['T'])*df['fan_mean']/(1-e_rf))
  aipw_s_rf = np.mean(mu1 + df['T']*(df['sci_mean']-mu1)/e_rf)
  aipw_f_rf = np.mean(mu0 + (1-df['T'])*(df['fan_mean']-mu0)/(1-e_rf))

  # Compile
  methods = [
      ('True',Y1_true,Y0_true),
      ('Naive',Y1_naive,Y0_naive),
      ('Strat3',Y1_strat,Y0_strat),
      ('Exact',Y1_exact,Y0_exact),
      ('PSM_ATT',Y1_att,Y0_att),
      ('IPW_Logit',ipw_s_logit,ipw_f_logit),
      ('AIPW_Logit',aipw_s_logit,aipw_f_logit),
      ('IPW_RF',ipw_s_rf,ipw_f_rf),
      ('AIPW_RF',aipw_s_rf,aipw_f_rf),
  ]
  res = pd.DataFrame(methods, columns=['Method','Y1','Y0'])
  return res
res=estimators(df,X)
res['ATE'] = (res['Y1']-res['Y0']).round(3)
true_vals = res.loc[res['Method'] == 'True', ['Y1','Y0','ATE']].iloc[0]

# Create a formatted DataFrame with bias in parentheses
formatted = res.copy()
for col in ['Y1','Y0','ATE']:
    def format_with_bias(x):
        bias = x - true_vals[col]
        return f"{x:.3f} ({bias:+.3f})"
    formatted[col] = formatted[col].apply(format_with_bias)
print(formatted)

#Bootstrap



def bootstrap_variance(df, X, B=200, random_state=None):
    """
    Bootstrap B times to estimate the variance of each estimator’s Y1 and Y0.

    Parameters
    ----------
    df : pd.DataFrame
        Your original data (must include columns 'T', 'sci_mean', 'fan_mean', etc.).
    X : pd.DataFrame or array-like
        The covariate matrix used in your estimators.
    B : int
        Number of bootstrap replications.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    var_df : pd.DataFrame
        DataFrame with columns ['Method', 'Var_Y1', 'Var_Y0'] giving the bootstrap
        variances for each estimator.
    """
    rng = np.random.RandomState(random_state)

    # first run on the full sample to get the list of method names
    base = estimators(df, X)
    methods = base['Method'].tolist()
    M = len(methods)

    # to store bootstrap estimates: shape (B, M, 2)
    boot = np.zeros((B, M, 2))

    n = len(df)
    idx = df.index.values

    for b in range(B):
        # draw a bootstrap sample of indices
        sample_idx = rng.choice(idx, size=n, replace=True)

        # subset & reset index
        df_bs = df.loc[sample_idx].reset_index(drop=True)
        # if X is a DataFrame:
        try:
            X_bs = X.loc[sample_idx].reset_index(drop=True)
        except:
            # if X is numpy array
            X_bs = X[sample_idx]

        # compute all estimators on the bootstrap sample
        res_bs = estimators(df_bs, X_bs)

        # store Y1 and Y0
        boot[b, :, 0] = res_bs['Y1'].values
        boot[b, :, 1] = res_bs['Y0'].values

    # compute variances across the B replications
    var_Y1 = boot[:, :, 0].var(axis=0, ddof=1)
    var_Y0 = boot[:, :, 1].var(axis=0, ddof=1)

    var_df = pd.DataFrame({
        'Method': methods,
        'Var_Y1': var_Y1,
        'Var_Y0': var_Y0
    })
    return var_df
var_estimates = bootstrap_variance(df, X, B=200, random_state=42)
print(var_estimates)
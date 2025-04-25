import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
#from ace_tools import display_dataframe_to_user
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm



df = pd.read_csv("With-incentive-data.csv", encoding="utf-8")


# Load incentive dataset
cols = df.columns.tolist()

# Identify columns
pref_col = next(c for c in cols if 'preferred movie genre' in c.lower())
choice1 = next(c for c in cols if c.startswith('36.If you can choose'))
choice2 = next(c for c in cols if c.startswith('61.If you can choose'))

# Preprocess and map actual choice
df[pref_col] = df[pref_col].str.strip().str.title()
df[choice1] = df[choice1].str.strip().str.title()
df[choice2] = df[choice2].str.strip().str.title()
df['choice'] = df.apply(lambda r: r[choice1] if r[pref_col] == 'Sci-Fi' else r[choice2], axis=1)
df['T'] = (df['choice'] == 'Sci-Fi').astype(int)


# 2. sci_mean and fan_mean
def cols_range(start, end):
    return [c for c in cols if any(c.startswith(f"{i}.") for i in range(start, end+1))]
ranges = {
    'sci1': cols_range(37,42), 'fan1': cols_range(43,48),
    'fan2': cols_range(49,54), 'sci2': cols_range(55,60),
    'fan3': cols_range(62,67), 'sci3': cols_range(68,73),
    'sci4': cols_range(74,79), 'fan4': cols_range(80,85)
}
sci_means, fan_means = [], []
for _, r in df.iterrows():
    if r[pref_col]=='Sci-Fi':
        sci_vals = r[ranges['sci1']] if r['T']==1 else r[ranges['sci2']]
        fan_vals = r[ranges['fan1']] if r['T']==1 else r[ranges['fan2']]
    else:
        sci_vals = r[ranges['sci3']] if r['T']==0 else r[ranges['sci4']]
        fan_vals = r[ranges['fan3']] if r['T']==0 else r[ranges['fan4']]
    sci_means.append(pd.to_numeric(sci_vals, errors='coerce').mean())
    fan_means.append(pd.to_numeric(fan_vals, errors='coerce').mean())
df['sci_mean'] = sci_means; df['fan_mean'] = fan_means

df['sci_mean'] = sci_means
df['fan_mean'] = fan_means

# Overall means
overall_sci = df['sci_mean'].mean()
overall_fan = df['fan_mean'].mean()

# Naive group means
naive_sci = df[df['T']==1]['sci_mean'].mean()
naive_fan = df[df['T']==0]['fan_mean'].mean()

# Prepare features
age_col = '31.Your age range'
sex_col = '32.Sex'
enjoy_col = '33.How much do you enjoy imaginative or emotional materials? '
curious_col = '34.How curious are you about scientific or technological progress? '
df['pref_binary'] = (df[pref_col] == 'Sci-Fi').astype(int)

X_ps = pd.concat([
    pd.get_dummies(df[[age_col, sex_col]].apply(lambda x: x.str.strip()), drop_first=True),
    df[[enjoy_col, curious_col, 'pref_binary']]
], axis=1).fillna(0)
y = df['T']

bool_cols = X_ps.select_dtypes(include='bool').columns
X_ps[bool_cols] = X_ps[bool_cols].astype(int)
# 5. propensity score
ps_model = LogisticRegression().fit(X_ps, df['T'])
ps = ps_model.predict_proba(X_ps)[:,1]


# 6. Logit model
X_sm = sm.add_constant(X_ps)
logit_res = sm.Logit(df['T'], X_sm).fit(disp=False)
coef_table = pd.DataFrame({
    'coef': logit_res.params,
    'ci_lower': logit_res.conf_int()[0],
    'ci_upper': logit_res.conf_int()[1]
})
#tools.display_dataframe_to_user('Logistic Regression Coefficients', coef_table)

# 7. Estimator
def estimators1(df, X):





    # PS
    e_logit = LogisticRegression(max_iter=1000).fit(X,df['T']).predict_proba(X)[:,1]
    e_rf    = RandomForestClassifier(random_state=42).fit(X,df['T']).predict_proba(X)[:,1]
    # Outcomes
    # Outcome models for AIPW: drop NaNs correctly
    df_t1 = df[(df['T']==1) & df['sci_mean'].notna()]
    df_t0 = df[(df['T']==0) & df['fan_mean'].notna()]
    rf1 = RandomForestRegressor(random_state=42).fit(X.loc[df_t1.index], df_t1['sci_mean'])
    rf0 = RandomForestRegressor(random_state=42).fit(X.loc[df_t0.index], df_t0['fan_mean'])
    mu1 = rf1.predict(X)
    mu0 = rf0.predict(X)
    #rf1 = RandomForestRegressor(random_state=42).fit(X[df['T']==1], df.loc[df['T']==1,'sci_mean'])
    #rf0 = RandomForestRegressor(random_state=42).fit(X[df['T']==0], df.loc[df['T']==0,'fan_mean'])
    #mu1, mu0 = rf1.predict(X), rf0.predict(X)
    res = {}
    # True & Naive
    res['True'] = (df['sci_mean'].mean(), df['fan_mean'].mean())
    res['Naive'] = (df[df['T']==1]['sci_mean'].mean(), df[df['T']==0]['fan_mean'].mean())

    # PSM-ATT
    nn = NearestNeighbors(n_neighbors=1).fit(e_logit[df['T']==0].reshape(-1,1))
    _,ind= nn.kneighbors(e_logit[df['T']==1].reshape(-1,1))
    tid = df.index[df['T']==1]; cid = df.index[df['T']==0][ind.flatten()]
    res['PSM_ATT'] = (df.loc[tid,'sci_mean'].mean(), rf0.predict(X.loc[tid]).mean())
    # IPW & AIPW
    for tag,ps in [('IPW_Logit', e_logit), ('IPW_RF', e_rf)]:
        res[tag] = ((df['T']*df['sci_mean']/ps).mean(),
                    ((1-df['T'])*df['fan_mean']/(1-ps)).mean())
    for tag,ps in [('AIPW_Logit', e_logit), ('AIPW_RF',   e_rf)]:
        res[tag] = ((mu1 + df['T']*(df['sci_mean']-mu1)/ps).mean(),
                    (mu0 + (1-df['T'])*(df['fan_mean']-mu0)/(1-ps)).mean())
    return res

res = estimators1(df, X_ps)

df_res = pd.DataFrame.from_dict(res, orient='index', columns=['Y1','Y0'])
df_res['ATE'] = (df_res['Y1'] - df_res['Y0']).round(3)
df_res.reset_index(inplace=True)
df_res.rename(columns={'index':'Method'}, inplace=True)

true_vals = df_res.loc[df_res['Method'] == 'True', ['Y1','Y0','ATE']].iloc[0]

# Create a formatted DataFrame with bias in parentheses
formatted = df_res.copy()
for col in ['Y1','Y0','ATE']:
    def format_with_bias(x):
        bias = x - true_vals[col]
        return f"{x:.3f} ({bias:+.3f})"
    formatted[col] = formatted[col].apply(format_with_bias)
print(formatted)
# Bootstrap
np.random.seed(42)
B = 200
boot = [estimators1(df.sample(frac=1, replace=True),X_ps) for _ in range(B)]
records = []
for b in boot:
    for method, (y1, y0) in b.items():
        records.append({
            'Method': method,
            'Y1': y1,
            'Y0': y0,
            'ATE': y1 - y0
        })
df_boot = pd.DataFrame(records)

# 2. Compute variances (ddof=1 for unbiased)
var_df = df_boot.groupby('Method').agg(
    Var_Y1=('Y1', lambda x: x.var(ddof=1)),
    Var_Y0=('Y0', lambda x: x.var(ddof=1)),
    Var_ATE=('ATE', lambda x: x.var(ddof=1))
).reset_index()



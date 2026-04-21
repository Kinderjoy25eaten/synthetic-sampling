import pandas as pd
import numpy as np

def pull_random(data, n=10000, seed=1):
    return data.sample(n=n, random_state=seed)

def custom_seq_sample(data, n=2500, seed=1):
    np.random.seed(seed)
    tot = len(data)
    
    a = 0 if np.random.rand() < 0.5 else np.random.randint(1, 15)
    
    if a == 0:
        step = tot // n
        start = np.random.randint(0, step)
        b, c = step, start
    else:
        b = np.random.randint(1, 100)
        c = np.random.randint(1, 500)
        
    i_vals = np.arange(1, (n * 10) + 1)
    pos = (a * (i_vals**2) + b * i_vals + c) % tot
    
    uniq = pd.unique(pos)[:n]
    return data.iloc[uniq]

def bias_cluster_sample(data, col='Education', keep=3, n=10000, seed=1):
    cats = data[col].unique()
    np.random.seed(seed)
    
    chosen = np.random.choice(cats, size=keep, replace=False)
    filt = data[data[col].isin(chosen)]
    
    return filt.sample(n=min(n, len(filt)), random_state=seed)

def prop_stratified_sample(data, col='Age', n=2500, seed=1):
    f = n / len(data)
    
    samp = data.groupby(col, group_keys=False).apply(
        lambda x: x.sample(frac=f, random_state=seed)
    )
    
    if len(samp) != n:
        samp = samp.sample(n=n, random_state=seed, replace=True)
    return samp

def edge_boundary_sample(data, col='BMI', n=2500, seed=1):
    srt = data.sort_values(by=col)
    h = n // 2
    
    comb = pd.concat([srt.head(h), srt.tail(n - h)])
    return comb.sample(frac=1, random_state=seed)

def logic_condition_sample(data, n=2500, seed=1):
    sub = data[(data['HighBP'] == 1.0) & (data['HighChol'] == 1.0)]
    return sub.sample(n=min(n, len(sub)), random_state=seed)

def skewed_weight_sample(data, col='HeartDiseaseorAttack', n=2500, seed=1):
    freqs = data[col].value_counts(normalize=True)
    wmap = data[col].map(lambda x: 1.0 / freqs[x])
    
    return data.sample(n=n, weights=wmap, random_state=seed)
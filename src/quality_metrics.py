import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

def get_adv_stats(real, fake, col='BMI'):
    res = {}
    
    # basic stats
    res['real_mean'] = real[col].mean()
    res['real_var'] = real[col].var()
    res['synth_mean'] = fake[col].mean()
    res['synth_var'] = fake[col].var()
    
    # wasserstein limit
    lim = min(5000, len(real), len(fake))
    r_sub = real[col].sample(n=lim, random_state=42)
    f_sub = fake[col].sample(n=lim, random_state=42)
    
    res['wasserstein'] = wasserstein_distance(r_sub, f_sub)
    
    # corr differences
    cols = ['Diabetes_binary', 'HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'Age']
    valid = [c for c in cols if c in real.columns]
    
    rc = real[valid].corr()
    fc = fake[valid].corr()
    res['corr_residuals'] = (rc - fc).abs()
    
    # discriminator setup
    r_lab = real.sample(n=lim, random_state=42).copy()
    r_lab['is_real'] = 1
    
    f_lab = fake.sample(n=lim, random_state=42).copy()
    f_lab['is_real'] = 0
    
    combo = pd.concat([r_lab, f_lab])
    x = combo.drop('is_real', axis=1)
    y = combo['is_real']
    
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(xtr, ytr)
    
    preds = model.predict_proba(xte)[:, 1]
    res['discriminator_auc'] = roc_auc_score(yte, preds)
    res['roc_data'] = roc_curve(yte, preds)
    
    return res, r_sub, f_sub


def make_dash(res, r_dist, f_dist, col, path=None):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(top=0.85)
    fig.suptitle("Quality Check", fontsize=16)

    # density
    sns.kdeplot(r_dist, ax=ax[0], color='black', label='Real')
    sns.kdeplot(f_dist, ax=ax[0], color='red', label='Fake', linestyle='--')
    ax[0].set_title(f"Density: {col}")
    ax[0].legend()
    
    stats = (
        f"W-Dist: {res['wasserstein']:.3f}\n"
        f"Real (m, v): {res['real_mean']:.1f}, {res['real_var']:.1f}\n"
        f"Fake (m, v): {res['synth_mean']:.1f}, {res['synth_var']:.1f}"
    )
    ax[0].text(0.05, 0.95, stats, transform=ax[0].transAxes, va='top', bbox=dict(fc='white', alpha=0.8))

    # heatmap
    sns.heatmap(res['corr_residuals'], ax=ax[1], annot=True, fmt=".2f", cmap='Reds', cbar=False, vmin=0, vmax=0.25)
    ax[1].set_title("Corr Residuals")

    # roc
    fpr, tpr, _ = res['roc_data']
    auc = res['discriminator_auc']
    
    ax[2].plot(fpr, tpr, color='blue', label=f'Model (AUC = {auc:.2f})')
    ax[2].plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax[2].set(title="Discriminator ROC", xlabel="FPR", ylabel="TPR")
    ax[2].legend(loc='lower right')

    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=300)
    plt.close()
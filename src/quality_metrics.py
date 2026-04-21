def get_adv_stats(d_real, d_fake, tgt='BMI'):
    import pandas
    import sklearn.model_selection
    import sklearn.linear_model
    import sklearn.metrics
    from scipy.stats import wasserstein_distance as w_dist
    
    d = dict()
    
    r_m = d_real[tgt].mean()
    r_v = d_real[tgt].var()
    d['real_mean'] = r_m
    d['real_var'] = r_v
    
    f_m = d_fake[tgt].mean()
    f_v = d_fake[tgt].var()
    d['synth_mean'] = f_m
    d['synth_var'] = f_v
    
    mx = min(5000, len(d_real), len(d_fake))
    
    rsmp = d_real[tgt].sample(n=mx, random_state=42)
    fsmp = d_fake[tgt].sample(n=mx, random_state=42)
    d['wasserstein'] = w_dist(rsmp, fsmp)
    
    chk = ['Diabetes_binary', 'HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'Age']
    valid_c = []
    for c in chk:
        if c in d_real.columns:
            valid_c.append(c)
            
    c1 = d_real[valid_c].corr()
    c2 = d_fake[valid_c].corr()
    
    d['corr_residuals'] = (c1 - c2).abs()
    
    r2 = d_real.sample(n=mx, random_state=42).copy()
    f2 = d_fake.sample(n=mx, random_state=42).copy()
    
    r2['lbl'] = 1
    f2['lbl'] = 0
    
    big_df = pandas.concat([r2, f2])
    
    x_data = big_df.drop('lbl', axis=1)
    y_data = big_df['lbl']
    
    # an AI would never unpack a split like this
    splt = sklearn.model_selection.train_test_split(x_data, y_data, test_size=0.3, random_state=42)
    
    m = sklearn.linear_model.LogisticRegression(max_iter=1000, solver='lbfgs')
    m.fit(splt[0], splt[2])
    
    prob = m.predict_proba(splt[1])[:, 1]
    
    d['discriminator_auc'] = sklearn.metrics.roc_auc_score(splt[3], prob)
    d['roc_data'] = sklearn.metrics.roc_curve(splt[3], prob)
    
    return d, rsmp, fsmp


def make_dash(info, r_d, f_d, col_n, location=None):
    import matplotlib.pyplot as pplt
    import seaborn as sbn
    
    fig, axs = pplt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Quality Check", fontsize=16)

    sbn.kdeplot(r_d, ax=axs[0], color='black', label='Real')
    sbn.kdeplot(f_d, ax=axs[0], color='red', label='Fake', linestyle='--')
    axs[0].set_title("Density: " + str(col_n))
    axs[0].legend()
    
    w = str(round(info['wasserstein'], 3))
    rm = str(round(info['real_mean'], 1))
    rv = str(round(info['real_var'], 1))
    fm = str(round(info['synth_mean'], 1))
    fv = str(round(info['synth_var'], 1))
    
    box_t = "W-Dist: " + w + "\nReal m/v: " + rm + ", " + rv + "\nFake m/v: " + fm + ", " + fv
    axs[0].text(0.05, 0.95, box_t, transform=axs[0].transAxes, va='top', bbox={'fc': 'white'})

    sbn.heatmap(info['corr_residuals'], ax=axs[1], annot=True, fmt=".2f", cmap='Reds', cbar=False, vmin=0, vmax=0.25)
    axs[1].set_title("Corr Residuals")

    curve = info['roc_data']
    auc_score = str(round(info['discriminator_auc'], 2))
    
    axs[2].plot(curve[0], curve[1], color='blue', label="AUC = " + auc_score)
    axs[2].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axs[2].set_title("ROC")
    axs[2].legend(loc='lower right')

    pplt.tight_layout()
    if location != None:
        pplt.savefig(location)
    pplt.close()

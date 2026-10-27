"""MLflow experiment tracking + data/prediction drift analysis
Domain: Healthcare - diabetes disease progression prediction
Author: Mayur Khamamkar (DS 03)
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow, mlflow.sklearn
from scipy.stats import ks_2samp
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

SEED = 42
os.makedirs("reports", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Diabetes_Progression_Drift_Analysis")

# ---------- 1. data ----------
raw = load_diabetes(as_frame=True, scaled=False)
df = raw.frame.rename(columns={"s1": "tc", "s2": "ldl", "s3": "hdl", "s4": "tch", "s5": "ltg", "s6": "glu"})
features = [c for c in df.columns if c != "target"]

# Reference = younger cohort (age <= median), Current = older cohort (age > median)
med_age = df["age"].median()
ref = df[df["age"] <= med_age].reset_index(drop=True)
cur = df[df["age"] > med_age].reset_index(drop=True)

# ---------- 2. PSI helper ----------
def psi(expected, actual, bins=10):
    if len(np.unique(expected)) <= 5:  # categorical / binary column
        cats = np.union1d(np.unique(expected), np.unique(actual))
        e = np.array([(expected == c).mean() for c in cats]); a = np.array([(actual == c).mean() for c in cats])
        e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
        return float(np.sum((a - e) * np.log(a / e)))
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

with mlflow.start_run(run_name="rf_reference_vs_older_cohort") as run:
    params = dict(n_estimators=200, max_depth=6, min_samples_leaf=3, random_state=SEED)
    mlflow.log_params(params)
    mlflow.log_params({"dataset": "sklearn load_diabetes", "n_records": len(df),
                       "reference_rows": len(ref), "current_rows": len(cur),
                       "split_rule": f"age <= {med_age} vs age > {med_age}"})

    # ---------- 3. train on reference ----------
    Xtr, Xte, ytr, yte = train_test_split(ref[features], ref["target"], test_size=0.25, random_state=SEED)
    cv = cross_val_score(RandomForestRegressor(**params), ref[features], ref["target"],
                         cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2")
    model = RandomForestRegressor(**params).fit(ref[features], ref["target"])
    p_cur = model.predict(cur[features])
    metrics = {
        "ref_cv5_r2_mean": float(cv.mean()), "ref_cv5_r2_std": float(cv.std()),
        "current_r2": r2_score(cur["target"], p_cur),
        "current_rmse": root_mean_squared_error(cur["target"], p_cur),
        "current_mae": mean_absolute_error(cur["target"], p_cur),
    }
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, name="rf_model", input_example=ref[features].head(3),
                         skops_trusted_types=["sklearn.tree._tree.Tree"])

    # ---------- 4. drift tests ----------
    from sklearn.model_selection import cross_val_predict
    p_ref_all = cross_val_predict(RandomForestRegressor(**params), ref[features], ref["target"],
                                  cv=KFold(5, shuffle=True, random_state=SEED))  # out-of-fold, unbiased
    rows = []
    for f in features + ["target", "prediction"]:
        a = p_ref_all if f == "prediction" else ref[f]
        b = p_cur if f == "prediction" else cur[f]
        ks = ks_2samp(a, b)
        v = psi(np.asarray(a, float), np.asarray(b, float))
        rows.append(dict(column=f, ref_mean=float(np.mean(a)), cur_mean=float(np.mean(b)),
                         ks_stat=ks.statistic, ks_pvalue=ks.pvalue, psi=v,
                         drift=bool(ks.pvalue < 0.05 or v > 0.2)))
    rep = pd.DataFrame(rows)
    rep.to_csv("reports/drift_report.csv", index=False)
    monitored = [f for f in features if f != "age"]  # age is the cohort split key, excluded from verdict
    feat_rep = rep[rep.column.isin(monitored)]
    n_drift = int(feat_rep.drift.sum())
    share = n_drift / len(monitored)
    mlflow.log_metrics({"ref_oof_rmse": root_mean_squared_error(ref["target"], p_ref_all),
                        "ref_oof_mae": mean_absolute_error(ref["target"], p_ref_all)})
    metrics["ref_oof_rmse"] = root_mean_squared_error(ref["target"], p_ref_all)
    metrics["ref_oof_mae"] = mean_absolute_error(ref["target"], p_ref_all)
    mlflow.log_metrics({"n_drifted_features": n_drift, "share_drifted_features": share,
                        "prediction_psi": float(rep.loc[rep.column == "prediction", "psi"].iloc[0]),
                        "prediction_ks_pvalue": float(rep.loc[rep.column == "prediction", "ks_pvalue"].iloc[0])})
    for r in rows:
        mlflow.log_metric(f"psi_{r['column']}", r["psi"])
        mlflow.log_metric(f"ks_p_{r['column']}", r["ks_pvalue"])

    # ---------- 5. plots ----------
    cols = features + ["prediction"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, f in zip(axes.ravel(), cols):
        a = p_ref_all if f == "prediction" else ref[f]
        b = p_cur if f == "prediction" else cur[f]
        ax.hist(a, bins=20, alpha=.55, density=True, label="reference (younger)")
        ax.hist(b, bins=20, alpha=.55, density=True, label="current (older)")
        r = rep[rep.column == f].iloc[0]
        ax.set_title(f"{f}  PSI={r.psi:.2f}  p={r.ks_pvalue:.3f}" + ("  DRIFT" if r.drift else ""), fontsize=9)
    axes.ravel()[0].legend(fontsize=7); axes.ravel()[0].set_title("age (cohort split key, not monitored)", fontsize=9)
    axes.ravel()[-1].axis("off")
    plt.tight_layout(); plt.savefig("reports/feature_distributions.png", dpi=110); plt.close()

    plt.figure(figsize=(8, 4))
    s = feat_rep.sort_values("psi")
    plt.barh(s.column, s.psi, color=["#c0392b" if d else "#2e86c1" for d in s.drift])
    plt.axvline(0.1, ls="--", c="grey"); plt.axvline(0.2, ls="--", c="red")
    plt.xlabel("PSI"); plt.title("Population Stability Index per feature")
    plt.tight_layout(); plt.savefig("reports/psi_bar.png", dpi=110); plt.close()

    # control: two random halves of reference should show NO drift (sanity check of the method)
    h1 = ref.sample(frac=.5, random_state=SEED); h2 = ref.drop(h1.index)
    ctrl = sum(ks_2samp(h1[f], h2[f]).pvalue < 0.05 or psi(h1[f].values.astype(float), h2[f].values.astype(float)) > 0.2 for f in monitored)
    mlflow.log_metric("control_n_drifted_features", int(ctrl))
    summary = dict(control_n_drifted=int(ctrl), run_id=run.info.run_id, experiment_id=run.info.experiment_id,
                   metrics=metrics, n_drifted_features=n_drift, share_drifted=share,
                   drifted=feat_rep[feat_rep.drift].column.tolist())
    json.dump(summary, open("reports/summary.json", "w"), indent=2, default=float)
    mlflow.log_artifacts("reports", artifact_path="drift_reports")

print(rep.round(4).to_string(index=False))
print(json.dumps(summary, indent=2, default=float))

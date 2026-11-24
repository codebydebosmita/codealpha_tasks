"""
Credit Scoring Model
--------------------
Predicts the probability that a borrower will seriously default (90+ days
late within 2 years), compares Logistic Regression / Decision Tree / Random
Forest, and converts the result to a 300-850 credit score.

Install:  pip install pandas numpy scikit-learn matplotlib joblib
Demo:     python credit_scoring.py train                      (synthetic data)
Real:     python credit_scoring.py train --data cs-training.csv
Predict:  python credit_scoring.py predict sample_applicants.csv

Real dataset: Kaggle "Give Me Some Credit" (cs-training.csv, target SeriousDlqin2yrs).
"""
import argparse
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

TARGET = "SeriousDlqin2yrs"
UTIL = "RevolvingUtilizationOfUnsecuredLines"
L30, L60, L90 = ("NumberOfTime30-59DaysPastDueNotWorse",
                 "NumberOfTime60-89DaysPastDueNotWorse",
                 "NumberOfTimes90DaysLate")
RAW_COLS = [UTIL, "age", L30, "DebtRatio", "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans", L90,
            "NumberRealEstateLoansOrLines", L60, "NumberOfDependents"]
OUT_DIR = "credit_output"
MODEL_PATH = "credit_model.joblib"


# ------------------------------------------------------------- Data ----
def make_synthetic(n=20000, seed=42):
    """Synthetic borrowers with the same columns as 'Give Me Some Credit'."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)                                   # hidden financial "discipline"
    age = np.clip(rng.normal(50, 14, n), 21, 90).round()
    income = np.exp(rng.normal(8.6, 0.55, n)).round()
    util = np.clip(rng.beta(1.2, 3.0, n) + 0.12 * z, 0, 1.3)
    debt_ratio = np.clip(rng.lognormal(-1.2, 0.8, n), 0, 5)
    late30 = rng.poisson(np.exp(-2.0 + 0.8 * z + 0.8 * util))
    late60 = rng.poisson(np.exp(-2.8 + 0.8 * z + 0.6 * util))
    late90 = rng.poisson(np.exp(-3.3 + 0.9 * z + 0.6 * util))
    open_lines = rng.poisson(8, n)
    real_estate = rng.poisson(1.0, n)
    deps = rng.poisson(0.8, n).astype(float)
    logit = (-3.9 + 0.5 * z + 1.8 * util + 0.35 * np.minimum(debt_ratio, 2)
             - 0.022 * (age - 50) - 0.3 * np.log(income / 5400)
             + 0.8 * np.minimum(late90, 4) + 0.4 * np.minimum(late30 + late60, 5)
             + 0.05 * deps)
    default = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({TARGET: default, UTIL: util, "age": age, L30: late30,
                       "DebtRatio": debt_ratio, "MonthlyIncome": income.astype(float),
                       "NumberOfOpenCreditLinesAndLoans": open_lines, L90: late90,
                       "NumberRealEstateLoansOrLines": real_estate, L60: late60,
                       "NumberOfDependents": deps})
    df.loc[rng.random(n) < 0.18, "MonthlyIncome"] = np.nan   # missing income, like real data
    df.loc[rng.random(n) < 0.03, "NumberOfDependents"] = np.nan
    return df


def load_data(path):
    if path is None:
        print("No --data given: using synthetic demo data.")
        return make_synthetic()
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    missing = [c for c in [TARGET] + RAW_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing columns: {missing}")
    return df[[TARGET] + RAW_COLS]


# ---------------------------------------------------- Feature engineering ----
def engineer(df):
    """Raw borrower columns -> model features. Same code is used at predict time."""
    d = df.copy()
    for col in RAW_COLS:                                     # make sure every column exists
        if col not in d.columns:
            d[col] = np.nan
    for c in (L30, L60, L90):                                # 96/98 are data-entry codes
        d.loc[d[c] >= 90, c] = np.nan
    d[UTIL] = d[UTIL].clip(upper=1.5)
    d["DebtRatio"] = d["DebtRatio"].clip(upper=5)
    d["MonthlyIncome"] = d["MonthlyIncome"].clip(upper=50000)

    d["income_missing"] = d["MonthlyIncome"].isna().astype(int)   # missingness is informative
    d["TotalLate"] = d[[L30, L60, L90]].sum(axis=1, min_count=1)
    d["AnyLate"] = (d["TotalLate"] > 0).astype(int)
    d["SevereLate"] = (d[L90] > 0).astype(int)
    d["LogIncome"] = np.log1p(d["MonthlyIncome"])
    d["MonthlyDebtPayment"] = d["DebtRatio"] * d["MonthlyIncome"]
    d["IncomePerDependent"] = d["MonthlyIncome"] / (1 + d["NumberOfDependents"])
    d["HighUtilization"] = (d[UTIL] > 0.8).astype(int)
    d["Young"] = (d["age"] < 30).astype(int)
    return d.drop(columns=[TARGET], errors="ignore")


# ------------------------------------------------------------ Models ----
def build_models():
    impute = lambda: SimpleImputer(strategy="median")
    return {
        "Logistic Regression": Pipeline([("imp", impute()), ("sc", StandardScaler()),
                                         ("m", LogisticRegression(max_iter=1000))]),
        "Decision Tree": Pipeline([("imp", impute()),
                                   ("m", DecisionTreeClassifier(max_depth=5, min_samples_leaf=100,
                                                                random_state=42))]),
        "Random Forest": Pipeline([("imp", impute()),
                                   ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                                n_jobs=-1, random_state=42))]),
    }


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[np.argmax(f1)])


def to_score(p_default, base=600, pdo=20, base_odds=50):
    """Scorecard scaling: 600 points = 50:1 good:bad odds, +20 points doubles the odds."""
    p = np.clip(np.asarray(p_default), 1e-6, 1 - 1e-6)
    factor = pdo / np.log(2)
    offset = base - factor * np.log(base_odds)
    return np.clip(offset + factor * np.log((1 - p) / p), 300, 850).round().astype(int)


# ----------------------------------------------------------- Training ----
def train(data_path):
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data(data_path)
    print(f"Rows: {len(df)} | default rate: {df[TARGET].mean():.2%}")

    X, y = engineer(df), df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    results, fitted, thresholds, oof_auc = [], {}, {}, {}
    for name, model in build_models().items():
        print(f"Cross-validating {name}...")
        oof = cross_val_predict(model, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        oof_auc[name] = roc_auc_score(ytr, oof)
        thresholds[name] = best_f1_threshold(ytr, oof)        # tuned on train only (no leakage)
        model.fit(Xtr, ytr)
        fitted[name] = model

        p = model.predict_proba(Xte)[:, 1]
        pred = (p >= thresholds[name]).astype(int)
        results.append({"Model": name, "CV ROC-AUC": oof_auc[name],
                        "Test ROC-AUC": roc_auc_score(yte, p),
                        "PR-AUC": average_precision_score(yte, p),
                        "Precision": precision_score(yte, pred),
                        "Recall": recall_score(yte, pred),
                        "F1": f1_score(yte, pred),
                        "Threshold": thresholds[name]})
    table = pd.DataFrame(results).set_index("Model")
    print("\n=== Test-set results (threshold tuned for F1 on training folds) ===")
    print(table.round(3).to_string())
    table.round(4).to_csv(os.path.join(OUT_DIR, "model_comparison.csv"))

    best = max(oof_auc, key=oof_auc.get)                      # choose by CV, not by test
    print(f"\nBest model by cross-validated ROC-AUC: {best}")
    p = fitted[best].predict_proba(Xte)[:, 1]
    pred = (p >= thresholds[best]).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    print(f"Confusion matrix @ threshold {thresholds[best]:.3f}: "
          f"TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Caught {tp}/{tp + fn} defaulters; wrongly flagged {fp}/{tn + fp} good borrowers.")

    # --- plots ---
    plt.figure(figsize=(6, 5))
    for name, model in fitted.items():
        fpr, tpr, _ = roc_curve(yte, model.predict_proba(Xte)[:, 1])
        plt.plot(fpr, tpr, label=f"{name} (AUC {table.loc[name, 'Test ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curves.png"), dpi=130); plt.close()

    rf = fitted["Random Forest"]
    imp = pd.Series(rf.named_steps["m"].feature_importances_, index=X.columns).nlargest(12)[::-1]
    plt.figure(figsize=(7, 5)); imp.plot.barh()
    plt.title("Random Forest - top feature importances"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=130); plt.close()

    # --- save model + a few raw applicants to try predict on ---
    joblib.dump({"model": fitted[best], "name": best, "threshold": thresholds[best]}, MODEL_PATH)
    df.loc[Xte.index[:8], RAW_COLS].to_csv("sample_applicants.csv", index=False)
    print(f"\nSaved {MODEL_PATH}, sample_applicants.csv and plots in ./{OUT_DIR}/")


# --------------------------------------------------------- Prediction ----
def predict(csv_path):
    bundle = joblib.load(MODEL_PATH)
    raw = pd.read_csv(csv_path)
    p = bundle["model"].predict_proba(engineer(raw))[:, 1]
    out = raw.copy()
    out["default_probability"] = p.round(4)
    out["credit_score"] = to_score(p)
    out["decision"] = np.where(p >= bundle["threshold"], "HIGH RISK - review/decline", "Approve")
    out.to_csv("predictions.csv", index=False)
    print(f"Model: {bundle['name']} | risk threshold: {bundle['threshold']:.3f}\n")
    print(out[["default_probability", "credit_score", "decision"]].to_string())
    print("\nSaved full results to predictions.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--data", default=None, help="path to cs-training.csv")
    p = sub.add_parser("predict"); p.add_argument("csv")
    a = ap.parse_args()
    train(a.data) if a.cmd == "train" else predict(a.csv)

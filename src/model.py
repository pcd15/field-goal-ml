import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer


def load_model_data(path: str = "../data/fg_data.csv"):
    print(f"[load_model_data] Loading data from {path} ...")
    df = pd.read_csv(path)

    y = df["fg_made"]

    numeric_features = [
        "kick_distance",
        "half_seconds_remaining",
        "score_differential",
        "temp_f",
        "humidity_pct",
        "wind_speed_mph",
    ]
    categorical_features = [
        "roof",
        "surface",
        "wind_dir",
        "weather_type",
    ]

    X = df[numeric_features + categorical_features]
    print(f"[load_model_data] Target distribution: made={y.sum():,}, missed/blocked={(len(y) - y.sum()):,}")
    return X, y, numeric_features, categorical_features


def build_logreg_preprocessor(numeric_features, categorical_features):
    logreg_numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    logreg_categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    logreg_preprocessor = ColumnTransformer(
        transformers=[
            ("num", logreg_numeric_transformer, numeric_features),
            ("cat", logreg_categorical_transformer, categorical_features),
        ]
    )
    return logreg_preprocessor


def build_tree_preprocessor(numeric_features, categorical_features):
    tree_numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    tree_categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", tree_numeric_transformer, numeric_features),
            ("cat", tree_categorical_transformer, categorical_features),
        ]
    )
    return tree_preprocessor


def make_logreg_pipeline(logreg_preprocessor):
    return Pipeline(
        steps=[
            ("preprocess", logreg_preprocessor),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )


def make_gb_default_pipeline(tree_preprocessor):
    return Pipeline(
        steps=[
            ("preprocess", tree_preprocessor),
            ("model", HistGradientBoostingClassifier()),
        ]
    )


def make_gb_tuned_pipeline(tree_preprocessor):
    return Pipeline(
        steps=[
            ("preprocess", tree_preprocessor),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=31,
                min_samples_leaf=50,
                l2_regularization=0.0,
            )),
        ]
    )


def run_once(X, y, logreg_preprocessor, tree_preprocessor, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    logreg_clf = make_logreg_pipeline(logreg_preprocessor)
    logreg_clf.fit(X_train, y_train)
    logreg_preds = logreg_clf.predict_proba(X_test)[:, 1]
    logreg_auc = roc_auc_score(y_test, logreg_preds)

    gb_default_clf = make_gb_default_pipeline(tree_preprocessor)
    gb_default_clf.fit(X_train, y_train)
    gb_default_preds = gb_default_clf.predict_proba(X_test)[:, 1]
    gb_default_auc = roc_auc_score(y_test, gb_default_preds)

    gb_tuned_clf = make_gb_tuned_pipeline(tree_preprocessor)
    gb_tuned_clf.fit(X_train, y_train)
    gb_tuned_preds = gb_tuned_clf.predict_proba(X_test)[:, 1]
    gb_tuned_auc = roc_auc_score(y_test, gb_tuned_preds)

    return (
        logreg_auc,
        gb_default_auc,
        gb_tuned_auc,
        logreg_clf,
        gb_default_clf,
        gb_tuned_clf,
        X_test,
        y_test,
        logreg_preds,
        gb_default_preds,
        gb_tuned_preds,
    )


def compute_logreg_auc(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features,
    categorical_features,
    random_state: int = 42,
) -> float:
    features = numeric_features + categorical_features
    X_sub = X[features]
    preprocessor = build_logreg_preprocessor(numeric_features, categorical_features)
    clf = make_logreg_pipeline(preprocessor)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    return auc


def run_logreg_ablation_single(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features,
    categorical_features,
    random_state: int = 42,
) -> pd.DataFrame:
    print("\n[run_logreg_ablation_single] Running single-feature ablation study...")
    all_features = numeric_features + categorical_features

    base_auc = compute_logreg_auc(
        X, y, numeric_features, categorical_features, random_state=random_state
    )
    print(f"[run_logreg_ablation_single] Baseline AUC with all features: {base_auc:.4f}")

    rows = []
    for i, feat in enumerate(all_features, start=1):
        if feat in numeric_features:
            num_reduced = [f for f in numeric_features if f != feat]
            cat_reduced = categorical_features
        else:
            num_reduced = numeric_features
            cat_reduced = [f for f in categorical_features if f != feat]

        auc_reduced = compute_logreg_auc(
            X, y, num_reduced, cat_reduced, random_state=random_state
        )
        delta = base_auc - auc_reduced
        print(f"[run_logreg_ablation_single] AUC without '{feat}': {auc_reduced:.4f} (Δ={delta:.4f})")

        rows.append(
            {
                "feature_removed": feat,
                "auc_full": base_auc,
                "auc_without_feature": auc_reduced,
                "delta_auc": delta,
            }
        )

    ablation_df = pd.DataFrame(rows).sort_values("delta_auc", ascending=False)
    csv_path = "../data/ablation_single_logreg.csv"
    plot_path = "../plots/ablation_single_logreg.png"
    ablation_df.to_csv(csv_path, index=False)
    print(f"[run_logreg_ablation_single] Saved single-feature ablation results to {csv_path}")

    plt.figure()
    plt.bar(ablation_df["feature_removed"], ablation_df["delta_auc"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Change in AUC")
    plt.title("LogReg Single-Feature Ablation")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[run_logreg_ablation_single] Saved ablation bar plot to {plot_path}")
    return ablation_df


def run_logreg_ablation_groups(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features,
    categorical_features,
    random_state: int = 42,
) -> pd.DataFrame:
    print("\n[run_logreg_ablation_groups] Running grouped-feature ablation study...")
    all_features = numeric_features + categorical_features

    base_auc = compute_logreg_auc(
        X, y, numeric_features, categorical_features, random_state=random_state
    )
    print(f"[run_logreg_ablation_groups] Baseline AUC with all features: {base_auc:.4f}")

    groups = {
        "no_distance": ["kick_distance"],
        "no_weather": ["wind_speed_mph", "wind_dir", "humidity_pct", "weather_type", "temp_f"],
        "no_stadium": ["roof", "surface"],
        "no_pressure": ["half_seconds_remaining", "score_differential"],
        "distance_only": [f for f in all_features if f != "kick_distance"],
    }

    rows = []
    for i, (group_name, removed_feats) in enumerate(groups.items(), start=1):
        remaining = [f for f in all_features if f not in removed_feats]
        num_reduced = [f for f in remaining if f in numeric_features]
        cat_reduced = [f for f in remaining if f in categorical_features]

        if len(num_reduced) + len(cat_reduced) == 0:
            print(f"[run_logreg_ablation_groups] Skipping group '{group_name}' (no remaining features).")
            continue

        auc_reduced = compute_logreg_auc(
            X, y, num_reduced, cat_reduced, random_state=random_state
        )
        delta = base_auc - auc_reduced
        print(f"[run_logreg_ablation_groups] AUC for group '{group_name}': {auc_reduced:.4f} (Δ={delta:.4f})")

        rows.append(
            {
                "group_removed": group_name,
                "features_removed": ", ".join(removed_feats),
                "auc_full": base_auc,
                "auc_after_removal": auc_reduced,
                "delta_auc": delta,
            }
        )

    group_df = pd.DataFrame(rows).sort_values("delta_auc", ascending=False)
    csv_path = "../data/ablation_groups_logreg.csv"
    plot_path = "../plots/ablation_groups_logreg.png"
    group_df.to_csv(csv_path, index=False)
    print(f"[run_logreg_ablation_groups] Saved group ablation results to {csv_path}")

    plt.figure()
    plt.bar(group_df["group_removed"], group_df["delta_auc"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Change in AUC")
    plt.title("LogReg Group Ablation")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[run_logreg_ablation_groups] Saved ablation bar plot to {plot_path}")
    return group_df


def compute_loglikelihood(y_true, p_pred, eps: float = 1e-15) -> float:
    p = np.clip(p_pred, eps, 1 - eps)
    y = np.asarray(y_true, dtype=float)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def run_logreg_info_criteria(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features,
    categorical_features,
    random_state: int = 42,
) -> pd.DataFrame:
    print("\n[run_logreg_info_criteria] Computing AIC/BIC for full vs distance-only logistic regression...")

    def fit_and_summarize(num_feats, cat_feats, model_name: str) -> dict:
        feature_list = num_feats + cat_feats
        X_sub = X[feature_list]
        preprocessor = build_logreg_preprocessor(num_feats, cat_feats)
        clf = make_logreg_pipeline(preprocessor)

        X_train, X_test, y_train, y_test = train_test_split(
            X_sub,
            y,
            test_size=0.2,
            random_state=random_state,
            stratify=y,
        )

        clf.fit(X_train, y_train)
        p_test = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, p_test)
        logL_full = compute_loglikelihood(y_test, p_test)

        X_train_trans = clf.named_steps["preprocess"].transform(X_train)
        n_features_eff = X_train_trans.shape[1]
        k = n_features_eff + 1
        n = len(y_test)

        aic = 2 * k - 2 * logL_full
        bic = k * np.log(n) - 2 * logL_full

        print(
            f"[run_logreg_info_criteria] '{model_name}': "
            f"AUC={auc:.4f}, effective_params={k}, AIC={aic:.2f}, BIC={bic:.2f}"
        )

        return {
            "model": model_name,
            "num_raw_features": len(feature_list),
            "effective_params": int(k),
            "AUC": auc,
            "AIC": aic,
            "BIC": bic,
        }

    results = []
    results.append(
        fit_and_summarize(
            numeric_features,
            categorical_features,
            model_name="logreg_full",
        )
    )
    results.append(
        fit_and_summarize(
            ["kick_distance"],
            [],
            model_name="logreg_distance_only",
        )
    )
    info_df = pd.DataFrame(results)
    csv_path = "../data/logreg_info_criteria.csv"
    info_df.to_csv(csv_path, index=False)
    print(f"[run_logreg_info_criteria] Saved info-criteria results to {csv_path}")
    return info_df

def main():
    print("[main] Starting supervised learning pipeline...\n")

    X, y, numeric_features, categorical_features = load_model_data(
        "../data/fg_data.csv"
    )

    logreg_preprocessor = build_logreg_preprocessor(
        numeric_features, categorical_features
    )
    tree_preprocessor = build_tree_preprocessor(
        numeric_features, categorical_features
    )

    N_RUNS = 10
    rng_seeds = range(42, 42 + N_RUNS)
    logreg_aucs = []
    gb_default_aucs = []
    gb_tuned_aucs = []

    print(f"\n[main] Running {N_RUNS} repeated train/test splits for model comparison...")
    for i, seed in enumerate(rng_seeds, start=1):
        print(f"\n[main] === Run {i}/{N_RUNS} (random_state={seed}) ===")
        log_auc, gb_def_auc, gb_t_auc, *_ = run_once(
            X, y, logreg_preprocessor, tree_preprocessor, random_state=seed
        )
        logreg_aucs.append(log_auc)
        gb_default_aucs.append(gb_def_auc)
        gb_tuned_aucs.append(gb_t_auc)

    mean_logreg_auc = np.mean(logreg_aucs)
    mean_gb_default_auc = np.mean(gb_default_aucs)
    mean_gb_tuned_auc = np.mean(gb_tuned_aucs)

    std_logreg_auc = np.std(logreg_aucs)
    std_gb_default_auc = np.std(gb_default_aucs)
    std_gb_tuned_auc = np.std(gb_tuned_aucs)

    print("\n[main] Summary of AUC over multiple random splits (N_RUNS = 10):")
    print(f"  Logistic Regression      : {mean_logreg_auc:.4f} ± {std_logreg_auc:.4f}")
    print(f"  Default Gradient Boosted : {mean_gb_default_auc:.4f} ± {std_gb_default_auc:.4f}")
    print(f"  Tuned Gradient Boosted   : {mean_gb_tuned_auc:.4f} ± {std_gb_tuned_auc:.4f}")

    single_df = run_logreg_ablation_single(X, y, numeric_features, categorical_features)
    print("\n[main] Single-feature ablation (LogReg):")
    print(single_df)

    group_df = run_logreg_ablation_groups(X, y, numeric_features, categorical_features)
    print("\n[main] Group ablation (LogReg):")
    print(group_df)

    info_df = run_logreg_info_criteria(X, y, numeric_features, categorical_features)
    print("\n[main] Model statistics (test split):")
    print(info_df)

    print("\n[main] Supervised learning analysis complete.")


if __name__ == "__main__":
    main()
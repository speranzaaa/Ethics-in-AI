"""KDE evaluation layer"""

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from .config import KDE_MODEL_PATH, WINDOW_PATH


class EthicalKDEAnomalyDetector:
    def __init__(
        self,
        bandwidth = 1.0,
        kernel = "gaussian",
    ):
        self.bandwidth = bandwidth
        self.kernel = kernel

        self.scaler = RobustScaler()
        self.kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        self.score_scaler: MinMaxScaler | None = None
        self.best_bandwidth = bandwidth
        self.fitted = False

    def fit(
        self,
        X_train,
        cv = 5,
        bandwidth_cv = True,
    ):
        X_scaled = self.scaler.fit_transform(X_train)
        n_samples = len(X_scaled)

        if bandwidth_cv and n_samples >= 10:
            bandwidths = np.logspace(-1.3, 0.7, 30)
            n_folds = max(2, min(cv, n_samples))
            grid_cv = GridSearchCV(
                KernelDensity(kernel=self.kernel),
                {"bandwidth": bandwidths},
                cv=n_folds,
                n_jobs=-1,
            )
            grid_cv.fit(X_scaled)
            self.best_bandwidth = float(grid_cv.best_params_["bandwidth"])
            print(
                f"Bandwidth CV: best={self.best_bandwidth} over "
                f"{len(bandwidths)} candidates, {n_folds} folds."
            )
        else:
            self.best_bandwidth = self.bandwidth
            if n_samples < 10:
                print(
                    f"Only {n_samples} training samples - "
                    f"bandwidth CV skipped, using {self.bandwidth}"
                )

        self.kde = KernelDensity(bandwidth=self.best_bandwidth, kernel=self.kernel)
        self.kde.fit(X_scaled)

        raw_train_nll = -self.kde.score_samples(X_scaled)
        self.score_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self.score_scaler.fit(raw_train_nll.reshape(-1, 1))

        self.fitted = True

        print(
            f"KDE fitted: {X_train.shape[0]} samples, {X_train.shape[1]} features. "
            f"Bandwidth={self.best_bandwidth} "
        )
        print(f"Raw NLL [{float(raw_train_nll.min())}, {float(raw_train_nll.max())}]")

        return self

    # Score

    def get_anomaly_signal(self, X):
        if not self.fitted:
            raise RuntimeError("Call .fit() before scoring.")

        numeric_X = X.select_dtypes(include=[np.number])
        X_scaled = self.scaler.transform(numeric_X.values)
        raw_nll = -self.kde.score_samples(X_scaled)

        if self.score_scaler is not None:
            calibrated = self.score_scaler.transform(raw_nll.reshape(-1, 1)).ravel()
            signal = np.clip(calibrated, 0.0, None)
        else:
            signal = raw_nll

        return pd.Series(signal, index=X.index, name="anomaly_signal")

    def __repr__(self):
        bw_label = (
            f"{self.best_bandwidth:.4f} (CV-optimised)"
            if self.best_bandwidth != self.bandwidth
            else f"{self.bandwidth:.4f}"
        )
        return (
            f"EthicalKDEAnomalyDetector(bandwidth={bw_label}, kernel='{self.kernel}', fitted={self.fitted})"
        )


# Training and loading helpers

def train_kde(window_path=None, model_path=None):

    if window_path is None:
        window_path = WINDOW_PATH
    if model_path is None:
        model_path = KDE_MODEL_PATH

    windows_neg = pd.read_parquet(window_path, engine="pyarrow")

    normal_mask = windows_neg["num_visits_90d_t"] <= 2
    kde_train_cols = [
        c for c in windows_neg[normal_mask].select_dtypes(include=[np.number]).columns
    ]
    X_train_neg = windows_neg[normal_mask][kde_train_cols].values

    print(f"Training: {len(X_train_neg)} finestre | {len(kde_train_cols)} feature")

    detector = EthicalKDEAnomalyDetector(bandwidth=1.0, kernel="gaussian")
    detector.fit(X_train_neg, cv=5)
    print(f"  {detector}")

    joblib.dump(detector, model_path)
    print(f"Modello salvato in: {model_path}")

    return detector


def load_or_train_kde(model_path=None, window_path=None):

    if model_path is None:
        model_path = KDE_MODEL_PATH
    if window_path is None:
        window_path = WINDOW_PATH

    if model_path.exists():
        detector = joblib.load(model_path)
        print(f"{detector}")
    else:
        print(f"Modello KDE non trovato in {model_path}, avvio training...")
        detector = train_kde(window_path=window_path, model_path=model_path)

    kde_numeric_cols = (
        pd.read_parquet(window_path)
        .select_dtypes(include=[np.number])
        .columns.tolist()
    )

    return detector, kde_numeric_cols

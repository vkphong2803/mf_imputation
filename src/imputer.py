"""
Matrix Factorization — Missing Data Imputation
Core imputation logic using SGD-based MF.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────
# Column mapping: original → Vietnamese names
# ─────────────────────────────────────────────
COL_MAP = {
    "Age":                     "Độ_tuổi",
    "Gender":                  "Giới_tính",
    "BMI":                     "BMI",
    "Blood_Pressure_Systolic": "HuyetAp_TamThu",
    "Glucose_Level":           "DuongHuyet",
    "LDL":                     "Cholesterol",
    "Physical_Activity_Level": "ThoiGian_VanDong_Daily",
    "Stress_Level":            "ChiSo_Stress",
    "HbA1c":                   "HbA1c",
}

ORDINAL_MAP = {
    "Gender": {"Male": 1, "Female": 0},
    "Physical_Activity_Level": {
        "Sedentary": 1, "Lightly Active": 2,
        "Moderately Active": 3, "Highly Active": 4,
    },
}


# ─────────────────────────────────────────────
# 1. Load & prepare
# ─────────────────────────────────────────────
def load_and_prepare(input_file: str, n_rows: int = 100) -> pd.DataFrame:
    """Read Excel, extract 9 health columns, encode categoricals."""
    raw = pd.read_excel(input_file)

    missing_cols = [c for c in COL_MAP if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Các cột không tìm thấy trong file: {missing_cols}")

    df = raw[list(COL_MAP.keys())].copy()
    df = df.rename(columns=COL_MAP)

    for orig_col, mapping in ORDINAL_MAP.items():
        vi_col = COL_MAP[orig_col]
        df[vi_col] = df[vi_col].map(mapping)

    df = df.head(n_rows).reset_index(drop=True).astype(float)
    return df


# ─────────────────────────────────────────────
# 2. Create missing data
# ─────────────────────────────────────────────
def create_missing(df: pd.DataFrame, rate: float = 0.15,
                   seed: int = 42) -> pd.DataFrame:
    """Inject NaN into ~`rate` fraction of cells (MCAR pattern)."""
    rng  = np.random.default_rng(seed)
    out  = df.copy()
    flat = rng.choice(out.size, int(out.size * rate), replace=False)
    rows, cols = np.unravel_index(flat, out.shape)
    for r, c in zip(rows, cols):
        out.iat[r, c] = np.nan
    return out


# ─────────────────────────────────────────────
# 3. Matrix Factorization (SGD)
# ─────────────────────────────────────────────
def _minmax_scale(df: pd.DataFrame):
    mn, mx = df.min().values, df.max().values
    safe   = np.where(mx - mn == 0, 1, mx - mn)
    return (df.values - mn) / safe, mn, mx


def _inverse_scale(arr: np.ndarray, mn, mx) -> np.ndarray:
    return arr * (mx - mn) + mn


def _sgd_mf(R: np.ndarray, mask: np.ndarray,
            k: int, n_iter: int, lr: float, reg: float, seed: int):
    """
    SGD Matrix Factorization: R ≈ U @ V.T
    Chỉ cập nhật trên các ô quan sát (mask=True).
    """
    rng = np.random.default_rng(seed + 99)
    n, m = R.shape
    U = rng.random((n, k)) * 0.1
    V = rng.random((m, k)) * 0.1
    loss_log = []

    for it in range(n_iter):
        for i in range(n):
            obs = np.where(mask[i])[0]
            if len(obs) == 0:
                continue
            err    = R[i, obs] - U[i] @ V[obs].T
            grad_U = err @ V[obs] - reg * U[i]
            grad_V = np.outer(err, U[i]) - reg * V[obs]
            U[i]   += lr * grad_U
            V[obs] += lr * grad_V

        if it % 50 == 0:
            pred = U @ V.T
            loss = np.mean((R[mask] - pred[mask]) ** 2)
            loss_log.append((it, loss))

    return np.clip(U @ V.T, 0, 1), loss_log


def impute(full_df: pd.DataFrame, missing_df: pd.DataFrame,
           rank: int = 5, lr: float = 0.005, reg: float = 0.02,
           n_iters: int = 500, seed: int = 42):
    """Run MF imputation; return (imputed_df, loss_log)."""
    norm_full, mn, mx = _minmax_scale(full_df)
    norm_miss, _,  _  = _minmax_scale(missing_df)

    mask = ~np.isnan(norm_miss)
    R    = np.where(mask, norm_miss, 0.0)

    R_hat, loss_log = _sgd_mf(R, mask, k=rank, n_iter=n_iters,
                               lr=lr, reg=reg, seed=seed)

    R_filled = np.where(mask, norm_miss, R_hat)
    imputed  = pd.DataFrame(
        _inverse_scale(R_filled, mn, mx),
        columns=full_df.columns,
    )
    return imputed, loss_log


# ─────────────────────────────────────────────
# 4. Evaluate
# ─────────────────────────────────────────────
def evaluate(full_df: pd.DataFrame, missing_df: pd.DataFrame,
             imputed_df: pd.DataFrame):
    """Return (per-column RMSE DataFrame, overall normalized RMSE)."""
    miss_mask = missing_df.isna()
    rows = []
    for col in full_df.columns:
        idx = miss_mask[col]
        if not idx.any():
            continue
        y_true = full_df.loc[idx, col].values
        y_pred = imputed_df.loc[idx, col].values
        rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
        rows.append({"Cột": col, "n_thiếu": int(idx.sum()),
                     "RMSE": round(rmse, 4)})

    rmse_df = pd.DataFrame(rows).set_index("Cột")

    scaler   = MinMaxScaler().fit(full_df)
    f_scaled = scaler.transform(full_df)
    i_scaled = scaler.transform(imputed_df)
    flat     = missing_df.isna().values
    overall  = np.sqrt(mean_squared_error(f_scaled[flat], i_scaled[flat]))

    return rmse_df, overall

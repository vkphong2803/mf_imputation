import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_rmse(rmse_df: pd.DataFrame,
              overall_norm: float,
              out_path: str) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, len(rmse_df)))

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.barh(rmse_df.index, rmse_df["RMSE"],
                   color=colors, height=0.55, edgecolor="white")

    # Nhãn giá trị ở đầu mỗi cột
    for bar, v in zip(bars, rmse_df["RMSE"]):
        ax.text(bar.get_width() + max(rmse_df["RMSE"]) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9)

    ax.set_xlabel("RMSE (thang giá trị gốc)", fontsize=10)
    ax.set_title(
        "RMSE sau Matrix Factorization Imputation\n"
        f"(RMSE tổng thể chuẩn hóa 0–1: {overall_norm:.4f})",
        fontsize=11, fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Biểu đồ RMSE lưu tại: {out_path}")

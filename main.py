import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.imputer    import load_and_prepare, create_missing, impute, evaluate
from src.visualizer import plot_rmse


# ────────────────────────────────────────────────────────────────
# Default config 
# ────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "input_file":   ROOT / "data" / "personalised_dataset.xlsx",
    "n_rows":       100,
    "missing_rate": 0.15,
    "seed":         42,
    "mf_rank":      5,
    "mf_lr":        0.005,
    "mf_reg":       0.02,
    "mf_iters":     500,
    # Output paths
    "out_full":     ROOT / "output" / "full_data.csv",
    "out_missing":  ROOT / "output" / "missing_data.csv",
    "out_imputed":  ROOT / "output" / "imputed_data.csv",
    "out_rmse_csv": ROOT / "output" / "rmse_results.csv",
    "out_rmse_txt": ROOT / "output" / "rmse_results.txt",
    "out_plot":     ROOT / "output" / "rmse_chart.png",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Matrix Factorization Imputation — personalised_dataset"
    )
    parser.add_argument(
        "--input", type=Path,
        default=DEFAULT_CONFIG["input_file"],
        help="Đường dẫn tới file .xlsx (default: data/personalised_dataset.xlsx)",
    )
    parser.add_argument(
        "--rows", type=int, default=DEFAULT_CONFIG["n_rows"],
        help="Số dòng lấy từ dataset (default: 100)",
    )
    parser.add_argument(
        "--missing-rate", type=float, default=DEFAULT_CONFIG["missing_rate"],
        help="Tỷ lệ missing (default: 0.15)",
    )
    parser.add_argument(
        "--rank", type=int, default=DEFAULT_CONFIG["mf_rank"],
        help="Rank của ma trận (default: 5)",
    )
    parser.add_argument(
        "--iters", type=int, default=DEFAULT_CONFIG["mf_iters"],
        help="Số vòng lặp SGD (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_CONFIG["seed"],
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = {**DEFAULT_CONFIG}
    cfg["input_file"]   = args.input
    cfg["n_rows"]       = args.rows
    cfg["missing_rate"] = args.missing_rate
    cfg["mf_rank"]      = args.rank
    cfg["mf_iters"]     = args.iters
    cfg["seed"]         = args.seed

    # ── Kiểm tra file đầu vào ────────────────────────────────────
    if not Path(cfg["input_file"]).exists():
        print(f"[ERROR] Không tìm thấy file: {cfg['input_file']}")
        print("        → Đặt file .xlsx vào thư mục  data/  rồi chạy lại.")
        sys.exit(1)

    # ── Tạo thư mục output nếu chưa có ──────────────────────────
    Path(cfg["out_full"]).parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────
    full_data = load_and_prepare(str(cfg["input_file"]), cfg["n_rows"])
    full_data.to_csv(cfg["out_full"], index=False, encoding="utf-8-sig")
    print(f"[1] full_data    → {cfg['out_full']}  shape={full_data.shape}")

    # ── 2. Missing ───────────────────────────────────────────────
    missing_data = create_missing(full_data, cfg["missing_rate"], cfg["seed"])
    missing_data.to_csv(cfg["out_missing"], index=False, encoding="utf-8-sig")
    n_miss = int(missing_data.isna().sum().sum())
    print(
        f"[2] missing_data → {cfg['out_missing']}  "
        f"({n_miss}/{full_data.size} ô = {n_miss/full_data.size:.1%})"
    )

    # ── 3. Matrix Factorization ──────────────────────────────────
    print(
        f"[3] Matrix Factorization  "
        f"rank={cfg['mf_rank']}  iter={cfg['mf_iters']}  lr={cfg['mf_lr']} ..."
    )
    imputed_data, loss_log = impute(
        full_data, missing_data,
        rank=cfg["mf_rank"], lr=cfg["mf_lr"],
        reg=cfg["mf_reg"],   n_iters=cfg["mf_iters"],
        seed=cfg["seed"],
    )
    imputed_data.to_csv(cfg["out_imputed"], index=False, encoding="utf-8-sig")
    print(f"    → {cfg['out_imputed']}  |  loss cuối: {loss_log[-1][1]:.6f}")

    # ── 4. Evaluate ──────────────────────────────────────────────
    rmse_df, overall_norm = evaluate(full_data, missing_data, imputed_data)
    print("\n[4] RMSE theo cột (thang gốc):")
    print(rmse_df.to_string())
    print(f"\n    RMSE tổng thể (chuẩn hóa 0–1): {overall_norm:.4f}")

    # Lưu RMSE ra CSV
    rmse_df.to_csv(cfg["out_rmse_csv"], encoding="utf-8-sig")
    print(f"    → RMSE CSV: {cfg['out_rmse_csv']}")

    # Lưu RMSE ra TXT 
    with open(cfg["out_rmse_txt"], "w", encoding="utf-8") as f:
        f.write("=" * 48 + "\n")
        f.write("  Matrix Factorization — RMSE kết quả\n")
        f.write("=" * 48 + "\n\n")
        f.write(rmse_df.to_string())
        f.write(f"\n\nRMSE tổng thể (chuẩn hóa 0–1): {overall_norm:.4f}\n")
        f.write(f"\nCấu hình: rank={cfg['mf_rank']}  iters={cfg['mf_iters']}"
                f"  lr={cfg['mf_lr']}  missing_rate={cfg['missing_rate']}\n")
    print(f"    → RMSE TXT: {cfg['out_rmse_txt']}")

    # ── 5. Visualize ─────────────────────────────────────────────
    plot_rmse(rmse_df, overall_norm, str(cfg["out_plot"]))

    print("\n✓ Hoàn tất! Kết quả lưu trong thư mục output/")


if __name__ == "__main__":
    main()

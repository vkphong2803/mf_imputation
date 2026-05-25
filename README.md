# Matrix Factorization — Missing Data Imputation

PT2: điền giá trị thiếu bằng phương pháp **Matrix Factorization (SGD)** trên bộ dữ liệu `personalised_dataset.xlsx`.

---

## Cấu trúc thư mục

```
mf_imputation/
├── data/                       
│   └── personalised_dataset.xlsx
├── output/                      ← Kết quả sau khi chạy (CSV + PNG)
│   ├── full_data.csv
│   ├── missing_data.csv
│   ├── imputed_data.csv
│   └── mf_results.png
├── src/
│   ├── __init__.py
│   ├── imputer.py               ← Load dữ liệu, tạo missing, MF, evaluate
│   └── visualizer.py            ← Vẽ 6-panel figure
├── main.py                      ← Entry point chính
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Yêu cầu

- Python **3.10+**
- Git

---

## Cài đặt & chạy

### Bước 1 — Clone repo

```bash
git clone https://github.com/vkphong2803/mf_imputation.git
cd mf_imputation
```

### Bước 2 — Tạo môi trường ảo

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Bước 3 — Cài thư viện

```bash
pip install -r requirements.txt
```

### Bước 4 — Đặt file dữ liệu

Sao chép `personalised_dataset.xlsx` vào thư mục `data/`:

```
data/
└── personalised_dataset.xlsx
```

### Bước 5 — Chạy chương trình

```bash
python main.py
```

Hoặc tuỳ chỉnh tham số:

```bash
python main.py \
  --input data/personalised_dataset.xlsx \
  --rows 100 \
  --missing-rate 0.15 \
  --rank 5 \
  --iters 500 \
  --seed 42
```

---

## Tham số dòng lệnh

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--input` | `data/personalised_dataset.xlsx` | Đường dẫn file Excel |
| `--rows` | `100` | Số dòng lấy từ dataset |
| `--missing-rate` | `0.15` | Tỷ lệ thiếu dữ liệu (0–1) |
| `--rank` | `5` | Rank của phân rã ma trận |
| `--iters` | `500` | Số vòng lặp SGD |
| `--seed` | `42` | Random seed |

---

## Pipeline

```
[1] Load & chuẩn bị   →  9 cột y tế  |  100 dòng
[2] Tạo missing        →  ~15% NaN ngẫu nhiên (MCAR)
[3] Matrix Factorization (SGD)  →  R ≈ U @ Vᵀ
[4] Evaluate           →  RMSE từng cột + RMSE tổng (chuẩn hóa)
[5] Visualize          →  6-panel figure  →  output/mf_results.png
```

---

## 9 cột y tế được sử dụng

| Tên gốc | Tên tiếng Việt | Kiểu |
|---|---|---|
| Age | Độ_tuổi | Liên tục |
| Gender | Giới_tính | Nhị phân (0/1) |
| BMI | BMI | Liên tục |
| Blood_Pressure_Systolic | HuyetAp_TamThu | Liên tục |
| Glucose_Level | DuongHuyet | Liên tục |
| LDL | Cholesterol | Liên tục |
| Physical_Activity_Level | ThoiGian_VanDong_Daily | Thứ tự (1–4) |
| Stress_Level | ChiSo_Stress | Liên tục |
| HbA1c | HbA1c | Liên tục |

---

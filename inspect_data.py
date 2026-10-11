from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    print("\nNO CSV FILE FOUND.")
    print("Put your Kaggle .csv file inside the 'data' folder.")
    print("Example: Flight_Delay_Prediction/data/your_dataset.csv")
    raise SystemExit(1)

csv_path = max(csv_files, key=lambda p: p.stat().st_size)
print(f"\nCSV found: {csv_path}")
print(f"File size: {csv_path.stat().st_size / (1024**3):.2f} GB")
print("\nReading only 5 rows to check the columns (not the whole 2 GB file)...")

try:
    df = pd.read_csv(csv_path, nrows=5, low_memory=False)
    print("\nCOLUMNS:")
    for col in df.columns:
        print("-", col)
    print("\nFIRST 5 ROWS:")
    print(df.head().to_string())
    print("\nCheck complete. If the columns look correct, run train.py.")
except Exception as e:
    print("\nCould not read this CSV:", e)
    print("If the file uses a different separator or encoding, share this error.")

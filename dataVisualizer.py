import pandas as pd
from pathlib import Path

file_path = Path("nba_2024_25_gamelogs.parquet")

df = pd.read_parquet(file_path)

print(df.head(10).to_string())

print("\n--- Available Columns ---")
print(df.columns.tolist())
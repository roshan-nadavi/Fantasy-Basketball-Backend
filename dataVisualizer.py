import pandas as pd
from pathlib import Path

file_path = Path("nba_2024_25_gamelogs.parquet")

df = pd.read_parquet(file_path)

print(df.iloc[19000:19010].to_string())

print("\n--- Available Columns ---")
print(df.columns.tolist())

# 2. Get the data type for JUST the GAME_DATE column
game_date_dtype = df['GAME_DATE'].dtype
print(f"\nThe data type of GAME_DATE is: {game_date_dtype}")
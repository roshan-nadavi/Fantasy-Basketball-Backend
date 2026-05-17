import time
import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import leaguegamelog

def collect_nba_data():
    current_dir = Path(__file__).parent
    file_path = current_dir / "nba_2023_24_gamelogs.parquet"

    custom_headers = {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com',
    }

    print("Initiating connection to NBA servers for the 2025-26 season...")
    start_time = time.time()

    try:
        log = leaguegamelog.LeagueGameLog(
            season='2023-24', 
            player_or_team_abbreviation='P',
            headers=custom_headers
        )
        
        df = log.get_data_frames()[0]
        
        end_time = time.time()
        print(f"Success! Data retrieved in {end_time - start_time:.2f} seconds.")
        print(f"Total Rows: {len(df)}")

        df.to_parquet(file_path, index=False, engine='pyarrow')
        
        print(f"--- COMPLETE ---")
        print(f"File created at: {file_path}")

    except Exception as e:
        print(f"An error occurred during data collection: {e}")

if __name__ == "__main__":
    collect_nba_data()
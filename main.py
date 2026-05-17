from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

# Global dictionary to hold multiple seasons
season_data = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global season_data
    data_dir = Path(__file__).parent
    
    for file_path in data_dir.glob("*.parquet"):
        season_key = file_path.stem.split("_", 1)[1] 
        season_data[season_key] = pd.read_parquet(file_path)
        print(f"Loaded {season_key}: {len(season_data[season_key])} rows.")
        
    yield
    season_data.clear()
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

def apply_scoring(df_slice: pd.DataFrame, weights: dict) -> pd.Series:
    score = pd.Series(0.0, index=df_slice.index)
    for stat, weight in weights.items():
        if stat in df_slice.columns and weight != 0:
            score += df_slice[stat] * weight
    return score

class ScoringWeights(BaseModel):
    FGM: float = 0.0
    FGA: float = 0.0
    FG3M: float = 0.0
    FG3A: float = 0.0
    FTM: float = 0.0
    FTA: float = 0.0
    OREB: float = 0.0
    DREB: float = 0.0
    REB: float = 1.2
    AST: float = 1.5
    STL: float = 3.0
    BLK: float = 3.0
    TOV: float = -1.0
    PF: float = 0.0
    PTS: float = 1.0
@app.post("/{season}/players/fantasy-scores")
def get_fantasy_scores(season: str, weights: ScoringWeights):
    global season_data

    if season not in season_data:
        raise HTTPException(status_code=404, detail="Season data not found")
        
    df_logs = season_data[season].copy()
    
    w = weights.model_dump()
    
    custom_scores = pd.Series(0.0, index=df_logs.index)
    
    for stat, weight in w.items():
        if weight != 0:
            custom_scores += df_logs[stat] * weight
            
    df_logs['CUSTOM_FP'] = custom_scores
    
    grouped = df_logs.groupby(['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION']).agg(
        games_played=('GAME_ID', 'count'),
        total_fantasy_pts=('CUSTOM_FP', 'sum'),
        avg_fantasy_pts=('CUSTOM_FP', 'mean')
    ).reset_index()
    
    grouped = grouped.sort_values(by='total_fantasy_pts', ascending=False)
    grouped['total_fantasy_pts'] = grouped['total_fantasy_pts'].round(2)
    grouped['avg_fantasy_pts'] = grouped['avg_fantasy_pts'].round(2)
    
    return grouped.to_dict(orient="records")

# 1. New Request Model combining Dates and Weights
class DateRangeScoringRequest(BaseModel):
    start_date: str  # Expected format: "YYYY-MM-DD"
    end_date: str    # Expected format: "YYYY-MM-DD"
    weights: ScoringWeights

# 2. New Filtered Endpoint
@app.post("/{season}/players/fantasy-scores/date-range")
def get_fantasy_scores_by_date(season: str, req: DateRangeScoringRequest):
    global season_data

    if season not in season_data:
        raise HTTPException(status_code=404, detail="Season data not found")
        
    df_logs = season_data[season]
    
    # 3. Filter by Date Range First (Reduces rows to process)
    # Pandas handles ISO string comparisons ("YYYY-MM-DD") natively and quickly
    date_mask = (df_logs['GAME_DATE'] >= req.start_date) & (df_logs['GAME_DATE'] <= req.end_date)
    df_filtered = df_logs[date_mask].copy()
    
    if df_filtered.empty:
        return [] # Return empty list gracefully if no games happened in this range

    # 4. Compute Vectorized Math on the filtered subset
    w = req.weights.model_dump()
    custom_scores = pd.Series(0.0, index=df_filtered.index)
    
    for stat, weight in w.items():
        if weight != 0 and stat in df_filtered.columns:
            custom_scores += df_filtered[stat] * weight
            
    # Assign to the local dataframe copy to protect global state integrity
    df_filtered['CUSTOM_FP'] = custom_scores
    
    # 5. Group and Aggregate
    grouped = df_filtered.groupby(['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION']).agg(
        games_played=('GAME_ID', 'count'),
        total_fantasy_pts=('CUSTOM_FP', 'sum'),
        avg_fantasy_pts=('CUSTOM_FP', 'mean')
    ).reset_index()
    
    # 6. Format and Sort Output
    grouped = grouped.sort_values(by='total_fantasy_pts', ascending=False)
    grouped['total_fantasy_pts'] = grouped['total_fantasy_pts'].round(2)
    grouped['avg_fantasy_pts'] = grouped['avg_fantasy_pts'].round(2)
    
    return grouped.to_dict(orient="records")

@app.post("/{season}/player/{player_id}/games")
def get_player_games(season: str, player_id: int, weights: ScoringWeights):
    global season_data
    
    if season not in season_data:
        raise HTTPException(status_code=404, detail="Season data not found")
        
    df_logs = season_data[season].copy()
    
    player_df = df_logs[df_logs['PLAYER_ID'] == player_id].copy()
    
    if player_df.empty:
        raise HTTPException(status_code=404, detail="Player not found")

    player_df['fantasy_score'] = apply_scoring(player_df, weights.model_dump())
    
    result = player_df[['GAME_DATE', 'MIN', 'fantasy_score']].sort_values(by='GAME_DATE', ascending=False)
    
    return result.to_dict(orient="records")

class GameDetailRequest(BaseModel):
    player_name: str
    game_id: str
    weights: ScoringWeights

@app.post("/{season}/game/detail")
def get_game_detail(season: str, req: GameDetailRequest):
    global season_data
    
    if season not in season_data:
        raise HTTPException(status_code=404, detail=f"Season '{season}' not found")
        
    df_logs = season_data[season]
    
    mask = (df_logs['PLAYER_NAME'] == req.player_name) & (df_logs['GAME_ID'] == req.game_id)
    game_row = df_logs[mask].copy()
    
    if game_row.empty:
        raise HTTPException(status_code=404, detail="Game or Player record not found")

    game_row['fantasy_score'] = apply_scoring(game_row, req.weights.model_dump())
    
    to_drop = ['TEAM_ABBREVIATION', 'TEAM_ID']
    result = game_row.drop(columns=[col for col in to_drop if col in game_row.columns])
    
    return result.iloc[0].to_dict()
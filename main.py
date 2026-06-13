from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel, field_validator
from datetime import date
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager

# Global dictionary to hold multiple seasons
season_data = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global season_data
    data_dir = Path(__file__).parent
    
    for file_path in data_dir.glob("*.parquet"):
        # If filename is 'nba_2025_26_gamelogs' -> parts will be ['nba', '2025', '26', 'gamelogs']
        parts = file_path.stem.split("_")
        
        # Grab just the year segments and join them with an underscore
        season_key = f"{parts[1]}_{parts[2]}"  # Results in '2025_26'
        
        season_data[season_key] = pd.read_parquet(file_path)
        print(f"Loaded {season_key}: {len(season_data[season_key])} rows.")
        
    yield
    season_data.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
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


# Endpoint 1: All players fantasy scores for a season
@app.post("/{season}/players/fantasy-scores")
def get_fantasy_scores(season: str, weights: ScoringWeights, limit: int = 50, offset: int = 0):
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

    paginated = grouped.iloc[offset: offset + limit]
    return {
        "total": len(grouped),
        "offset": offset,
        "limit": limit,
        "data": paginated.to_dict(orient="records")
    }


# Endpoint 2: All players fantasy scores for a season filtered by date range
class DateRangeScoringRequest(BaseModel):
    start_date: str  # Expected format: "YYYY-MM-DD"
    end_date: str    # Expected format: "YYYY-MM-DD"
    weights: ScoringWeights

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date(cls, v):
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format '{v}'. Expected YYYY-MM-DD.")
        return v

    @field_validator('end_date')
    @classmethod
    def end_after_start(cls, v, info):
        if 'start_date' in info.data and v < info.data['start_date']:
            raise ValueError("end_date must be on or after start_date.")
        return v

@app.post("/{season}/players/fantasy-scores/date-range")
def get_fantasy_scores_by_date(season: str, req: DateRangeScoringRequest, limit: int = 50, offset: int = 0):
    if season not in season_data:
        raise HTTPException(status_code=404, detail="Season data not found")
        
    df_logs = season_data[season]
    
    date_mask = (df_logs['GAME_DATE'] >= req.start_date) & (df_logs['GAME_DATE'] <= req.end_date)
    df_filtered = df_logs[date_mask].copy()
    
    if df_filtered.empty:
        return {"total": 0, "offset": offset, "limit": limit, "data": []}

    w = req.weights.model_dump()
    custom_scores = pd.Series(0.0, index=df_filtered.index)
    
    for stat, weight in w.items():
        if weight != 0 and stat in df_filtered.columns:
            custom_scores += df_filtered[stat] * weight
            
    df_filtered['CUSTOM_FP'] = custom_scores
    
    grouped = df_filtered.groupby(['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION']).agg(
        games_played=('GAME_ID', 'count'),
        total_fantasy_pts=('CUSTOM_FP', 'sum'),
        avg_fantasy_pts=('CUSTOM_FP', 'mean')
    ).reset_index()
    
    grouped = grouped.sort_values(by='total_fantasy_pts', ascending=False)
    grouped['total_fantasy_pts'] = grouped['total_fantasy_pts'].round(2)
    grouped['avg_fantasy_pts'] = grouped['avg_fantasy_pts'].round(2)

    paginated = grouped.iloc[offset: offset + limit]
    return {
        "total": len(grouped),
        "offset": offset,
        "limit": limit,
        "data": paginated.to_dict(orient="records")
    }


# Endpoint 3: All games for a specific player in a season with fantasy scores
@app.post("/{season}/player/{player_id}/games")
def get_player_games(season: str, player_id: int, weights: ScoringWeights):
    if season not in season_data:
        raise HTTPException(status_code=404, detail="Season data not found")
        
    df_logs = season_data[season].copy()
    
    player_df = df_logs[df_logs['PLAYER_ID'] == player_id].copy()
    
    if player_df.empty:
        raise HTTPException(status_code=404, detail="Player not found")

    player_df['fantasy_score'] = apply_scoring(player_df, weights.model_dump())

    stat_cols = ['GAME_DATE', 'GAME_ID', 'MATCHUP', 'WL', 'MIN',
                 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
                 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
                 'fantasy_score']
    result = player_df[stat_cols].sort_values(by='GAME_DATE', ascending=False)
    
    return result.to_dict(orient="records")


# Endpoint 4: All stats and fantasy score for a specific player in a specific game
class GameDetailRequest(BaseModel):
    player_name: str
    game_id: str
    weights: ScoringWeights

@app.post("/{season}/game/detail")
def get_game_detail(season: str, req: GameDetailRequest):
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

    row = result.iloc[0]
    return {
        k: (int(v.item()) if hasattr(v, 'item') and not isinstance(v, float) else round(float(v), 2) if isinstance(v, float) else v)
        for k, v in row.items()
    }



class PrecisionAuctionConfig(BaseModel):
    weights: ScoringWeights
    regular_weeks: int = 20
    playoff_weeks: int = 4
    post_season_weightage: float = 1.5
    num_teams: int = 10
    roster_size: int = 13
    total_budget_per_team: float = 200.0

@app.post("/{season}/players/precision-auction-values")
async def calculate_precision_auction_values(season: str, config: PrecisionAuctionConfig):
    global season_data
    
    if season not in season_data:
        raise HTTPException(status_code=404, detail=f"Season '{season}' not found")
        
    df = season_data[season].copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # 1. Align with the Monday anchor
    first_game_date = df['GAME_DATE'].min()
    days_to_subtract = first_game_date.weekday() 
    monday_anchor = first_game_date - pd.Timedelta(days=days_to_subtract)
    
    # 2. Calculate the actual calendar week (7-day intervals)
    df['days_since_monday_anchor'] = (df['GAME_DATE'] - monday_anchor).dt.days
    df['calendar_week'] = (df['days_since_monday_anchor'] // 7) + 1
    
    # 3. Map Calendar Weeks to Fantasy Weeks to handle the 14-day All-Star Week
    # We use np.select to apply conditional routing to the entire column at once
    conditions = [
        df['calendar_week'] <= 16,                         # Before All-Star Break
        df['calendar_week'].isin([17, 18]),                # The 14-day All-Star Week combo
        df['calendar_week'] > 18                           # Post All-Star Break
    ]
    
    choices = [
        df['calendar_week'],                               # Keep 1-16 exactly as is
        17,                                                # Force both 17 and 18 into Fantasy Week 17
        df['calendar_week'] - 1                            # Shift later weeks down by 1 to compress
    ]
    
    # Assign the final compressed fantasy week numbers
    df['week_number'] = np.select(conditions, choices, default=1)
    
    # 4. Filter for your total active fantasy timeline
    # 20 regular weeks + playoff_weeks (e.g., 4) = 24 total fantasy weeks
    total_fantasy_weeks = config.regular_weeks + config.playoff_weeks
    df = df[df['week_number'] <= total_fantasy_weeks]
    
    # Calculate base fantasy points per game
    df['base_fp'] = apply_scoring(df, config.weights.model_dump())
    
    # Apply post-season weights
    df['final_fp'] = np.where(
        df['week_number'] > config.regular_weeks,
        df['base_fp'] * config.post_season_weightage,
        df['base_fp']
    )
    
    # --- STEP 2: WEEKLY PLAYER SUMMATION ---
    # Aggregate points per player, PER WEEK
    weekly_player_stats = df.groupby(
        ['week_number', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION']
    )['final_fp'].sum().reset_index()
    
    # --- STEP 3: DYNAMIC WEEKLY REPLACEMENT BASES ---
    # Determine how many players are starting across the league each week
    replacement_rank = config.num_teams * config.roster_size
    
    weekly_vorp_list = []
    
    # Loop through each individual week to find its unique replacement baseline
    for week in range(1, total_fantasy_weeks + 1):
        week_df = weekly_player_stats[weekly_player_stats['week_number'] == week].copy()
        
        if week_df.empty:
            continue
            
        # Sort top scorers down to lowest scorers for this specific week
        week_df = week_df.sort_values(by='final_fp', ascending=False).reset_index(drop=True)
        
        # Identify the boundary baseline score for this week
        rep_index = min(replacement_rank, len(week_df) - 1)
        replacement_baseline_score = week_df.loc[rep_index, 'final_fp']
        
        # CRITICAL CHANGE: Calculate VORP and enforce the lower bound floor of 0
        week_df['weekly_vorp'] = week_df['final_fp'] - replacement_baseline_score
        week_df['weekly_vorp'] = week_df['weekly_vorp'].clip(lower=0)
        
        weekly_vorp_list.append(week_df)
    
    # Merge all processed weeks back into one contiguous table
    processed_weekly_df = pd.concat(weekly_vorp_list, ignore_index=True)
    
    # --- STEP 4: SEASONAL AGGREGATION OF WEEKLY VALUE ---
    # Sum up the non-negative weekly VORP metrics for every player
    final_player_pool = processed_weekly_df.groupby(
        ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION']
    ).agg(
        total_weighted_points=('final_fp', 'sum'),
        total_accumulated_vorp=('weekly_vorp', 'sum')
    ).reset_index()
    
    # --- STEP 5: PURE ECONOMIC AUCTION VALUATION ---
    top_players_vorp = final_player_pool.head(rep_index)['total_accumulated_vorp']
    total_league_vorp = top_players_vorp.sum()
    total_league_cash_pool = config.num_teams * config.total_budget_per_team
    
    # Direct proportional distribution without artificial baseline floors
    if total_league_vorp > 0:
        final_player_pool['auction_value'] = (
            final_player_pool['total_accumulated_vorp'] / total_league_vorp
        ) * total_league_cash_pool
    else:
        final_player_pool['auction_value'] = 0.0
        
    # Formatting adjustments for frontend presentation
    final_player_pool = final_player_pool.sort_values(by='auction_value', ascending=False).reset_index(drop=True)
    final_player_pool['auction_value'] = final_player_pool['auction_value'].round(2)
    final_player_pool['total_weighted_points'] = final_player_pool['total_weighted_points'].round(2)
    final_player_pool['total_accumulated_vorp'] = final_player_pool['total_accumulated_vorp'].round(2)
    
    return final_player_pool.to_dict(orient="records")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel, field_validator
from datetime import date
from typing import Literal
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
        season_key = f"{parts[1]}"  # Results in '2025-26'
        
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

def paginate(df: pd.DataFrame, limit: int, offset: int) -> dict:
    total = len(df)
    paginated = df.iloc[offset: offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "data": paginated.to_dict(orient="records")
    }

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
def get_fantasy_scores(
    season: str,
    weights: ScoringWeights,
    limit: int = 50,
    offset: int = 0,
    sort_by: Literal["total", "avg"] = "total",
):
    if season not in season_data:
        raise HTTPException(status_code=404, detail="Season data not found")
        
    df_logs = season_data[season].copy()
    
    w = weights.model_dump()
    
    custom_scores = pd.Series(0.0, index=df_logs.index)
    
    for stat, weight in w.items():
        if weight != 0:
            custom_scores += df_logs[stat] * weight
            
    df_logs['CUSTOM_FP'] = custom_scores
    
    grouped = df_logs.groupby(['PLAYER_ID', 'PLAYER_NAME']).agg(
        games_played=('GAME_ID', 'count'),
        total_fantasy_pts=('CUSTOM_FP', 'sum'),
        avg_fantasy_pts=('CUSTOM_FP', 'mean')
    ).reset_index()
    
    sort_col = 'total_fantasy_pts' if sort_by == "total" else 'avg_fantasy_pts'
    grouped = grouped.sort_values(by=sort_col, ascending=False)
    grouped['total_fantasy_pts'] = grouped['total_fantasy_pts'].round(2)
    grouped['avg_fantasy_pts'] = grouped['avg_fantasy_pts'].round(2)

    return paginate(grouped, limit, offset)


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
def get_fantasy_scores_by_date(
    season: str,
    req: DateRangeScoringRequest,
    limit: int = 50,
    offset: int = 0,
    sort_by: Literal["total", "avg"] = "total",
):
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
    
    grouped = df_filtered.groupby(['PLAYER_ID', 'PLAYER_NAME']).agg(
        games_played=('GAME_ID', 'count'),
        total_fantasy_pts=('CUSTOM_FP', 'sum'),
        avg_fantasy_pts=('CUSTOM_FP', 'mean')
    ).reset_index()
    
    sort_col = 'total_fantasy_pts' if sort_by == "total" else 'avg_fantasy_pts'
    grouped = grouped.sort_values(by=sort_col, ascending=False)
    grouped['total_fantasy_pts'] = grouped['total_fantasy_pts'].round(2)
    grouped['avg_fantasy_pts'] = grouped['avg_fantasy_pts'].round(2)

    return paginate(grouped, limit, offset)


# Endpoint 3: All games for a specific player in a season with fantasy scores
@app.post("/{season}/player/{player_id}/games")
def get_player_games(
    season: str,
    player_id: int,
    weights: ScoringWeights,
    limit: int = 50,
    offset: int = 0,
    sort_by: Literal["date", "fantasy_score"] = "date",
    order: Literal["asc", "desc"] = "desc",
):
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
    sort_col = 'GAME_DATE' if sort_by == "date" else 'fantasy_score'
    ascending = order == "asc"
    result = player_df[stat_cols].sort_values(by=sort_col, ascending=ascending)
    
    return paginate(result, limit, offset)


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
async def calculate_precision_auction_values(
    season: str,
    config: PrecisionAuctionConfig,
    limit: int = 50,
    offset: int = 0,
):
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
    
    # --- STEP 1b: MODULAR ALL-STAR WEEK CONFIGURATION ---
    # Maps specific seasons to the calendar week containing the All-Star break.
    # Standard leagues combine this week and the following week into a single match.
    ALL_STAR_WEEK_MAP = {
        "2022-23": 18,
        "2023-24": 17,
        "2024-25": 17,
        "2025-26": 17,  # Default or dynamically adjusted based on schedule anchor
    }
    
    # Safely fetch the break week for the current route parameter, defaulting to 17
    asb_calendar_week = ALL_STAR_WEEK_MAP.get(season, 17)
    following_week = asb_calendar_week + 1

    # --- STEP 2: WEEKLY PLAYER SUMMATION (Pre-existing grouping) ---
    # [Your base logic continues here: df['base_fp'] calculation and final_fp weights]

    # --- STEP 3: DYNAMIC WEEKLY REPLACEMENT BASES ---
    # 3. Map Calendar Weeks to Fantasy Weeks to handle the 14-day All-Star Week
    conditions = [
        df['calendar_week'] < asb_calendar_week,
        df['calendar_week'].isin([asb_calendar_week, following_week]),
        df['calendar_week'] > following_week
    ]
    
    choices = [
        df['calendar_week'],                # Weeks before the break stay 1:1
        asb_calendar_week,                  # Both break weeks merge into the anchor week
        df['calendar_week'] - 1             # Weeks after shift down by 1 to compress the timeline
    ]
    
    df['week_number'] = np.select(conditions, choices, default=1)
    
    # Filter for your total active fantasy timeline (Regular + Playoffs)
    total_fantasy_weeks = config.regular_weeks + config.playoff_weeks
    df = df[df['week_number'] <= total_fantasy_weeks]
    
    # 4. Filter for your total active fantasy timeline
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
    weekly_player_stats = df.groupby(
        ['week_number', 'PLAYER_ID', 'PLAYER_NAME']
    )['final_fp'].sum().reset_index()
    
    # --- STEP 3: DYNAMIC WEEKLY REPLACEMENT BASES ---
    total_draftable_slots = config.num_teams * config.roster_size
    weekly_vorp_list = []
    
    for week in range(1, total_fantasy_weeks + 1):
        week_df = weekly_player_stats[weekly_player_stats['week_number'] == week].copy()
        
        if week_df.empty:
            continue
            
        week_df = week_df.sort_values(by='final_fp', ascending=False).reset_index(drop=True)
        
        # Identify the boundary baseline score for this week
        rep_index = min(total_draftable_slots, len(week_df) - 1)
        replacement_baseline_score = week_df.loc[rep_index, 'final_fp']
        
        # Calculate weekly VORP and enforce the lower bound floor of 0
        week_df['weekly_vorp'] = week_df['final_fp'] - replacement_baseline_score
        week_df['weekly_vorp'] = week_df['weekly_vorp'].clip(lower=0)
        
        weekly_vorp_list.append(week_df)
    
    processed_weekly_df = pd.concat(weekly_vorp_list, ignore_index=True)
    
    # --- STEP 4: SEASONAL AGGREGATION OF WEEKLY VALUE ---
    final_player_pool = processed_weekly_df.groupby(
        ['PLAYER_ID', 'PLAYER_NAME']
    ).agg(
        total_weighted_points=('final_fp', 'sum'),
        total_accumulated_vorp=('weekly_vorp', 'sum')
    ).reset_index()
    
    # --- STEP 4b: CRITICAL RETROACTIVE SEASONAL BASELINE ADJUSTMENT ---
    # 1. Sort by total accumulated VORP descending to find the draftable boundary
    final_player_pool = final_player_pool.sort_values(by='total_accumulated_vorp', ascending=False).reset_index(drop=True)
    
    # 2. Find the exact VORP score of the replacement_rank player (e.g. index 129 for a 130-player league)
    boundary_index = min(total_draftable_slots, len(final_player_pool) - 1)
    seasonal_replacement_vorp = final_player_pool.loc[boundary_index, 'total_accumulated_vorp']
    
    # 3. Subtract that boundary VORP from EVERY player, and clip at 0
    # This guarantees the replacement player hits EXACTLY 0, and anyone below them drops to 0
    final_player_pool['adjusted_vorp'] = final_player_pool['total_accumulated_vorp'] - seasonal_replacement_vorp
    final_player_pool['adjusted_vorp'] = final_player_pool['adjusted_vorp'].clip(lower=0)
    
    # --- STEP 5: LINEAR RESTRICTED-POOL ECONOMIC VALUATION ---
    total_league_cash_pool = config.num_teams * config.total_budget_per_team

    # Isolate only the top drafted tier to compute the clean market denominator
    top_drafted_df = final_player_pool.head(total_draftable_slots).copy()
    total_league_vorp = top_drafted_df['adjusted_vorp'].sum()

    # Reserve $1.00 minimum for every single drafted slot
    mandatory_roster_cost = total_draftable_slots * 1.0
    premium_bidding_pool = total_league_cash_pool - mandatory_roster_cost

    # Initialize the whole pool's values to $1.00
    final_player_pool['auction_value'] = 1.0

    # Distribute premium capital based on the linear adjusted VORP share
    if total_league_vorp > 0:
        premium_values = (top_drafted_df['adjusted_vorp'] / total_league_vorp) * premium_bidding_pool
        final_player_pool.loc[:total_draftable_slots - 1, 'auction_value'] = 1.0 + premium_values
    else:
        final_player_pool.loc[:total_draftable_slots - 1, 'auction_value'] = 1.0
        
    # Final formatting adjustments
    # Primary sort: auction_value desc. Tiebreakers: adjusted_vorp desc, then total_weighted_points desc.
    final_player_pool = final_player_pool.sort_values(
        by=['auction_value', 'total_accumulated_vorp', 'total_weighted_points'],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    final_player_pool['auction_value'] = final_player_pool['auction_value'].round(2)
    final_player_pool['total_weighted_points'] = final_player_pool['total_weighted_points'].round(2)
    final_player_pool['total_accumulated_vorp'] = final_player_pool['total_accumulated_vorp'].round(2)
    final_player_pool['adjusted_vorp'] = final_player_pool['adjusted_vorp'].round(2)
    
    return paginate(final_player_pool, limit, offset)
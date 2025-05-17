import pandas as pd
import numpy as np
import re
import time
import warnings
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
from typing import List
warnings.filterwarnings('ignore')

def calculate_study_title_score(df: pd.DataFrame) -> pd.Series:
    ordered_levels = [
        "Middle school diploma",
        "High school graduation",
        "Professional qualification",
        "Three-year degree",
        "Five-year degree",
        "master's degree",
        "Doctorate"
    ]

    level_to_rank = {level: idx for idx, level in enumerate(ordered_levels)}
    max_distance = len(ordered_levels) - 1  # 6 in this case

    def _calculate_score(candidate_level, required_level):
        if pd.isna(candidate_level) or pd.isna(required_level):
            return np.nan
        if candidate_level not in level_to_rank or required_level not in level_to_rank:
            return np.nan

        diff = level_to_rank[candidate_level] - level_to_rank[required_level]
        return diff / max_distance  # preserve sign, normalize

    return df.apply(
        lambda row: _calculate_score(row.get('Study Title'), row.get('Study Level')),
        axis=1
    )


def calculate_experience_match_score(df: pd.DataFrame) -> pd.Series:
    candidate_exps = df['Years Experience_int']
    job_exps = df['Years Experience.1_int']
    global_min = pd.concat([candidate_exps, job_exps]).min()
    global_max = pd.concat([candidate_exps, job_exps]).max()
    max_range = global_max - global_min if global_max != global_min else 1  
 
    def _calculate_score(candidate_exp, job_req_exp):
        if pd.isna(job_req_exp):
            return 0

        diff = candidate_exp - job_req_exp
        return diff / max_range

    return df.apply(
        lambda row: _calculate_score(row.get('Years Experience_int'), row.get('Years Experience.1_int')),
        axis=1
    ) 

def calculate_salary_fit_score(df: pd.DataFrame, is_expected=True) -> pd.Series:
    def _calculate_score(expected_ral, min_ral, max_ral):

        if pd.isna(expected_ral) or pd.isna(min_ral) or pd.isna(max_ral):
            return np.nan

        if expected_ral >= min_ral and expected_ral <= max_ral:
            return 1.0 

        if expected_ral < min_ral:
            distance = expected_ral - min_ral
        elif expected_ral > max_ral:
            distance = expected_ral - max_ral

        range_size = max_ral - min_ral
        scale_factor = range_size if range_size > 0 else min_ral # Avoid zero/negative division

        if scale_factor <= 0: scale_factor = 1000 

        return distance / scale_factor
    return df.apply(
        lambda row: _calculate_score(row.get('Expected Ral' if is_expected else 'Current Ral'), row.get('Minimum Ral'), row.get('Ral Maximum')),
        axis=1
    )


model = SentenceTransformer('all-MiniLM-L6-v2') 

def calculate_study_area_score(df):
    # Precompute embeddings
    all_study_areas = pd.concat([df['Study area'], df['Study Area.1']]).dropna().unique()
    embeddings = {s: model.encode(s, convert_to_tensor=True) for s in all_study_areas}

    def _score(a, b):
        if pd.isna(a) or pd.isna(b):
            return np.nan
        emb_a = embeddings.get(a)
        emb_b = embeddings.get(b)
        return float(util.cos_sim(emb_a, emb_b))

    return df.apply(lambda row: _score(row.get('Study area'), row.get('Study Area.1')), axis=1)

def calculate_professional_similarity_score(df: pd.DataFrame) -> pd.Series:
    def build_text(*fields):
        non_empty = [str(f).strip() for f in fields if pd.notna(f) and str(f).strip()]
        if not non_empty:
            return None
        return ' | '.join(non_empty)

    embedding_cache = {}

    def get_embedding(text):
        if text in embedding_cache:
            return embedding_cache[text]
        embedding = model.encode(text, convert_to_tensor=True)
        embedding_cache[text] = embedding
        return embedding

    def _similarity(row):
        candidate_text = build_text(row.get('Sector'), row.get('Last Role'))
        job_text = build_text(row.get('Job Family Hiring'), row.get('Job Title Hiring'))

        if candidate_text is None or job_text is None:
            return np.nan

        emb_a = get_embedding(candidate_text)
        emb_b = get_embedding(job_text)
        return float(util.cos_sim(emb_a, emb_b))

    return df.apply(_similarity, axis=1)

def create_candidate_text(row):
    parts = []

    if pd.notna(row.get('Study Title')) and pd.notna(row.get('Study area')):
        parts.append(f"{row['Study Title']} in {row['Study area']}")
    elif pd.notna(row.get('Study Title')):
        parts.append(f"Studied {row['Study Title']}")
    elif pd.notna(row.get('Study area')):
        parts.append(f"Studied in {row['Study area']}")

    if pd.notna(row.get('Sector')):
        parts.append(f"Worked in the {row['Sector']} sector")

    if pd.notna(row.get('Last Role')):
        parts.append(f"Last held the role of {row['Last Role']}")

    if pd.notna(row.get('Years Experience')):
        parts.append(f"with {row['Years Experience']} years of experience")

    if pd.notna(row.get('TAG')):
        parts.append(f"Key skills include: {row['TAG']}")

    return ". ".join(parts) + "."

def create_job_text(row):
    parts = []

    if pd.notna(row.get('Job Title Hiring')):
        parts.append(f"Job title: {row['Job Title Hiring']}")

    if pd.notna(row.get('Job Family Hiring')):
        parts.append(f"Department: {row['Job Family Hiring']}")

    if pd.notna(row.get('Recruitment Request')):
        parts.append(f"Recruitment context: {row['Recruitment Request']}")

    if pd.notna(row.get('Job Description')):
        parts.append(f"Job description: {row['Job Description']}")

    if pd.notna(row.get('Candidate Profile')):
        parts.append(f"Ideal candidate profile: {row['Candidate Profile']}")

    if pd.notna(row.get('Study Level')) and pd.notna(row.get('Study Area.1')):
        parts.append(f"Educational requirement: {row['Study Level']} in {row['Study Area.1']}")
    elif pd.notna(row.get('Study Level')):
        parts.append(f"Educational requirement: {row['Study Level']}")
    elif pd.notna(row.get('Study Area.1')):
        parts.append(f"Field of study required: {row['Study Area.1']}")

    if pd.notna(row.get('Years Experience.1')):
        parts.append(f"Requires {row['Years Experience.1']} years of experience")

    return ". ".join(parts) + "."


def prepare_nlp_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates combined text columns for candidate profile and job requirements.

    Args:
        df: Input DataFrame. Assumes parsed experience columns exist if needed in text.

    Returns:
        DataFrame with 'candidate_text' and 'job_text' columns added.
    """
    df_processed = df.copy() # Work on a copy

    # Create text columns, filling potential NaNs with empty strings for NLP
    df_processed['candidate_text'] = df_processed.apply(create_candidate_text, axis=1).fillna("")
    df_processed['job_text'] = df_processed.apply(create_job_text, axis=1).fillna("")

    return df_processed


def calculate_distance(coord1, coord2):
    try:
        return geodesic(coord1, coord2).kilometers
    except:
        return None  

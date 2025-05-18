import pandas as pd
import numpy as np
import re
import time
import warnings
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
from typing import List, Tuple, Optional, Dict, Union

warnings.filterwarnings('ignore')


def calculate_study_title_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the normalized difference between candidate and required study levels.

    This function maps education levels to a numerical ranking and computes the
    normalized difference between a candidate's level and the job's requirement.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'Study Title' and 'Study Level' columns.

    Returns
    -------
    pandas.Series
        A Series of normalized score differences between candidate and job study levels.
    """
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
    max_distance = len(ordered_levels) - 1

    def _calculate_score(candidate_level: str, required_level: str):
        if pd.isna(candidate_level) or pd.isna(required_level):
            return np.nan
        if candidate_level not in level_to_rank or required_level not in level_to_rank:
            return np.nan

        diff = level_to_rank[candidate_level] - level_to_rank[required_level]
        return diff / max_distance

    return df.apply(
        lambda row: _calculate_score(row.get('Study Title'), row.get('Study Level')),
        axis=1
    )


def calculate_experience_match_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the normalized difference between candidate and required experience.

    The function compares years of experience and returns a normalized score
    based on the range of values found in the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'Years Experience_int' and 'Years Experience.1_int'.

    Returns
    -------
    pandas.Series
        A Series containing normalized experience difference scores.
    """
    candidate_exps = df['Years Experience_int']
    job_exps = df['Years Experience.1_int']
    global_min = pd.concat([candidate_exps, job_exps]).min()
    global_max = pd.concat([candidate_exps, job_exps]).max()
    max_range = global_max - global_min if global_max != global_min else 1  

    def _calculate_score(candidate_exp: float, job_req_exp: float) -> float:
        if pd.isna(job_req_exp):
            return 0
        diff = candidate_exp - job_req_exp
        return diff / max_range

    return df.apply(
        lambda row: _calculate_score(row.get('Years Experience_int'), row.get('Years Experience.1_int')),
        axis=1
    ) 


def calculate_salary_fit_score(df: pd.DataFrame, is_expected: bool = True) -> pd.Series:
    """
    Calculate the salary fit score between a candidate's salary and job's salary range.

    Returns 1.0 if candidate's salary is within range; otherwise, returns a normalized
    score based on how far it is from the closest bound.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with salary information, including candidate and job salary columns.
    is_expected : bool, optional
        If True, uses 'Expected Ral'; if False, uses 'Current Ral'. Default is True.

    Returns
    -------
    pandas.Series
        A Series of salary fit scores.
    """
    def _calculate_score(expected_ral: float, min_ral: float, max_ral: float):
        if pd.isna(expected_ral) or pd.isna(min_ral) or pd.isna(max_ral):
            return np.nan
        if expected_ral >= min_ral and expected_ral <= max_ral:
            return 1.0 

        distance = expected_ral - min_ral if expected_ral < min_ral else expected_ral - max_ral
        range_size = max_ral - min_ral
        scale_factor = range_size if range_size > 0 else min_ral
        if scale_factor <= 0:
            scale_factor = 1000
        return distance / scale_factor

    return df.apply(
        lambda row: _calculate_score(
            row.get('Expected Ral' if is_expected else 'Current Ral'),
            row.get('Minimum Ral'),
            row.get('Ral Maximum')
        ),
        axis=1
    )


model = SentenceTransformer('all-MiniLM-L6-v2') 


def calculate_study_area_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate semantic similarity between candidate and required study areas.

    Uses sentence embeddings and cosine similarity to quantify alignment between
    study fields.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'Study area' and 'Study Area.1' columns.

    Returns
    -------
    pandas.Series
        A Series of cosine similarity scores.
    """
    all_study_areas = pd.concat([df['Study area'], df['Study Area.1']]).dropna().unique()
    embeddings = {s: model.encode(s, convert_to_tensor=True) for s in all_study_areas}

    def _score(a: str, b: str):
        if pd.isna(a) or pd.isna(b):
            return np.nan
        emb_a = embeddings.get(a)
        emb_b = embeddings.get(b)
        return float(util.cos_sim(emb_a, emb_b))

    return df.apply(lambda row: _score(row.get('Study area'), row.get('Study Area.1')), axis=1)


def calculate_professional_similarity_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate semantic similarity between candidate's background and job description.

    Compares sector and last role against job family and job title using sentence
    embeddings and cosine similarity.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'Sector', 'Last Role', 'Job Family Hiring', and 'Job Title Hiring'.

    Returns
    -------
    pandas.Series
        A Series of professional similarity scores.
    """
    def build_text(*fields: str) -> Optional[str]:
        non_empty = [str(f).strip() for f in fields if pd.notna(f) and str(f).strip()]
        if not non_empty:
            return None
        return ' | '.join(non_empty)

    embedding_cache: Dict[str, any] = {}

    def get_embedding(text: str):
        if text in embedding_cache:
            return embedding_cache[text]
        embedding = model.encode(text, convert_to_tensor=True)
        embedding_cache[text] = embedding
        return embedding

    def _similarity(row: pd.Series):
        candidate_text = build_text(row.get('Sector'), row.get('Last Role'))
        job_text = build_text(row.get('Job Family Hiring'), row.get('Job Title Hiring'))

        if candidate_text is None or job_text is None:
            return np.nan

        emb_a = get_embedding(candidate_text)
        emb_b = get_embedding(job_text)
        return float(util.cos_sim(emb_a, emb_b))

    return df.apply(_similarity, axis=1)


def create_candidate_text(row: pd.Series) -> str:
    """
    Create a text description summarizing a candidate's profile.

    Combines fields such as education, sector, last role, experience, and skills
    into a single formatted string.

    Parameters
    ----------
    row : pandas.Series
        A row from the candidate DataFrame.

    Returns
    -------
    str
        A text summary of the candidate.
    """
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


def create_job_text(row: pd.Series) -> str:
    """
    Create a text description summarizing a job posting.

    Combines job title, department, job description, and requirements into a
    single formatted string for use in NLP models.

    Parameters
    ----------
    row : pandas.Series
        A row from the job DataFrame.

    Returns
    -------
    str
        A text summary of the job posting.
    """
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
    Create candidate_text and job_text columns for NLP similarity calculations.

    This function adds text summaries for both candidate and job profiles to the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame with candidate and job information.

    Returns
    -------
    pandas.DataFrame
        DataFrame with added 'candidate_text' and 'job_text' columns.
    """
    df_processed = df.copy()
    df_processed['candidate_text'] = df_processed.apply(create_candidate_text, axis=1).fillna("")
    df_processed['job_text'] = df_processed.apply(create_job_text, axis=1).fillna("")
    return df_processed


def calculate_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> Optional[float]:
    """
    Compute geodesic distance in kilometers between two coordinate pairs.

    Parameters
    ----------
    coord1 : tuple of float
        First coordinate as (latitude, longitude).
    coord2 : tuple of float
        Second coordinate as (latitude, longitude).

    Returns
    -------
    float or None
        Distance in kilometers, or None if calculation fails.
    """
    try:
        return geodesic(coord1, coord2).kilometers
    except:
        return None

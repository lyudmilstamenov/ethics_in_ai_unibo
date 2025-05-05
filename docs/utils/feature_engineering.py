# feature_engineering.py

import pandas as pd
import numpy as np
import re
import time
import warnings

# Import all necessary libraries directly
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

from typing import Tuple, List
# Ignore warnings for cleaner output, but be cautious in real applications
warnings.filterwarnings('ignore')

# --- Global/Module-level Constants and Objects ---
# Geocoding cache and geolocator instance
# Making it global for simplicity, manage lifespan carefully in complex applications
_geocoding_cache = {}
_geolocator = Nominatim(user_agent="geopy_distance_calculator_v7") # Use a unique user agent
# Define HQ location as a constant
HQ_LOCATION_STR = "Via dei Fornari 12, Bologna, Italy"
_hq_coords = None # Will store HQ coordinates after first call


# --- Helper Functions ---

def parse_experience_string(exp_str):
    """Parses a string like '[0-1]', '[+10]', '[3-5] | [1-3]' into a float."""
    if pd.isna(exp_str):
        return np.nan
    if not isinstance(exp_str, str):
         # If it's already a number, return it as float
         try:
             return float(exp_str)
         except (ValueError, TypeError):
             return np.nan # Handle cases where non-string/non-numeric sneaks in


    exp_str = exp_str.strip()
    if not exp_str:
        return np.nan

    # Handle combined formats like '[3-5] | [1-3]'
    parts = exp_str.split('|')
    values = []

    for part in parts:
        part = part.strip().replace('[', '').replace(']', '') # Remove brackets
        if not part: continue

        try:
            if '-' in part:
                if part.startswith('+'): # Handle '+10' case explicitly if it appears as '+10'
                     # Assuming '+10' in a range context means >= 10
                     value = float(part.replace('+', '')) # Treat +10 as 10 or start of range
                else:
                    low, high = map(float, part.split('-'))
                    value = (low + high) / 2.0 # Midpoint of range
            elif part.startswith('+'): # Handle '+10' when it's just '+10' after stripping []
                 value = float(part.replace('+', '')) # Treat +10 as 10
            else:
                value = float(part) # Handle single numbers like '0', '5', '10'

            values.append(value)
        except ValueError:
            # print(f"Warning: Could not parse experience string part '{part}' from '{exp_str}'")
            continue # Ignore parts that cannot be parsed

    if values:
        # For multiple ranges/values separated by '|', take the maximum experience
        return max(values)
    else:
        return np.nan # Return NaN if no parts could be parsed

def get_coordinates_cached(location_str):
    """Gets coordinates for a location string using cache and respecting Nominatim limits."""
    if pd.isna(location_str) or str(location_str).strip() == "":
        return None
    location_str = str(location_str).strip() # Ensure it's a string

    if location_str in _geocoding_cache:
        # print(f"Cache hit for '{location_str}'") # Debugging cache
        return _geocoding_cache[location_str]

    try:
        # Use try-except for geocoding failures and rate limiting
        # Increased timeout for robustness
        location = _geolocator.geocode(location_str, timeout=10)
        if location:
            coords = (location.latitude, location.longitude)
            _geocoding_cache[location_str] = coords
             # Add a small delay to respect Nominatim rate limits *only on success*
            time.sleep(1.1) # Slightly more than 1 second
            # print(f"Geocoded '{location_str}'") # Debugging geocoding
            return coords
        else:
            _geocoding_cache[location_str] = None # Cache failure
            time.sleep(0.5) # Shorter wait on failure but still wait
            # print(f"Geocoding failed for '{location_str}' (no result)") # Debugging geocoding
            return None
    except Exception as e:
        # print(f"Geocoding failed for '{location_str}': {e}") # Debugging geocoding errors
        _geocoding_cache[location_str] = None # Cache failure
        time.sleep(1.1) # Still wait to avoid hitting the service too hard on errors
        return None

def calculate_distance_to_hq(row_location_str, hq_coords):
    """Calculates geodesic distance between a location string and HQ coordinates."""
    if hq_coords is None:
        return np.nan # Cannot calculate if HQ coords are unknown

    candidate_coords = get_coordinates_cached(row_location_str)

    if candidate_coords:
        try:
            # Calculate geodesic distance in kilometers
            distance_km = geodesic(candidate_coords, hq_coords).km
            return distance_km
        except Exception as e:
            # print(f"Distance calculation failed for coords {candidate_coords} and {hq_coords}: {e}") # Debugging distance
            return np.nan
    else:
        return np.nan # Could not geocode candidate location


# --- Feature Calculation Functions ---

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


from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, fast, accurate

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


def calculate_nlp_similarity_st(df_with_text: pd.DataFrame, st_model: SentenceTransformer) -> pd.Series:
    """
    Calculates NLP similarity using a pre-loaded Sentence Transformer model.
    Assumes 'candidate_text' and 'job_text' columns exist and st_model is loaded.

    Args:
        df_with_text: Input DataFrame with 'candidate_text' and 'job_text' columns.
        st_model: A pre-loaded SentenceTransformer model instance.

    Returns:
        A pandas Series with NLP similarity scores (0 to 1 range usually).
    """
    if st_model is None:
         print("Error: Sentence Transformer model is None.")
         return pd.Series([np.nan] * len(df_with_text), index=df_with_text.index)

    try:
        # Generate embeddings in batches for efficiency (SentenceTransformer handles this)
        # Ensure texts are strings (handled by .fillna("") in prepare_nlp_text_columns)
        candidate_embeddings = st_model.encode(df_with_text['candidate_text'].tolist(), show_progress_bar=False, convert_to_numpy=True)
        job_embeddings = st_model.encode(df_with_text['job_text'].tolist(), show_progress_bar=False, convert_to_numpy=True)

        # Calculate cosine similarity using sklearn's utility for clarity
        # Handle potential empty strings resulting in zero vectors - cosine_similarity handles rows of zeros by returning 0 or NaN depending on sklearn version.
        # Let's manually set similarity to 0 if either text was empty or embedding calculation failed
        similarity_scores = []
        for i in range(len(df_with_text)):
            # Check if original text was empty (after fillna) or if embedding might be zero/nan (less likely with ST but defensive)
            if not df_with_text.loc[i, 'candidate_text'] or not df_with_text.loc[i, 'job_text']:
                 similarity_scores.append(0.0)
            else:
                # Cosine similarity between two vectors
                cand_emb = candidate_embeddings[i].reshape(1, -1)
                job_emb = job_embeddings[i].reshape(1, -1)
                score = cosine_similarity(cand_emb, job_emb)[0][0]
                similarity_scores.append(score)

        return pd.Series(similarity_scores, index=df_with_text.index)

    except Exception as e:
        print(f"Error during ST embedding or similarity calculation: {e}")
        return pd.Series([np.nan] * len(df_with_text), index=df_with_text.index)


def calculate_nlp_similarity_tfidf(df_with_text: pd.DataFrame, tfidf_vectorizer: TfidfVectorizer) -> pd.Series:
    """
    Calculates NLP similarity using a pre-fitted TF-IDF vectorizer.
    Assumes 'candidate_text' and 'job_text' columns exist and tfidf_vectorizer is fitted
    on the corpus containing these texts.

    Args:
        df_with_text: Input DataFrame with 'candidate_text' and 'job_text' columns.
        tfidf_vectorizer: A pre-fitted TfidfVectorizer instance.

    Returns:
        A pandas Series with NLP similarity scores (0 to 1 range).
    """
    if tfidf_vectorizer is None:
         print("Error: TF-IDF vectorizer is None.")
         return pd.Series([np.nan] * len(df_with_text), index=df_with_text.index)

    try:
        # Transform candidate and job texts into TF-IDF vectors
        # TfidfVectorizer handles empty strings by returning zero vectors
        candidate_tfidf = tfidf_vectorizer.transform(df_with_text['candidate_text'])
        job_tfidf = tfidf_vectorizer.transform(df_with_text['job_text'])

        # Calculate cosine similarity between candidate and job vectors
        # Using row-wise calculation
        similarity_scores = []
        for i in range(len(df_with_text)):
             # Get the sparse vectors for this row
             cand_vec = candidate_tfidf[i]
             job_vec = job_tfidf[i]

             # Cosine similarity between two vectors
             # Handle case where one or both vectors are zero (e.g., from empty text)
             if cand_vec.nnz == 0 or job_vec.nnz == 0: # nnz is number of non-zero elements
                  score = 0.0 # Assuming 0 similarity if text is missing/empty
             else:
                 # Calculate dot product of normalized vectors
                 score = cosine_similarity(cand_vec, job_vec)[0][0]
             similarity_scores.append(score)

        return pd.Series(similarity_scores, index=df_with_text.index)

    except Exception as e:
        print(f"Error during TF-IDF vectorization or similarity calculation: {e}")
        return pd.Series([np.nan] * len(df_with_text), index=df_with_text.index)


def calculate_geo_features(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Calculates geographical distance and proximity score to the HQ.
    Assumes residence columns exist.

    Args:
        df: Input DataFrame with residence columns.

    Returns:
        A tuple containing two pandas Series: (distance_km, proximity_score).
        Returns Series of NaNs if HQ geocoding fails.
    """
    # Get HQ coordinates once if not already cached
    global _hq_coords
    if _hq_coords is None:
         print(f"Attempting to geocode HQ location: '{HQ_LOCATION_STR}'")
         _hq_coords = get_coordinates_cached(HQ_LOCATION_STR)

    if _hq_coords is None:
         print(f"Error: Could not geocode HQ location '{HQ_LOCATION_STR}'. Cannot calculate geographical features.")
         nan_series = pd.Series([np.nan] * len(df), index=df.index)
         return nan_series, nan_series

    print("\n--- Calculating geographical distances... This may take some time depending on API rate limits. ---")

    # Apply distance calculation row by row
    # This can be slow due to geocoding API calls and delays
    distance_km_series = df.apply(
        lambda row: calculate_distance_to_hq(
             # Construct location string based on available residence columns
             ", ".join([
                 str(row[col]).strip() for col in ['Residence Italian City', 'Residence Italian Province', 'Residence Italian Region', 'Residence Country']
                 if pd.notna(row.get(col)) and str(row.get(col)).strip() # Check existence and non-empty after strip
             ]),
             _hq_coords
        ),
        axis=1
    )
    print("--- Geographical distance calculation finished. ---")


    # Create a proximity score from distance
    # Handle potential negative distance though unlikely, and NaNs
    proximity_score_series = distance_km_series.apply(
        lambda x: 1 / (x + 1) if pd.notna(x) and x >= 0 else (1.0 if pd.notna(x) and x < 0 else np.nan)
    )

    return distance_km_series, proximity_score_series


def calculate_overall_score(df: pd.DataFrame, score_columns: List[str]) -> pd.Series:
    """
    Calculates a simple average of specified score columns.
    Includes scaling 'Overall' if present and needed.

    Args:
        df: Input DataFrame containing the score columns.
        score_columns: List of column names to average (can include 'Overall_scaled').

    Returns:
        A pandas Series with the overall score.
    """
    # Ensure score columns exist, calculate scaled Overall if needed before averaging
    df_temp = df.copy() # Work on a copy

    # Scale 'Overall' performance score to 0-1 range if needed (assuming max is 5)
    # Check if 'Overall' exists and 'Overall_scaled' is requested but not present
    if 'Overall' in df_temp.columns and 'Overall_scaled' in score_columns and 'Overall_scaled' not in df_temp.columns:
        df_temp['Overall_scaled'] = pd.to_numeric(df_temp['Overall'], errors='coerce') / 5.0
    # Also ensure 'Overall_scaled' exists if it was already created outside this function
    elif 'Overall_scaled' in score_columns and 'Overall_scaled' not in df_temp.columns:
         # Handle case where Overall_scaled is requested but Overall wasn't in input or conversion failed
         print("Warning: 'Overall_scaled' requested for overall score, but 'Overall' column not found or conversion failed.")
         df_temp['Overall_scaled'] = np.nan # Add column with NaNs if it doesn't exist

    # Filter for columns that actually exist in the DataFrame for averaging
    actual_score_cols = [col for col in score_columns if col in df_temp.columns]

    if not actual_score_cols:
        print("Warning: No valid score columns provided for overall score calculation.")
        return pd.Series([np.nan] * len(df_temp), index=df_temp.index)

    # Simple average, ignoring NaNs
    # Ensure all columns in actual_score_cols are numeric for .mean()
    for col in actual_score_cols:
        if not pd.api.types.is_numeric_dtype(df_temp[col]):
             # Attempt to convert if not numeric (should ideally be done earlier)
             df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')


    return df_temp[actual_score_cols].mean(axis=1)


def drop_columns_except(df: pd.DataFrame, columns_to_keep: List[str]) -> pd.DataFrame:
    """
    Drops all columns from the DataFrame except those specified in columns_to_keep.

    Args:
        df: Input DataFrame.
        columns_to_keep: List of column names to retain.

    Returns:
        DataFrame containing only the specified columns.
    """
    df_processed = df.copy() # Work on a copy

    # Ensure columns to keep actually exist in the DataFrame
    actual_columns_to_keep = [col for col in columns_to_keep if col in df_processed.columns]

    # Identify columns to drop
    all_df_cols = set(df_processed.columns)
    columns_to_drop = list(all_df_cols - set(actual_columns_to_keep))

    if columns_to_drop:
        return df_processed.drop(columns=columns_to_drop)
    else:
        return df_processed # Nothing to drop
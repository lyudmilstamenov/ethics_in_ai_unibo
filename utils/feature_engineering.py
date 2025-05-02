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

# def parse_and_convert_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Parses specific string columns (like experience ranges) and converts
#     other numeric-like columns (RALs, Overall) to numeric.

#     Args:
#         df: Input DataFrame.

#     Returns:
#         DataFrame with parsed and converted numeric columns added/updated.
#     """
#     df_processed = df.copy() # Work on a copy

#     # Apply custom parsing to experience columns
#     df_processed['Years Experience_parsed'] = df_processed['Years Experience'].apply(parse_experience_string)
#     df_processed['Years Experience.1_parsed'] = df_processed['Years Experience.1'].apply(parse_experience_string)

#     # Convert RAL columns and Overall to numeric, coercing errors
#     # Added 'Overall' here as it's used in the final score and may need cleaning
#     ral_overall_cols = ['Current Ral', 'Expected Ral', 'Minimum Ral', 'Ral Maximum', 'Overall']
#     for col in ral_overall_cols:
#          # Simple conversion. If 'k' format exists, this will turn it to NaN.
#          # A more complex parser would be needed for 'k'.
#          df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

#     return df_processed

def calculate_experience_match_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculates a score indicating how well candidate experience matches job requirement.
    Assumes 'Years Experience_parsed' and 'Years Experience.1_parsed' exist.

    Args:
        df: Input DataFrame with parsed experience columns.

    Returns:
        A pandas Series with experience match scores.
    """
    # Simple inverse of difference, scaled. Smaller difference = higher score.
    def _calculate_score(candidate_exp_parsed, job_req_exp_parsed):
        diff = candidate_exp_parsed - job_req_exp_parsed
        # Score decreases as difference increases, never 0 unless diff is infinite
        return diff

    return df.apply(
        lambda row: _calculate_score(row.get('Years Experience'), row.get('Years Experience.1')), # Use .get for safety
        axis=1
    )

def calculate_salary_fit_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculates a score indicating how well expected RAL fits within the job range.
    Assumes RAL columns are numeric.

    Args:
        df: Input DataFrame with numeric RAL columns.

    Returns:
        A pandas Series with salary fit scores.
    """
    # Score based on whether expected RAL is within the min/max range.
    def _calculate_score(expected_ral, min_ral, max_ral):
        # pd.isna handles both np.nan and None
        if pd.isna(expected_ral) or pd.isna(min_ral) or pd.isna(max_ral):
            return np.nan

        # if min_ral > max_ral: # Handle illogical ranges
        #      return np.nan

        if expected_ral >= min_ral and expected_ral <= max_ral:
            return 1.0 # Perfect fit

        # If outside the range, score decreases with distance from the range
        if expected_ral < min_ral:
            distance = min_ral - expected_ral
        else: # expected_ral > max_ral
            distance = expected_ral - max_ral

        # Scale the distance (adjust denominator based on expected RAL scale)
        # Using the range size + min_ral as a scaling factor
        range_size = max_ral - min_ral
        scale_factor = range_size if range_size > 0 else min_ral # Avoid zero/negative division

        # If min_ral is also 0 or negative, use a default scale factor
        if scale_factor <= 0: scale_factor = 1000 # Default scale if range/min is non-positive

        # Simple scaled inverse distance for scores > 0
        score = 1 / (distance / scale_factor + 1)

        return score

    # Use .get for safety in case columns are missing
    return df.apply(
        lambda row: _calculate_score(row.get('Expected Ral'), row.get('Minimum Ral'), row.get('Ral Maximum')),
        axis=1
    )


def prepare_nlp_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates combined text columns for candidate profile and job requirements.

    Args:
        df: Input DataFrame. Assumes parsed experience columns exist if needed in text.

    Returns:
        DataFrame with 'candidate_text' and 'job_text' columns added.
    """
    df_processed = df.copy() # Work on a copy

    # Function to create a text summary for the candidate
    def create_candidate_text(row):
        # Combine relevant candidate features into a descriptive string
        parts = []
        # Include columns relevant to candidate's background/profile
        # Corrected 'Study area' to 'Study Area' based on user's column list
        candidate_cols = ['Study Area', 'Sector', 'Last Role']
        for col in candidate_cols:
            if pd.notna(row.get(col)): parts.append(f"{str(col).replace('_', ' ')}: {str(row[col])}")
        # Add parsed experience if it provides useful context as text
        if pd.notna(row.get('Years Experience_parsed')): parts.append(f"Experience: {row['Years Experience_parsed']} years")
        return ". ".join(parts) if parts else ""

    # Function to combine relevant job text
    def create_job_text(row):
        parts = []
         # Include columns relevant to job requirements
        job_cols = ['Job Title Hiring', 'Job Description', 'Candidate Profile', 'Study Area.1']
        for col in job_cols:
             if pd.notna(row.get(col)): parts.append(f"{str(col).replace('_', ' ')}: {str(row[col])}")
        # Add parsed required experience as text
        if pd.notna(row.get('Years Experience.1_parsed')): parts.append(f"Required Experience: {row['Years Experience.1_parsed']} years")
        return ". ".join(parts) if parts else ""

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


def calculate_geo_features(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
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


def calculate_overall_score(df: pd.DataFrame, score_columns: list[str]) -> pd.Series:
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


def drop_columns_except(df: pd.DataFrame, columns_to_keep: list[str]) -> pd.DataFrame:
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
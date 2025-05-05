def clean_dataframe_columns(df):
    """
    Clean and sanitize the column names of a pandas DataFrame.

    This function performs the following operations:
    1. Strips leading/trailing whitespace from each column name.
    2. Removes any non-ASCII characters from column names.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame whose columns need to be cleaned.

    Returns:
    -------
    pandas.DataFrame
        The same DataFrame with cleaned column names.
    """
    df.columns = [c.strip() for c in df.columns]
    df.columns = [c.encode('ascii', 'ignore').decode().strip() for c in df.columns]
    return df

def split_duplicate_ids_by_invariant_columns(df, invariant_columns=None):
    """
    Split duplicate IDs in a DataFrame when invariant columns vary within the same ID group.

    This function ensures that each group of rows sharing an ID also shares identical values 
    in the specified invariant columns. If not, the ID is suffixed to distinguish subgroups.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing duplicate IDs.
    
    invariant_columns : list of str
        Columns that should have the same values across all rows for a given ID. If differences
        are found, new IDs will be generated to separate subgroups.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with updated IDs and no unintended merges across differing invariant data.
    """
    if invariant_columns is None:
        raise ValueError("You must provide a list of invariant_columns.")

    unique_ids_before = df['ID'].nunique()

    df['Original_ID'] = df['ID']

    def split_id(group):
        unique_combos = group[invariant_columns].drop_duplicates()

        if len(unique_combos) == 1:
            group['New_ID'] = group['Original_ID'].iloc[0]
        else:
            for idx, (_, subgroup) in enumerate(group.groupby(invariant_columns, dropna=False)):
                group.loc[subgroup.index, 'New_ID'] = f"{group['Original_ID'].iloc[0]}_{idx + 1}"
        return group

    df = df.groupby('Original_ID', group_keys=False).apply(split_id)
    df['ID'] = df['New_ID'].fillna(df['Original_ID'])

    df = df.drop(columns=['New_ID', 'Original_ID']).reset_index(drop=True)

    unique_ids_after = df['ID'].nunique()
    print(f"🔵 Unique IDs before cleaning: {unique_ids_before}")
    print(f"🟢 Unique IDs after cleaning: {unique_ids_after}")
    print(f"🧮 Difference: {unique_ids_after - unique_ids_before} new IDs created")

    return df


def remove_initial_stage_candidates(df):
    """
    Removes candidates who are only present in the earliest stages of the selection process
    and have no sector information. These are typically low-signal entries that did not
    progress in the recruitment pipeline.

    A candidate row will be removed if:
    - It is the only row for that candidate ID.
    - The 'Candidate State' is one of ['imported', 'first contact', 'in selection'].
    - The 'Sector' is missing (NaN).

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing candidate records.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with early-stage, low-information candidates removed.
    """
    df['Candidate State'] = df['Candidate State'].str.strip().str.lower()

    ids_to_drop = []
    for id_value, group in df.groupby('ID'):
        is_single_row = len(group) == 1
        is_initial_stage_only = group['Candidate State'].isin(['imported', 'first contact', 'in selection']).all()
        has_no_sector_info = group['Sector'].isna().all()

        if is_single_row and is_initial_stage_only and has_no_sector_info:
            ids_to_drop.append(id_value)

    df_cleaned = df[~df['ID'].isin(ids_to_drop)].reset_index(drop=True)
    print(f"🗂️ Removed {len(ids_to_drop)} initial-stage only candidates.")
    return df_cleaned
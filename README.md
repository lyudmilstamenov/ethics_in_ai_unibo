# Predictive Modeling on Real Hiring Data: Exploring Salary Stereotypes and Bias Mitigation

## Overview

This project analyzes real hiring data to explore salary stereotypes and apply bias mitigation strategies. It involves thorough data preprocessing, advanced feature engineering, predictive modeling, and fairness evaluation using multiple mitigation techniques.

## Key Components

### Main Notebook (`src/main_notebook.ipynb`)
This notebook is the core of the project and includes the following steps:

- **Data Loading & Cleaning**
  - Load the dataset
  - Drop rows and duplicates
  - Clean and normalize column names
  - Remove irrelevant or invalid entries (e.g., early-stage candidates, inconsistent outcomes, invalid job titles)
  - Drop NaN values and finalize the cleaned dataset

- **ID and Structure Handling**
  - Generate new unique candidate IDs
  - Separate different individuals sharing duplicate IDs

- **Feature Engineering**
  - Extract and preprocess features such as `number_of_searches`
  - Map categorical columns (raw and overall mappings)
  - Create boolean protected attribute indicators
  - Extract residence information
  - Aggregate records 
  - Save and reload the preprocessed dataset for further use

- **Custom Similarity Features**
  - Generate similarity-based features using:
    - TF-IDF
    - Encoder-based similarity
    - Cross-encoder similarity

- **Dataset Analysis & Modeling**
  - Analyze dataset composition and feature distributions
  - Train various machine learning models
  - Compare models and evaluate different feature subsets
  - Apply and compare upsampling and downsampling strategies
  - Assess the impact of removing sensitive attributes on fairness

- **Evaluation**
  - Use performance and fairness metrics to assess models
  - Compare the effect of resampling and attribute masking


### Inprocessing Bias Mitigation (`src/inprocessing_mitigation_techniques.ipynb`)
- Load data
- Apply mitigation techniques:
  - **Fauci**
  - **Prejudice Remover**

### Preprocessing Bias Mitigation (`src/preprocessing_mitigation_techniques.ipynb`)
- Load data
- Apply mitigation techniques:
  - **Disparate Impact Remover**
  - **Reweighing**

## Utilities

Utilities used throughout the project are organized into three categories:

### Feature Engineering Utilities
- `calculate_distance()`
- `calculate_experience_match_score()`
- `calculate_professional_similarity_score()`
- `calculate_salary_fit_score()`
- `calculate_study_area_score()`
- `calculate_study_title_score()`
- `create_candidate_text()`
- `create_job_text()`
- `prepare_nlp_text_columns()`

### Data Cleaning Utilities
- `clean_dataframe_columns()`
- `remove_initial_stage_candidates()`
- `remove_not_hired_valid_candidates()`
- `sort_group()`
- `split_duplicate_ids_by_invariant_columns()`

### Plotting Utilities
- `get_mean_std()`
- `plot_metrics()`
- `plot_metrics_grouped()`
- `print_fairness_results_table()`

## Highlights of the Work

- Extensive **data cleaning and preprocessing** to enhance quality.
- **Text similarity feature engineering** using:
  - Encoder-based methods
  - TF-IDF similarity
  - Cross-encoder similarity
- Evaluation of a wide range of **machine learning models**.
- Application of **resampling techniques** (upsampling/downsampling) to balance data.
- **Feature selection and reduction** to prevent overfitting.
- Analysis of the effect of removing sensitive attributes on fairness.
- Testing of **four bias mitigation techniques**.
- In-depth evaluation using multiple **performance and fairness metrics**.

## How to Run

To execute the notebooks, make sure you have the necessary dependencies and that the `FairLib` repository is available as described below.

### 1. Clone the FairLib Repository
Place it in the parent directory of this project:
```bash
cd ..
git clone git@github.com:pikalab-unibo-students/master-thesis-dizio-ay2324.git
```

### 2. Install Dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Run the Notebooks

All notebooks are located in the `src/` directory. Open and run them sequentially:

- `src/main_notebook.ipynb`
- `src/inprocessing_mitigation_techniques.ipynb`
- `src/preprocessing_mitigation_techniques.ipynb`

## Documentation

Full project documentation is available here:  
> [https://ethics-in-ai-unibo.readthedocs.io/en/latest/index.html#](https://ethics-in-ai-unibo.readthedocs.io/en/latest/index.html#)

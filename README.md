# Automation Risk Prediction Using Occupational Skill Profiles

## Project overview
This project predicts occupational automation risk using skill profiles from O*NET and automation probabilities from Frey & Osborne (2013). The workflow includes data preparation, data merging, clustering, model training, model evaluation, and visualization.

## Project workflow
1. Skills pivot
   - The raw O*NET skills data is transformed into one row per occupation.
   - Each occupation receives 35 skill importance scores.
   - This step has already been completed, so the workflow can also start directly from `Skills_clean.csv`.

2. Data merge
   - The cleaned skills dataset is merged with:
     - `FO2013_data.csv`
     - `SOC10_SOC18_crosswalk.csv`
     - `ONET19_SOC18_crosswalk.csv`
   - The result is saved as `merged_dataset_SOC2018.csv`.

3. Clustering
   - Clustering is performed on `merged_dataset_SOC2018.csv`.
   - The output is saved as `final_clustered.csv`.

4. Model pipeline
   - The model pipeline is run on `final_clustered.csv`.
   - The full pipeline performs train/test splitting, hyperparameter tuning, model training, and evaluation.
   - To save time, the preloaded model pipeline can be used. This loads the trained models and avoids rerunning the grid search.

5. Visualizations
   - Final figures are generated using the visualization script.

## How to run the project

### Full workflow

```bash
python Skills_pivot.py
python data_merge.py
python clustering.py
python model_pipeline.py
python visualizations.py

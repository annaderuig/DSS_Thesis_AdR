# Automation Risk Prediction Using Occupational Skill Profiles

## Project Background and Data Sources
This project was executed for the Master’s Thesis of Anna de Ruig, in partial fulfillment of the requirements for the degree of Master of Science in Data Science & Society at the Tilburg School of Humanities and Digital Sciences (TSHD), Tilburg University.

The original owners of the data retain ownership of the data during and after completion of this project.

The datasets used in this project originate from the following sources:

1. `Skills.txt`  
   National Center for O*NET Development. (2025, August). *O*NET® 30.0 Database*.  
   https://www.onetcenter.org/database.html

2. `FO2013_data.csv`  
   Frey, C. B., & Osborne, M. (2013). *The Future of Employment*.

3. `ONET19_SOC18_crosswalk.csv`  
   O*NET Research Center. (n.d.). *O*NET-SOC 2019 Occupations to 2018 SOC Occupations*.  
   https://www.onetcenter.org/crosswalks.html#soc

4. `SOC10_SOC18_crosswalk.csv`  
   U.S. Bureau of Labor Statistics. (2020). *Crosswalk from the 2010 SOC to the 2018 SOC*.  
   https://www.bls.gov/soc/2018/crosswalks_used_by_agencies.htm


## Project overview
This project predicts occupational automation risk using skill profiles from O*NET and automation probabilities from Frey & Osborne (2013). The workflow includes data preparation, data merging, clustering, model training and model evaluation.

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

## How to run the project

### Full workflow

```bash
python Skills_pivot.py
python data_merge.py
python clustering.py
python model_pipeline.py

import pandas as pd

# Load data
fo2013 = pd.read_csv('FO2013_data.csv')
skills_clean = pd.read_csv('Skills_clean.csv')
soc10_soc18 = pd.read_csv('SOC10_SOC18_crosswalk.csv', sep=';')
onet19_soc18 = pd.read_csv('ONET19_SOC18_crosswalk.csv')

print("FO2013 data rows:", len(fo2013))
print("Skills clean rows:", len(skills_clean))
print("SOC10->SOC18 crosswalk rows:", len(soc10_soc18))
print("O*NET19->SOC18 crosswalk rows:", len(onet19_soc18))

# Clean the SOC10 column names
soc10_soc18_clean = soc10_soc18.dropna()
soc10_soc18_clean.columns = ['SOC10_Code', 'SOC10_Title', 'SOC18_Code', 'SOC18_Title']
soc10_soc18_clean = soc10_soc18_clean.drop_duplicates()

print("Nr of unique mappings in crosswalk:", len(soc10_soc18_clean))

# Rename FO2013 SOC code column for matching
fo2013_renamed = fo2013.rename(columns={'SOCcode': 'SOC10_Code'})

# Merge FO2013 with SOC18 codes (keep all SOC18 matches per SOC10)
fo2013_soc18 = fo2013_renamed.merge(
    soc10_soc18_clean[['SOC10_Code', 'SOC18_Code', 'SOC18_Title']],
    on='SOC10_Code',
    how='left'
)

# Check for unmatched SOC10 codes
unmatched_fo2013 = fo2013_soc18[fo2013_soc18['SOC18_Code'].isna()]
unmatched_fo2013_detail = unmatched_fo2013[['SOC10_Code', 'Occupation', 'Probability']].drop_duplicates()
matched_count = len(fo2013_soc18) - len(unmatched_fo2013)
print("FO2013 matched:", matched_count, "out of", len(fo2013_soc18))

# Rename O*NET skills column for matching
skills_renamed = skills_clean.rename(columns={'O*NET-SOC Code': 'ONET_Code'})

# Create base code mapping from crosswalk (first 7 chars, e.g. "11-1011" from "11-1011.00")
onet19_soc18['ONET_Base'] = onet19_soc18['O*NET-SOC 2019 Code'].str[:7]

# Keep all base code -> SOC18 mappings (will aggregate on SOC18_Code later)
onet_base_soc18 = onet19_soc18[['ONET_Base', 'O*NET-SOC 2019 Code', '2018 SOC Code', '2018 SOC Title']]
onet_base_soc18.columns = ['ONET_Base', 'ONET_Full_Code', 'SOC18_Code', 'SOC18_Title']

# Extract base code from skills data
skills_renamed['ONET_Base'] = skills_renamed['ONET_Code'].str[:7]

# Merge on base code
skills_soc18 = skills_renamed.merge(
    onet_base_soc18[['ONET_Base', 'SOC18_Code', 'SOC18_Title']],
    on='ONET_Base',
    how='left'
)

# Check for unmatched O*NET codes
unmatched_skills = skills_soc18[skills_soc18['SOC18_Code'].isna()]
matched_count = len(skills_soc18) - len(unmatched_skills)
print("Matched Skills:", matched_count,  "out of", len(skills_soc18))

# Remove rows with missing SOC18 codes
fo2013_clean = fo2013_soc18.dropna(subset=['SOC18_Code'])
skills_clean_mapped = skills_soc18.dropna(subset=['SOC18_Code'])

print("FO2013 total matched:", len(fo2013_clean))
print("Skills total matched:", len(skills_clean_mapped))

# Aggregate FO2013 by SOC18 (take mean probability if multiple SOC10 codes map to same SOC18)
fo2013_agg = fo2013_clean.groupby('SOC18_Code').agg({
    'SOC18_Title': 'first',
    'Probability': 'mean'
}).reset_index()
fo2013_agg = fo2013_agg.round(2)

print("FO2013 after aggregation:", len(fo2013_agg))

# Get skill columns (without ID/mapping columns)
skill_columns = [col for col in skills_clean_mapped.columns 
                 if col not in ['ONET_Code', 'ONET_Base', 'ONET_Title', 'SOC18_Code', 'SOC18_Title']]

print("Number of skill columns:", len(skill_columns))

# Aggregate skills by SOC18 (take mean importance scores if multiple O*NET codes map to same SOC18)
skills_agg = skills_clean_mapped.groupby('SOC18_Code').agg({
    'SOC18_Title': 'first',
    **{col: 'mean' for col in skill_columns}
}).reset_index()
skills_agg = skills_agg.round(2)

print("Skills aggregated:", len(skills_agg))

# Merge FO2013 with skills on SOC18_Code and SOC18_Title
final_merged = fo2013_agg.merge(
    skills_agg,
    on=['SOC18_Code', 'SOC18_Title'],
    how='inner'
)

# Identify missing codes
fo2013_only = fo2013_agg[~fo2013_agg['SOC18_Code'].isin(skills_agg['SOC18_Code'])].copy()
skills_only = skills_agg[~skills_agg['SOC18_Code'].isin(fo2013_agg['SOC18_Code'])].copy()

print("\nFinal dataset:",
      "\nRows:", len(final_merged), 
      "\nColumns:", len(final_merged.columns))
print("\nMissing data:",
      "\nFO2013 only (no skills):", len(fo2013_only),
        "\nSkills only (no FO2013):", len(skills_only))

# Reorder columns: SOC18_Code, SOC18_Title, Probability, skills
final_columns = ['SOC18_Code', 'SOC18_Title', 'Probability'] + skill_columns
final_merged = final_merged[final_columns]

# Rename columns for clarity
final_merged = final_merged.rename(columns={
    'SOC18_Code': 'SOC2018_Code',
    'SOC18_Title': 'SOC2018_Occupation_Title',
    'Probability': 'Automation_Probability'
})

print("\nFirst rows final dataset:")
print(final_merged.head())

# Save to CSV
final_merged.to_csv('merged_dataset_SOC2018.csv', index=False)
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

#### ALLEEN VOOR EXCEL AI CREATIE => Create detailed aggregation log for documentation
fo2013_agg_overview = fo2013_clean.groupby('SOC18_Code').apply(lambda x: pd.DataFrame({
    'SOC18_Code': [x.name],
    'SOC18_Title': [x['SOC18_Title'].iloc[0]],
    'Count_SOC10_Codes': [len(x)],
    'SOC10_Codes': ['; '.join(x['SOC10_Code'].unique())],
    'Occupations': ['; '.join(x['Occupation'].unique())],
    'Individual_Probabilities': ['; '.join([f"{p:.6f}" for p in x['Probability']])],
    'Average_Probability': [x['Probability'].mean()]
}), include_groups=False).reset_index(drop=True)

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

#### ALLEEN VOOR EXCEL AI CREATIE => Create detailed aggregation log for skills documentation
skills_agg_overview = skills_clean_mapped.groupby('SOC18_Code').apply(lambda x: pd.DataFrame({
    'SOC18_Code': [x.name],
    'SOC18_Title': [x['SOC18_Title'].iloc[0]],
    'Count_ONET_Codes': [len(x)],
    'ONET_Codes': ['; '.join(x['ONET_Code'].unique())],
    'ONET_Base_Codes': ['; '.join(x['ONET_Base'].unique())]
}), include_groups=False).reset_index(drop=True)

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

# Save to Excel
final_merged.to_excel('merged_dataset_SOC2018.xlsx', index=False, sheet_name='Data')

##### ALLEEN VOOR EXCEL AI CREATIE => Create Excel with mapping details
with pd.ExcelWriter('mapping_documentation.xlsx', engine='openpyxl') as writer:
    
    # Sheet 1: SOC10 -> SOC18 Mapping
    soc10_mapping = fo2013_renamed.merge(
        soc10_soc18_clean[['SOC10_Code', 'SOC18_Code', 'SOC18_Title']],
        on='SOC10_Code',
        how='left'
    )[['SOC10_Code', 'Occupation', 'SOC18_Code', 'SOC18_Title']].drop_duplicates()
    soc10_mapping.to_excel(writer, sheet_name='SOC10_to_SOC18', index=False)
    
    # Sheet 1b: SOC10 -> SOC18 Aggregation Details
    fo2013_agg_overview.to_excel(writer, sheet_name='FO2013_Aggregation_Details', index=False)
    
    # Sheet 1c: Unmatched SOC10 codes
    unmatched_fo2013_detail.to_excel(writer, sheet_name='Unmatched_SOC10_Codes', index=False)
    
    # Sheet 2: O*NET19 -> SOC18 Mapping
    onet_mapping = skills_renamed[['ONET_Code', 'ONET_Base']].merge(
        onet_base_soc18[['ONET_Base', 'SOC18_Code', 'SOC18_Title']],
        on='ONET_Base',
        how='left'
    )[['ONET_Code', 'ONET_Base', 'SOC18_Code', 'SOC18_Title']].drop_duplicates()
    onet_mapping.to_excel(writer, sheet_name='ONET19_to_SOC18', index=False)
    
    # Sheet 2b: Skills Aggregation Details
    skills_agg_overview.to_excel(writer, sheet_name='Skills_Aggregation_Details', index=False)
    
    # Sheet 2c: Missing Data - FO2013 codes without Skills data
    if len(fo2013_only) > 0:
        fo2013_only_detail = fo2013_only[['SOC18_Code', 'SOC18_Title', 'Probability']].copy()
        fo2013_only_detail.to_excel(writer, sheet_name='FO2013_Missing_Skills', index=False)
    
    # Sheet 2d: Missing Data - Skills codes without FO2013 data
    if len(skills_only) > 0:
        skills_only_detail = skills_only[['SOC18_Code', 'SOC18_Title']].copy()
        skills_only_detail.to_excel(writer, sheet_name='Skills_Missing_FO2013', index=False)

    # Sheet 3: Data Coverage Summary
    summary_data = pd.DataFrame({
        'Metric': [
            'Original FO2013 records',
            'FO2013 records matched to SOC18',
            'FO2013 records NOT matched to SOC18',
            'Unique SOC10 codes not matched',
            'Unique SOC18 codes from FO2013',
            'SOC18 codes with aggregated FO2013 data',
            'Original Skills records',
            'Skills records matched to SOC18',
            'Unique SOC18 codes from Skills',
            'SOC18 codes with aggregated Skills data',
            'SOC18 codes in FO2013 only (no Skills)',
            'SOC18 codes in Skills only (no FO2013)',
            'Final merged records (both datasets)',
            'Total columns in final dataset',
            'Skill columns',
            'Date created'
        ],
        'Value': [
            len(fo2013),
            len(fo2013_clean),
            len(unmatched_fo2013),
            len(unmatched_fo2013_detail),
            len(fo2013_agg),
            (fo2013_agg_overview['Count_SOC10_Codes'] > 1).sum(),
            len(skills_clean),
            len(skills_clean_mapped),
            len(skills_agg),
            (skills_agg_overview['Count_ONET_Codes'] > 1).sum(),
            len(fo2013_only),
            len(skills_only),
            len(final_merged),
            len(final_merged.columns),
            len(skill_columns),
            pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
    })
    summary_data.to_excel(writer, sheet_name='Summary', index=False)


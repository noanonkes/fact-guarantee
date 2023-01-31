import numpy as np
import pandas as pd

#############################################
# Initial clean-up of the data              #
# Includes:                                 #
#   - removing irrelevant columns and rows  #
#   - filtering to 2 races                  #
#   - converting categorical to numerical   #
#############################################

df = pd.read_csv('diabetic_data.csv')

# Removed since not relevant
df.drop(['encounter_id', 'patient_nbr', 'payer_code'], axis=1, inplace=True)

# Removed since it contained only 1 value
df.drop(['examide', 'citoglipton', 'acetohexamide', 'metformin-rosiglitazone'], axis=1, inplace=True)

# Remove encounters which don't have information logged for these columns
# + obtain numerical values for each option in the 'diag.' columns
for column in ['diag_1', 'diag_2', 'diag_3']:
    df = df.loc[df[column] != '?']
    df[column] = df[column].replace(regex={r'E': '3.', r'V': '5.'})

# Filter down to 2 races
df = df.loc[df['race'].isin(['AfricanAmerican', 'Caucasian']) == True]

# Simplify things
df = df.replace({
    'No': 0,
    'NO': 0,
    'yes': 1,
    'Yes': 1,

    # Re-admission within 30 days or after 30 day
    # is still re-admisson
    '>30': 1,
    '<30': 1,

    # Change occured ('Ch) or not ('No')
    'Ch': 1,

    'Steady': 1,
    'Down': -1,
    'Up': 2,

    # They use these terms in the original experiments
    'AfricanAmerican': 'Black',
    'Caucasian': 'White'
})

numerical_fields = []
categorical_fields = []

# Numerical columns get ..
for k in df:
    try:
        df[k] = df[k].astype(float)
        max = df[k].max()
        df[k] = (df[k]/max).round(9)
        numerical_fields.append(k)
    except:
        categorical_fields.append(k)

# While categorical columns get turned into a binary column 
# for each possible categorical value it can take on
for k in categorical_fields:
    for v in df[k].unique():
        df[f'{k}:is_{v}'] = np.where(df[k] == v, 0, 1)
    # delete original categorical column
    df.drop(k, axis=1, inplace=True)

print('Number of samples: ', len(df))

df.to_csv('diabetes_processed.csv', index=False)
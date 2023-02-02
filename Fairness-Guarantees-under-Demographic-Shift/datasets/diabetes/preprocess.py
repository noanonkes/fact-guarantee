import numpy as np
import pandas as pd

def prep(dshift_var, df):

    # Removed since not relevant
    df.drop(['encounter_id', 'patient_nbr', 'payer_code'], axis=1, inplace=True)

    # Removed since it contained only 1 value
    df.drop(['examide', 'citoglipton', 'acetohexamide', 'metformin-rosiglitazone', 'glimepiride-pioglitazone'], axis=1, inplace=True)

    # Remove encounters which don't have information logged for these columns
    # + obtain numerical values for each option in the 'diag.' columns
    for column in ['diag_1', 'diag_2', 'diag_3']:
        df = df.loc[df[column] != '?']
        df[column] = df[column].replace(regex={r'E': '3.', r'V': '5.'})

    df = df.loc[df['medical_specialty'] != '?']

    if dshift_var == 'sex':
        # Filter down to 2 races
        df = df.loc[df['race'].isin(['AfricanAmerican', 'Caucasian']) == True]
        df.replace({        
                    'AfricanAmerican': 'Black',
                    'Caucasian': 'White'
                    }, inplace=True)
    elif dshift_var == 'race':
        df = df.loc[df['race'] != '?']
        df = df.loc[df['gender'] != 'Unknown/Invalid']
        df.replace({       
                    'Hispanic' : 1, 
                    'Caucasian': 2, 
                    'AfricanAmerican' : 3, 
                    'Asian' : 4, 
                    'Other' : 5
                    }, inplace=True)

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

        'Speech' : 1,
        'InternalMedicine' : 2,
        'Family/GeneralPractice' : 3,
        'Cardiology' : 4,
        'Surgery-General' : 5,
        'Orthopedics' : 6,
        'Gastroenterology' : 7,
        'Surgery-Cardiovascular/Thoracic' : 8,
        'Nephrology' : 9,
        'Orthopedics-Reconstructive' : 10,
        'Psychiatry' : 11,
        'Emergency/Trauma' : 12,
        'Pulmonology' : 13,
        'Surgery-Neuro' : 14,
        'Obsterics&Gynecology-GynecologicOnco' : 15,
        'ObstetricsandGynecology' : 16,
        'Pediatrics' : 17,
        'Otolaryngology' : 18,
        'Pediatrics-Endocrinology' : 19,
        'Endocrinology' : 20,
        'Hematology/Oncology' : 21,
        'Urology' : 22,
        'Pediatrics-CriticalCare' : 23,
        'Psychiatry-Child/Adolescent' : 24,
        'Pediatrics-Pulmonology' : 25,
        'Surgery-Colon&Rectal' : 26,
        'Neurology' : 27,
        'Anesthesiology-Pediatric' : 28,
        'Radiology' : 29,
        'Pediatrics-Hematology-Oncology' : 30,
        'Podiatry' : 31,
        'Gynecology' : 32,
        'Oncology' : 33,
        'Pediatrics-Neurology' : 34,
        'Surgery-Plastic' : 35,
        'Surgery-Thoracic' : 36,
        'Surgery-PlasticwithinHeadandNeck' : 37,
        'Psychology' : 38,
        'Ophthalmology' : 39,
        'Surgery-Pediatric' : 40,
        'PhysicalMedicineandRehabilitation' : 41,
        'InfectiousDiseases' : 42,
        'Anesthesiology' : 43,
        'Pediatrics-EmergencyMedicine' : 44,
        'Rheumatology' : 45,
        'AllergyandImmunology' : 46,
        'Surgery-Maxillofacial' : 47,
        'Pediatrics-InfectiousDiseases' : 48,
        'Pediatrics-AllergyandImmunology' : 49,
        'Dentistry' : 50,
        'Surgeon' : 51,
        'Surgery-Vascular' : 52,
        'Osteopath' : 53,
        'Psychiatry-Addictive' : 54,
        'Surgery-Cardiovascular' : 55,
        'PhysicianNotFound' : 56,
        'Hematology' : 57,
        'Proctology' : 58,
        'Obstetrics' : 59,
        'SurgicalSpecialty' : 60,
        'Radiologist' : 61,
        'Pathology' : 62,
        'Dermatology' : 63,
        'SportsMedicine' : 64,
        'Hospitalist' : 65,
        'OutreachServices' : 66,
        'Cardiology-Pediatric' : 67,
        'Perinatology' : 68,
        'Neurophysiology' : 69,
        'Endocrinology-Metabolism' : 70,
        'DCPTEAM' : 71,
        'Resident' : 72,

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
        v = df[k].unique()
        if len(v) == 2 and k != 'race':
            df[f'{k}:is_{v[0]}'] = np.where(df[k] == v[0], 0, 1)
            df.drop(k, axis=1, inplace=True)
        else:
            for v_ in v:
                df[f'{k}:is_{v_}'] = np.where(df[k] == v_, 0, 1)
                # delete original categorical column
            df.drop(k, axis=1, inplace=True)

    print('Number of samples: ', len(df))

    if dshift_var == 'sex':
        df.to_csv('diabetes_processed_sex.csv', index=False)
    elif dshift_var == 'race':
        df.to_csv('diabetes_processed_race.csv', index=False)


df = pd.read_csv('diabetic_data.csv')
prep('sex', df)
df = pd.read_csv('diabetic_data.csv')
prep('race', df)
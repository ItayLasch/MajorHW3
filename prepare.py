import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib import pylab
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
def _modify_features(data):
    # Add current location X and Y
    x_loc = data['current_location'].apply(lambda loc: float(loc.split(',')[0][2:-1]))
    y_loc = data['current_location'].apply(lambda loc: float(loc.split(',')[1][2:-2]))
    data['current_x'] = x_loc
    data['current_y'] = y_loc

    # Add blood type
    data['Blood_A'] = data['blood_type'].isin(["A+", "A-"])
    data['Blood_B'] = data['blood_type'].isin(["AB+", "AB-", "B+", "B-"])
    data['Blood_C'] = data['blood_type'].isin(["O+", "O-"])

    # Add symptoms
    symptoms = data["symptoms"].unique()
    sym_uniq = set()
    for symp in symptoms:
        sym_uniq.update(str(symp).split(';'))
        sym_uniq.discard('nan')

    for sym in sym_uniq:
        data[sym] = [str(sym_str).__contains__(sym) for sym_str in data["symptoms"]]

    # Transform gender
    data["sex"] = data["sex"] == "M" 

    # Transfrom date
    data["day"] = [int(str(date).split('-')[2]) for date in data["pcr_date"]]
    data["month"] = [int(str(date).split('-')[1]) for date in data["pcr_date"]]
    data["year"] = [int(str(date).split('-')[0]) for date in data["pcr_date"]]


    # drop coloumns we don't need anymore
    data.drop(["patient_id","symptoms","blood_type","current_location","pcr_date"],axis=1, inplace=True)

    # Replace True/False with 1/0
    data.replace({True:1, False:0}, inplace=True)

def prepare_data(training_data, new_data):

    copy_data = new_data.copy()
    copy_train_data = training_data.copy()

    # Add/Remove features
    _modify_features(copy_data)
    _modify_features(copy_train_data)

    min_max_list = ["age","num_of_siblings","happiness_score","household_income","conversations_per_day","sport_activity","PCR_01","PCR_02","PCR_03","PCR_07","PCR_08","PCR_09","current_x","current_y", "day", "month", "year"]
    standard_list = ["weight","sugar_levels","PCR_04","PCR_05","PCR_06","PCR_10"]

    #  Init Scalers
    standard_scaler = StandardScaler()
    standard_scaler.fit(copy_train_data[standard_list])
    copy_data[standard_list] = standard_scaler.transform(copy_data[standard_list])

    min_max_scaler = MinMaxScaler((-1,1))
    min_max_scaler.fit(copy_train_data[min_max_list])
    copy_data[min_max_list] = min_max_scaler.transform(copy_data[min_max_list])

    return copy_data


def main():
    filename = 'virus_data.csv'
    random_state = 2 + 12

    dataset = pd.read_csv(filename)
    train_df, test_df = train_test_split(dataset, test_size=0.2, train_size=0.8, random_state=random_state)
    train_df_prepared = prepare_data(train_df, train_df)
    test_df_prepared = prepare_data(train_df, test_df)

    test_df_prepared.to_csv('train_prepared.csv')
    test_df_prepared.to_csv('test_prepared.csv')

if __name__ == "__main__":
    main()
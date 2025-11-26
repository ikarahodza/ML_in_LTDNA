import os
import warnings
import joblib
import argparse
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
replace_dict = {
    'Allele': 0,
    'Back_Stutter': 1,
    'Forward_Stutter': 2,
    'Pull_Up': 3
}

replace_label_dict = {
    0: 'Allele',
    1: 'Back_Stutter',
    2: 'Forward_Stutter',
    3: 'Pull_Up'
}

train_model = {
        'M': joblib.load(r'./model/huber_model_M.pkl'),
        'S': joblib.load(r'./model/huber_model_S.pkl'),
        'S&M': joblib.load(r'./model/huber_model_S&M.pkl'),
    }

def float_to_label(preds):
    y_pred_rounded = np.round(preds)
    y_pred_clipped = np.clip(y_pred_rounded, 0, 3)
    y_pred = pd.DataFrame(y_pred_clipped, columns=['pred_y'])['pred_y'].replace(replace_label_dict)
    return y_pred

def train(train_data, label):
    stacking = HuberRegressor(epsilon=1.35, alpha=1e-5, max_iter=2000)

    train_data = np.array(train_data)
    label = np.array(label)
    train_data, label = over_sample(train_data, label)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data)
    stacking.fit(X_train_scaled, label)
    y_pred = stacking.predict(X_train_scaled)
    y_pred = float_to_label(y_pred)
    model_dict = {'huber': stacking, 'scaler': scaler}
    return model_dict, y_pred

def to_test(test_data, model):
    test_data = model['scaler'].transform(np.array(test_data))
    y_pred = model['huber'].predict(test_data)
    y_pred = float_to_label(y_pred)
    return y_pred

def over_sample(X_train, X_test):
    oversample = SMOTE(random_state=2024)
    X_train, X_test = oversample.fit_resample(X_train, X_test)
    return X_train, X_test

def data_processing(current_directory=None, mode="test"):
    # Get current directory
    if current_directory == None:
        current_directory = os.getcwd()
        print("Load files in current directory")
    column_1 = 'Allele'
    column_2 = 'Back_Stutter'
    column_3 = 'Forward_Stutter'
    column_4 = 'Pull_Up'
    column_y = 'Actual_y'
    x_data = pd.DataFrame()
    # Iterate over all files in the current directory
    sorted_contents = sorted(os.listdir(current_directory), key=str.lower)
    for index, filename in enumerate(sorted_contents):
        if filename.endswith('.xlsx'):
            # Build file path
            file_path = os.path.join(current_directory, filename)
            # Read Excel file
            df = pd.read_excel(file_path, engine='openpyxl')
            # Get training data
            try:
                column_data = df[[column_1, column_2, column_3, column_4]]
            except Exception as e:
                # Code that handles all exceptions
                print(f"Allele/Back_Stutter/Forward_Stutter/Pull_Up line not found in the directory: {filename}")
            x_data = pd.concat([x_data, column_data], axis=1)
            if index == len(sorted_contents) - 1 and mode != "test":
                y_data = df[[column_y]]
                y_data = y_data[column_y].replace(replace_dict)
                return x_data, y_data
            # print(f"File: {filename}")
        else:
            print(f"xlsx not found")

    return x_data

def output_file(y_pred, input_path):
    column_to_insert_after = 'prediction'
    sorted_contents = sorted(os.listdir(input_path), key=str.lower)
    x_data = pd.DataFrame()
    for filename in sorted_contents:
        if filename.endswith('.xlsx'):
            file_path = os.path.join(input_path, filename)
            x_data = pd.read_excel(file_path, engine='openpyxl')
            break
    if column_to_insert_after in x_data.columns:
        insert_position = x_data.columns.get_loc(column_to_insert_after) + 1
    else:
        insert_position = len(x_data.columns)
    x_data.insert(loc=insert_position, column='f_pre_y', value=y_pred)

    columns_to_drop = ['Allele', 'Back_Stutter', 'Forward_Stutter', 'Pull_Up']
    existing_columns = [col for col in columns_to_drop if col in x_data.columns]
    x_data = x_data.drop(existing_columns, axis=1)
    return x_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a script.")
    parser.add_argument('--Model', type=str, default="M",
                        help="Enter the type of model(\"S\",\"M\" and \"S&M\" respectively represents model trained"
                             " using peak data from single-source profiles, mixed DNA profiles and a combination of "
                             "peak data from single-source profiles and mixed DNA profiles)")
    parser.add_argument('--Customizable', type=str, default="N",
                        help="Please choose whether to customize the model. The default is not to customize. Enter any character other than 'N' to enable customization.")
    parser.add_argument('--TrainInputPath', type=str, default="None", help="Enter the path of training data file") # 单一+混合图谱峰 混合样本图谱峰
    parser.add_argument('--TestInputPath', type=str, default="./", help="Enter the path of testing data file")
    parser.add_argument('--OutputPath', type=str, default="./", help="Enter the path of output file")

    args = parser.parse_args()

    if args.Model:
        print(f"The type of model: {args.Model}")
    else:
        print("Please enter the type of model.")

    if args.TrainInputPath != "None":
        this_mode = "train"
        x_data, y_data = data_processing(current_directory=args.TrainInputPath, mode=this_mode)
        model_dict, y_pred = train(x_data, y_data)
        output_data = output_file(y_pred, args.TrainInputPath)
        # output_data.to_excel(args.OutputPath + r"/Stacking_" + args.model + "_" + this_mode + ".xlsx", index=False)
        joblib.dump(model_dict, r'./model/new_huber_model_' + args.Model + '.pkl')
    else:
        this_mode = "test"
        x_data = data_processing(current_directory=args.TestInputPath, mode=this_mode)
        if args.Customizable != "N":
            model = joblib.load(r'./model/' + 'new_huber_model_' + args.Model + '.pkl')
        else:
            model = train_model[args.Model]
        y_pred = to_test(x_data, model).to_frame(name='f_pre_y')
        output_data = output_file(y_pred, args.TestInputPath)
        output_data.to_excel(args.OutputPath + r"/Stacking_" + args.Model + "_" + this_mode + ".xlsx", index=False)


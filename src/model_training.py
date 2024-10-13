import pickle
from extract_load_data import BigQuery_DataOps
from feature_engineering import feature_engineering_training
from model_functions import StackedModel
import warnings
from datetime import datetime
import json
warnings.filterwarnings('ignore')

current_date = datetime.now()
formatted_date = current_date.strftime("%Y%m%d")
pickle_path = f"artifacts\model_v{formatted_date}.pkl"

print("==============================")
print("Initializing model training process")
print(f"Training on: {formatted_date}")

print("==============================\n")

print("==============================")
print("Start: Extracting Data")
df = BigQuery_DataOps(sub_db = 'prd') \
    .extract_data(tb_id = 'prd_analytical_base_table')

print(f"Number of rows used to train the model: {df.shape[0]}")
print("Finish: Extracting Data")

print("==============================\n")

print("==============================")
print("Start: Feature Engineering Process")
feature_types = {"object_features": [col for col in df.columns if df[col].dtype == 'object'],
                 "numeric_features": [col for col in df.columns if df[col].dtype != 'object']}

with open("artifacts/feature_types.json", "w") as json_file:
            json.dump(feature_types, json_file)

df = feature_engineering_training(df)
print("Finish: Feature Engineering Process")

print("==============================\n")

print("==============================")
print("Start: Training")
model = StackedModel(df).train_model()
print("Finish: Training")

print("==============================\n")

with open(pickle_path, 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model Saved! Process ended successfully")
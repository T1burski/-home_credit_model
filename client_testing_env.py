# Import
import requests
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from src.extract_load_data import BigQuery_DataOps

df = BigQuery_DataOps(sub_db = 'prd') \
    .extract_data(tb_id = 'prd_analytical_base_table')

data = df.drop(columns=['TARGET']).sort_values(by=["SK_ID_CURR"], ascending=False).head(20)

payload = {"data": data.to_json(orient="records")}

response = requests.post("http://localhost:3000/predict", json = payload).json()

response_df = {"SK_ID_CURR": list(data["SK_ID_CURR"]),
               "default_probability": list(response["Probability"]),
               "default_occurrence": list(response["Classification"])}

BigQuery_DataOps(sub_db = 'prd') \
    .load_data(tb_data = pd.DataFrame(response_df),
               tb_name = 'api_predictions')


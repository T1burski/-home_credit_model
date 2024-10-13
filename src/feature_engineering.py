import numpy as np
import pandas as pd
import json


def feature_engineering_training(df_in):

    df_in.drop_duplicates(inplace=True)

    df_in["REGION_RATING_CLIENT"] = df_in["REGION_RATING_CLIENT"].astype(object)
    df_in["REGION_RATING_CLIENT_W_CITY"] = df_in["REGION_RATING_CLIENT_W_CITY"].astype(object)

    string_cols = [c for c in list(df_in.columns) if df_in[c].dtype=='object']

    for i in df_in.columns:
        if df_in[i].name in string_cols:
            df_in[i] = np.where(df_in[i].isnull(), 'No Data', df_in[i])
        else:
            df_in[i] = np.where(df_in[i].isnull(), 0, df_in[i])
    
    cat_cols_to_drop = ['NAME_INCOME_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'NAME_FAMILY_STATUS', 'SK_ID_CURR']
    
    if 'ORGANIZATION_TYPE' not in cat_cols_to_drop:
        organization_type_df = pd.crosstab(df_in['ORGANIZATION_TYPE'], df_in['TARGET'], normalize='index').reset_index()
        organization_type_df.columns = ['ORGANIZATION_TYPE', '0', '1']
        organization_type_list = organization_type_df.loc[organization_type_df["1"] > 0.1]['ORGANIZATION_TYPE'].tolist()

        organization_feature_json = {}
        organization_feature_json['organization_type_list'] = organization_type_list

        df_in['ORGANIZATION_TYPE'] = np.where(df_in['ORGANIZATION_TYPE'].isin(organization_type_list), 1, 0)

        with open("artifacts/organization_feature_json.json", "w") as json_file:
            json.dump(organization_feature_json, json_file)
    
    df_in['CODE_GENDER'] = np.where(df_in['CODE_GENDER'].isin(['M']), 1, 0)
    df_in['NAME_EDUCATION_TYPE'] = np.where(df_in['NAME_EDUCATION_TYPE'].isin(['Academic degree', 'Higher education']), 1, 0)
    df_in = df_in.drop(columns=cat_cols_to_drop)
    df_in.reset_index(drop=True, inplace=True)

    new_columns = {
    "CREDIT_INCOME": (1 + df_in["AMT_CREDIT"]) / (1 + df_in["AMT_INCOME_TOTAL"]),
    "ANNUITY_INCOME": (1 + df_in["AMT_ANNUITY"]) / (1 + df_in["AMT_INCOME_TOTAL"]),
    "CREDIT_GOODS_PRICE": (1 + df_in["AMT_CREDIT"]) / (1 + df_in["AMT_GOODS_PRICE"]),
    "GOODS_PRICE_INCOME": (1 + df_in["AMT_GOODS_PRICE"]) / (1 + df_in["AMT_INCOME_TOTAL"]),
    "EXTS_2_3_DIFF": df_in["EXT_SOURCE_2"] - df_in["EXT_SOURCE_3"],
    "EXTS_2_3_MEAN": (df_in["EXT_SOURCE_2"] + df_in["EXT_SOURCE_3"]) / 2,
    "EXTS_2_3_SUM": df_in["EXT_SOURCE_2"] + df_in["EXT_SOURCE_3"]
    }

    df_in = pd.concat([df_in, pd.DataFrame(new_columns)], axis=1)

    return df_in


def feature_engineering_production(df_in):

    df_in["REGION_RATING_CLIENT"] = df_in["REGION_RATING_CLIENT"].astype(object)
    df_in["REGION_RATING_CLIENT_W_CITY"] = df_in["REGION_RATING_CLIENT_W_CITY"].astype(object)
    
    string_cols = [c for c in list(df_in.columns) if df_in[c].dtype=='object']

    for i in df_in.columns:
        if df_in[i].name in string_cols:
            df_in[i] = np.where(df_in[i].isnull(), 'No Data', df_in[i])
        else:
            df_in[i] = np.where(df_in[i].isnull(), 0, df_in[i])
        
    cat_cols_to_drop = ['NAME_INCOME_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'NAME_FAMILY_STATUS', 'SK_ID_CURR']

    if 'ORGANIZATION_TYPE' not in cat_cols_to_drop:

        organization_feature_json = json.load(open('artifacts/organization_feature_json.json'))

        organization_type_list = list(organization_feature_json['organization_type_list'])

        df_in['ORGANIZATION_TYPE'] = np.where(df_in['ORGANIZATION_TYPE'].isin(organization_type_list), 1, 0)
    
    df_in['CODE_GENDER'] = np.where(df_in['CODE_GENDER'].isin(['M']), 1, 0)
    df_in['NAME_EDUCATION_TYPE'] = np.where(df_in['NAME_EDUCATION_TYPE'].isin(['Academic degree', 'Higher education']), 1, 0)
    df_in = df_in.drop(columns=cat_cols_to_drop)
    df_in.reset_index(drop=True, inplace=True)

    new_columns = {
    "CREDIT_INCOME": (1 + df_in["AMT_CREDIT"]) / (1 + df_in["AMT_INCOME_TOTAL"]),
    "ANNUITY_INCOME": (1 + df_in["AMT_ANNUITY"]) / (1 + df_in["AMT_INCOME_TOTAL"]),
    "CREDIT_GOODS_PRICE": (1 + df_in["AMT_CREDIT"]) / (1 + df_in["AMT_GOODS_PRICE"]),
    "GOODS_PRICE_INCOME": (1 + df_in["AMT_GOODS_PRICE"]) / (1 + df_in["AMT_INCOME_TOTAL"]),
    "EXTS_2_3_DIFF": df_in["EXT_SOURCE_2"] - df_in["EXT_SOURCE_3"],
    "EXTS_2_3_MEAN": (df_in["EXT_SOURCE_2"] + df_in["EXT_SOURCE_3"]) / 2,
    "EXTS_2_3_SUM": df_in["EXT_SOURCE_2"] + df_in["EXT_SOURCE_3"]
    }

    df_in = pd.concat([df_in, pd.DataFrame(new_columns)], axis=1)

    return df_in
    

        
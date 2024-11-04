# Home Credit Default Risk - Modeling and Deploying

![image](https://github.com/user-attachments/assets/8aba4dac-c4ab-499d-a5f3-06b0bceb2ade)

### 1) The challenge:
This project was developed in order to tackle the challenge proposed by Home Credit (https://www.kaggle.com/competitions/home-credit-default-risk/overview), which focuses on predicting the Client's Default Risk. Along with the model development, this project also builds a system that manages the various data used (using Google BigQuery and dbt core) and the whole processing pipeline that makes the deployment of the model built into a production env possible using modularization, Docker and the final deployment of the model in a API REST.

As a spoiler right now, the model developed here, when applied to unseen data selected during the exploration and modeling processes, had a potential financial impact of:

### 1,37 Billion USD

(assuming that the Credit Amount provided was given in USD), all this considering the Credit Amount of the Client's credits that had default status as positive and were predicted so.

The image above shows the whole data stack built in order to manage the data and deploy the model ready to apply predictions.


### 2) The Data:
The data provided was very complex, with multiple datasets and a fairly complex relational modeling behind. In this project, we did not use every original dataset for data storage reasons. The first data ingestion was made using PySpark and the rest of the modeling and architecture was done using dbt core. The repo containing these first ingestion pipelines and dbt core code base can be found in the github repo in the following link: https://github.com/T1burski/-home_credit_bigquery/tree/main .

The first ingestion can be found in the mentioned repo in scr/initial_ingestion.py, along with the dbt queries in dbt_bigquery. The resulting Lineage Graph within dbt can be seen below, which was generated with dbt docs:

![image](https://github.com/user-attachments/assets/a7dc89cb-7b4f-4def-929e-a98492ed404b)

raw_bureau represents application data from previous loans that a client got from other institutions that npt Home Credit, raw_dimensions represents various information regarding the clients that had applications in Home Credit, raw_facts represents other information regaring the clients that had applications in Home Credit, here including the monetary amounts of the credits, along with other many features. In the end, raw_target contains the information for each client if they had or not default (TARGET column).

Using dbt core, the construction of each layer of data was created, stating the sources, SQL queries and dependencies. In prd, we can find ready to be consumed data, such as the table prd_analytical_base_table, that contains all ready aggregations and joins that create a table with the correct granularity for training and predicting the occurrence of TARGET = 1, meaning the occurrence of default by that client. As we can see in the image above, this table is the culmination of the modeling and transforming of every table used in the project.

With all this, in Google BigQuery we can check the data available:

![image](https://github.com/user-attachments/assets/51c22e51-5408-4457-ba1e-060495d1e2f3)


### 3) The Model:
All details regarding the whole exploratory data analysis along with feature engineering and model selection can be found in the notebook "eda.ipynb". In the end, after all exploration, the Stacked model below was selected according to the metric ROC UAC:

![image](https://github.com/user-attachments/assets/f61db035-82da-4ca1-a607-cf45e21ba821)

The whole model was encapsulated in a sklearn pipeline that has a preprocessor of categorical features using OneHot Encoding, followed by the Stacked model that has Random Forest, XGBoost and LightGBM as base learners and a final logistic regression as a meta learner.

All engineering processes (including data transformations, feature selection and feature creation) were also developed in python methods ready to be imported and consumed in prod afterwards.
Also, the method that has the whole training pipeline of the model also, automatically, performs probability calibration for the classification model and the threshold tunning for the predictions using F1-Score maximization.


### 4) The Results:
Here, let's state how well the model performed on unseen data after the training process. On unseen data, the model had a Recall of 0.56, a precision of 0.37 and a ROC AUC of 0.85. We have to remember here that this challenge has an extreme class imbalance with an extremely complex data behind. The Recall and Precision may no be outstanding (they also are completely dependent on the threshold found), but for a complex environment, we can accept the results as satisfatory (of course, the business would have this information of a baseline that we should be better than to consider the new model a success).

Also, as said in the beginning of this text, the model had, considering this testing data, a potential financial impact of 1,37 Billion USD assuming that the Credit Amount provided was given in USD, all this considering the Credit Amount of the Client's credits that had default status as positive and were predicted so.

### 5) Building a Production Version of the Model: Deployment:
The model (along with all necessary data pipelines) was deployed through a API REST using FastAPI using a Docker container. After deploying the model, a python script was developed to simulate the client's usage of the API. Below, a screen shot of the model deployed using Docker receiving the API calls, providing the predictions according to the client's selections.

![image](https://github.com/user-attachments/assets/069fca4f-98a4-467b-9f0c-fcce6bb98461)

The script that contains the code that simulates the client's requests is in client_testing.py. This script also saves the predicted data in Google BigQuery, returning the info in the format that the API's output was designed (default_probability: the predict_proba of the model and default_occurrence: a text saying "Yes" if there is a potential default risk accoring to the tunned threshold and "No" otherwise). Below, a example of the output:

![image](https://github.com/user-attachments/assets/b5c476ad-f7d9-4273-b875-d36f868f7c90)

As we can see in the example above, clients 456245, 456233 and 456231 are potentially going default. The probability is important to be shown so that experts can have a notion of the uncertainty of the "Yes" or "No" predictions, therefore being able to sometimes judge the predictions according to domain knowledge.

### 6) Conclusions:
This project showed various technical capabilities of data engineering, data analysis and data science. From the initial ingestions using PySpark and all the data modeling using dbt core to manage data in Google BigQuery to the complete EDA process followed by the creation and optimization of the complex Stacked Machine Learning models, a very broad variety of skills and knowledge was shown, all with the objective of solving a real problem, in a real company with "real" data, that delivers a solid and measureable result as shown.

Main technical skills worked: Google BigQuery, dbt core, PySpark, Python, SQL, Sklearn, Machine Learning, Statistical Analysis, Hypothesis Testing, Docker, FastAPI, Feature Engineering, Relational Modeling, ETL.

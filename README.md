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

Using dbt core, the construction of each layer of data was created using dbt models, stating the sources, SQL queries and dependencies. In prd, we can find ready to be consumed data.


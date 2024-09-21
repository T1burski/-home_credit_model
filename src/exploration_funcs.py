import scipy.stats as stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, normaltest
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def automatic_feature_analysis(df, target_col='TARGET', significance_level=0.05):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(20, 10))
    
    print(f"Automatic Analysis Report for TARGET: {target_col}")
    print("="*60)
    
    for col in categorical_cols:
        print(f"\nAnalyzing Categorical Feature: {col}")
        print("-"*60)
        
        cross_tab = pd.crosstab(df[col], df[target_col], normalize='index')
        print("Proportions of TARGET within each category:")
        print(cross_tab)
        
        cross_tab.plot(kind='bar', stacked=True)
        plt.title(f'Target Proportion by {col}')
        plt.ylabel('Proportion')
        plt.show()
        
        contingency_table = pd.crosstab(df[col], df[target_col])
        if contingency_table.size >= 2:
            if np.all(contingency_table >= 5):
                chi2, p, _, _ = chi2_contingency(contingency_table)
                print(f"Chi-square test p-value: {p}")
                if p < significance_level:
                    print(f"Significant relationship detected between {col} and {target_col} (p < {significance_level}).")
                else:
                    print(f"No significant relationship detected between {col} and {target_col} (p >= {significance_level}).")
            else:
                print("Warning: Some expected counts are less than 5, Chi-square test may not be valid.")
        else:
            print("Chi-square test is not applicable (insufficient categories).")
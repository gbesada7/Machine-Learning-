import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_info = pd.read_csv('lending_club_info.csv', index_col = 'LoanStatNew')

df = pd.read_csv('lending_club_loan_two.csv')

#print(df.info())

#sns.countplot(x = 'loan_status', data = df)
#plt.figure(figsize = (12, 4))
#sns.distplot(df['loan_amnt'], kde = False, bins = 40)#spikes in areas of loan, there are common loan amounts

#plt.figure(figsize = (12, 7))
#sns.heatmap(df.corr(), annot = True, cmap = 'viridis')#high correlation in installment feature (0.95 with loan)

#check installment feature

#feat_info('installment') corr to feat_info('loan_amt')

#sns.boxplot(x = 'loan_status', y = 'loan_amnt', data = df)#similarity/corr with amount and status

#loan grades


#charged off loans higher as letter continues (higher risk if B versus A)
#sns.countplot(x = 'grade', data = df, hue = 'loan_status')
#plt.figure(figsize = (12, 4))
#subgrade_sort = sorted(df['sub_grade'].unique())
#sns.countplot(x = 'sub_grade', data = df, order = subgrade_sort, hue = 'loan_status')

#observing paid/unpaid loans
#df['loan_repaid'] = df['loan_status'].map({'Fully Paid' : 1, 'Charged Off' : 0}) #dummy variables?
#print(df[['loan_repaid', 'loan_status']])

#df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind = 'bar') #has highest negative correlation on whether someone will pay loan
#high interest rate likely to cause lower likelihood of paying loan


#data preprocessing (dealing with missing data)

#missing data as a percentage of the number of values of data
#print(100*df.isnull().sum()/len(df))#missing data percentage (mort_acc is 9.54%), drop those who are less than .05%?

#print(df['emp_title'].nunique())
#print(df['emp_title'].value_counts()) #too many unique job titles to create dummy variables, cannot add 173105 0 and 1...we drop

df = df.drop('emp_title', axis = 1)

print(sorted(df['emp_length'].dropna().unique()))

emp_length_order = ['1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '< 1 year', '10+ years']

sns.countplot(x = 'emp_length', data = df, order = emp_length_order, hue = 'loan_status')#paid off high ratio between job of 10 years and fully paid loan

#strong relationship between employment length and being charged off, we want percentages of charge offs per category
emp_charge_off = df[df['loan_status'] == 'Charged Off'].groupby("emp_length").count()
emp_fully_paid = df[df['loan_status'] == 'Fully Paid'].groupby("emp_length").count()

charge_off_versus_fully_paid = emp_charge_off/(emp_fully_paid+emp_charge_off)

#charge off extremely similar...we can drop

df = df.drop('emp_length', axis = 1)


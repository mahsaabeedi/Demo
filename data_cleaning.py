# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 04:54:47 2020

@author: yiyan
"""
import numpy as np
import pandas as pd
import glassdoor_scraping_tool as gst

## scrape data from glassdoor by glassdoor_scraping_tool using selenium 

path = 'C:/Users/yiyan/Documents/salary_analysis_glassdoor/chromedriver'

# scrape data scientist, data analyst and statistician job listings in Canada 20200509-20200609

df_ds = gst.get_jobs("data", "scientist", 791, False, path,10)
df_ds.to_csv("scraped_data/data_scientist_job_postings_Canada_202000509-0609.csv", index = False) 

df_da = gst.get_jobs("analyst", "data", 234, False, path,10)
df_da.to_csv("scraped_data/data_analyst_job_postings_Canada_202000509-0609.csv", index = False) 

df_stat = gst.get_jobs1("statistician", 362, False, path,10)
df_stat.to_csv("scraped_data/statistician_job_postings_Canada_202000509-0609.csv", index = False) 

# scrape fulltime job listings in Ontario 2020/06/04-2020/06/09

df_ontario_lastweek = gst.get_jobs2("Ontario", "fulltime", 1000, False, path, 10)
df_ontario_lastweek.to_csv("job_postings_Ontario_fulltime_20200604-09.csv")



## load data

df_data_scientist = pd.read_csv("scraped_data/data_scientist_job_postings_Canada_202000509-0609.csv")
df_data_analyst = pd.read_csv("scraped_data/data_analyst_job_postings_Canada_202000509-0609.csv")
df_statistician = pd.read_csv("scraped_data/statistician_job_postings_Canada_202000509-0609.csv")

# df : combined dataframe of data containing 3 job positions

df = pd.concat([df_data_scientist, df_data_analyst,df_statistician], axis= 0)
df.shape # (1387, 15)
df.to_csv('scraped_data/Scraped_Data_Combined.csv', index=False)
df2 = df.copy()

#remeove duplicate rows

df2.drop_duplicates(keep = "first",inplace = True) # shape: (1335, 15)

## clean the variable-- Salary Estimate:

df2[df2['Salary Estimate']== '-1'].shape[0]  # 3 rows where salary is missing
df2['Salary Estimate'].value_counts()
df2 = df2[df2['Salary Estimate'] != '-1'] # nrow is 1332 now
Sal = df2['Salary Estimate'].apply(lambda x: x.split('(')[0])
Sal2 = Sal.apply(lambda x: x.replace('CA$', '').replace('k', ''))
df2['Minimum Salary'] = Sal2.apply(lambda x: x.split('-')[0]).astype('float')
df2['Maximum Salary'] = Sal2.apply(lambda x: x.split('-')[1]).astype('float')
df2['Average Salary'] = (df2['Minimum Salary'] + df2['Maximum Salary'])/2

## clean the variable-- 'Company Name'

df2['Rating'].value_counts() 
df2['Company Name v1'] = df2.apply(lambda x: x['Company Name'] if x['Rating']== -1 else x['Company Name'][:-3].strip('\n'), axis = 1)

## clean the variable-- Headquarters

df2['Headquarters'].value_counts()
df2['headquarter_country/state'] = df2['Headquarters'].apply(lambda x: np.NaN if x == '-1' else x.split(',')[1])

## clean the variable-- Founded

df2['Founded'].value_counts()
df2['company_age'] = df2['Founded'].apply(lambda x: np.NaN if x == -1 else 2020-x)

## clean the variable-- 'Job title'

df2['Job Title'].value_counts()

def job_title_simplifier(job_title):
    if 'data analyst' in job_title.lower() or 'data-analyst' in job_title.lower() or 'data specialist' in job_title.lower() or 'data quality analyst' in job_title.lower() or 'data & reporting analyst' in job_title.lower():
        return 'data analyst'
    elif 'data scientist' in job_title.lower() or 'quantitative research analyst' in job_title.lower() or'data science' in job_title.lower():
        return 'data scientist'
    elif 'data engineer' in job_title.lower():
        return 'data engineer'
    elif 'machine learning engineer' in job_title.lower() or 'big data engineer' in job_title.lower() or 'machine learning' in job_title.lower() or 'big data' in job_title.lower():
        return 'machine learning engineer'
    elif 'research scientist' in job_title.lower():
        return 'research scientist'
    elif 'actuarial analyst' in job_title.lower():
        return 'actuarial analyst'
    elif 'business intelligence analyst' in job_title.lower() or 'business analyst' in job_title.lower():
        return 'business intelligent analyst'
    elif 'statistical analyst' in job_title.lower() or 'statistical scientist' in job_title.lower() or 'statistician' in job_title.lower():
        return 'statistician'
    elif 'manager' in job_title.lower():
        return 'manager analytics'
    else:
        return 'other'

df2['simplified_job_title'] = df2['Job Title'].apply(job_title_simplifier)
df2['simplified_job_title'].value_counts() # 'other': 217
df2[df2['simplified_job_title'] == 'other']['Job Title'].value_counts()
df3 = df2[df2['simplified_job_title'] != 'other'] #remove 'other'


## create the variable -- seniority

def seniority(job_title):
    if 'jr' in job_title.lower() or 'jr.' in job_title.lower() or 'junior' in job_title.lower():
        return 'junior'
    elif 'sr.' in job_title.lower() or 'sr' in job_title.lower() or 'senior' in job_title.lower() or 'lead' in job_title.lower():
        return 'senior'
    elif 'co op' in job_title.lower() or 'coop' in job_title.lower() or 'co-op' in job_title.lower() or 'intern' in job_title.lower():
        return 'intern'
    else:
        return 'unspecified'


df3['seniority'] = df3['Job Title'].apply(seniority)
df3.seniority.value_counts()

## create the variable-- posting date

df3['job_age_delta'] = pd.to_timedelta(df3['job age'], unit = 'd')
df3['posting date'] = pd.Timestamp('20200610') - df3.job_age_delta

## create the variable -- Job Description Length

df3['Job Description Length'] = df3['Job Description'].apply(lambda x: len(x))

## create the variables -- Python, RStudio, Excel, AWS, SAS, Pytorch, sql

df3['python'] = df3['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
df3['rstudio'] = df3['Job Description'].apply(lambda x: 1 if 'r-studio' in x.lower() or 'r studio' in x.lower() else 0)
df3['excel'] = df3['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
df3['aws'] = df3['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
df3['sas'] = df3['Job Description'].apply(lambda x: 1 if 'sas' in x.lower() else 0)
df3['pytorch'] = df3['Job Description'].apply(lambda x: 1 if 'pytorch' in x.lower() else 0)
df3['sql'] = df3['Job Description'].apply(lambda x: 1 if 'sql' in x.lower() else 0)

## finally do some changes on df3 and get a cleaned dataset for EDA

df4 = df3[['Average Salary', 'posting date', 'simplified_job_title' ,'seniority', 'Location', 'Company Name v1',  'Rating', 'Headquarters', 'headquarter_country/state','Size', 'Founded', 'company_age', 'Type of ownership', 'Industry', 'Sector', 'Job Description', 'Job Description Length', 'python', 'rstudio','sql','aws','pytorch','sas','excel']]
df5 = df4.rename(columns = {'simplified_job_title' : 'job title', 'Company Name v1' : 'company name', 'headquarter_country/state': 'headquarter(country/state)', 'company_age': 'company age', 'Headquarters': 'headquarter'})
df5.columns = map(str.lower, df5.columns)
df5['average salary'] = df5['average salary']*1000
df6 = df5.replace('-1', np.NaN).replace(-1, np.NaN)
df_cleaned = df6.copy()
df_cleaned.to_csv('Cleaned Data(data analytics job postings combined).csv', index = False)


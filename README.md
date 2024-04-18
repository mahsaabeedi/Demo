# Analysis on Data Science Jobs in Canada with Salary Prediction Flask API Deployed in Herokou

## Table of Contents

1. [Project Highlights](#project-highlights)
2. [References](#references)
3. [Web Scraping](#web-scraping)
4. [Data Cleaning](#data-cleaning)
5. [Exploratory Data Analysis](#eda)
6. [Model Building](#model-building)
7. [Model Performance](#model-performance)
8. [Productionization](#productionization)

## Project Highlights

* Scraped job listings for 3 target positions from Glassdoor for a one-month period using python and selenium
* Cleaned, Visualized and analyzed job listing data in a variaty of ways using  matplotlib, seaborn, wordcloud, etc.  
* Built salary prediction models for various positions in data analysis area in Canada using Multivariate Linear, Lasso, Random Forest and SVM   
* Fine-tuned Lasso, and Random Forest and SVM using GridsearchCV to achieve the best model (MAE ~ $16k)
* Deployed SVM model ([deployment repo](https://github.com/ensembles4612/technology_term_extractor_app_streamlit_deployed_on_azure)) as a client-facing salary prediction tool on [this website](https://salary-estimator-shelley.herokuapp.com/) using Flask and Heroku 

## References

**Language:** Python 3.7  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, wordcloud, nltk.corpus, nltk.tokenize, missingno, dython.nominal, sklearn, joblib, selenium, flask  
**Project inspired by:** https://github.com/PlayingNumbers/ds_salary_proj
**Scraper Github:** https://github.com/arapfaik/scraping-glassdoor-selenium  
**Scraper Article:** https://towardsdatascience.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905  
**Flask Productionization:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

## Web Scraping

I adjusted the [web scraper](https://github.com/arapfaik/scraping-glassdoor-selenium) using Selenium to scrape the fulltime job postings from glassdoor.ca for 3 target positions -- "data scientist", "data analyst" and "statistician" for a one-month period(2020-05-09 to 2020-06-09) respectively. See code [here](https://github.com/ensembles4612/analysis_and_modeling_on_data_science_jobs_Canada/blob/master/glassdoor_scraping_tool.py). 

With each job, I obtained the following: Job title, Salary Estimate, Job Description, Rating, Company Name, Location, Company Headquarters, Company Size, Company Founded Date, Type of Ownership, Industry, Sector, Revenue, Competitors. Because other positions such as data engineer and machine learning engineer also showed up in the search results of the 3 target positions, the following positions appreared in the search results were also chosen to be included in the dataset for analysis and modeling: data engineer and machine learning engineer, research scientist, business intelligent analyst, manager analytics, actuarial analyst.


## Data Cleaning

After data scraping, I did some cleaning (See code [here](https://github.com/ensembles4612/analysis_and_modeling_on_data_science_jobs_Canada/blob/master/data_cleaning.py)) before building the models for salary prediction such as:
*  Combined 3 datasets and deleted the duplicate rows
*	 Parsed numeric data out of Salary
*  Removed rows with NAs in Salary and created column Average Salary 
*	 Removed unwanted Rating from Company Name text
*	 Transformed Founded date into Company Age
*  Transformed scraped similar job titles into one simplified job title for all positions
*  Created column Seniority from Job Title
*  Created column Job Description Length from Job Description
*	 Created columns for if different skills were listed in Job Description: Python, RStudio, Excel, AWS, SAS, Pytorch, sql

For missing values in predictor variables, I did the following:
* Detected if NAs were randomly distributed in predictor variables. If not, deleted those variables  
* Deleted variables that have too many NAs and levels
 

## EDA

After cleaning the scraped data, I did some brief data visualization and analysis. Based on the scraped data from Glassdoor of the one-month period, below are some of the plots and analyses. See code [here](https://github.com/ensembles4612/analysis_and_modeling_on_data_science_jobs_Canada/blob/master/EDA.ipynb). 

* Total number of new fulltime job postings for statistician, data analyst and data scientist were 1, 16, 12 respectively per day on average:  

![alt text](https://github.com/ensembles4612/analysis_and_modeling_on_data_science_jobs_Canada/blob/master/wordcloud_img/%23job_postings_by_date.png "job_postings_by_date")

* Most jobs were unspecified regarding seniority. For jobs sepecified with seniority, senior positions were demanded around 10 times more than junior and intern positions combined:

![alt text](https://github.com/ensembles4612/analysis_and_modeling_on_data_science_jobs_Canada/blob/master/wordcloud_img/%23jobs_by_seniority.png "jobs_by_seniority")

* Boxplot of salary distribution for data analyst of seniority made most sense since there was enough data for this position. As we can see for data analyst, only approx 1k rise on median salary from intern to junior, but more than 10k from junior to senior:

![alt text](https://github.com/ensembles4612/analysis_and_modeling_on_data_science_jobs_Canada/blob/master/wordcloud_img/boxplot_for_salary_by_job_title(color%3Dseniority).png "boxplot_for_salary_by_job_title")

* Graph below shows the percentage of different skills were listed in the job descriptions for each position. 
  * Excel was the most desirable for actuarial analyst and data analyst with Excel listed in more than 70% job discriptions for both positions. Excel was the least desirable for machine learning engineer (less than 30%) 
  * Python was the most desirable for data engineer (approx. 85%) followed by data scientist and machine learning engineer (approx. 80%)
  * R/R studio was desired for data analyst, data engineer and data scientist (all under 10%)
  * SAS was the most desirable for statistician (almost 80%) then actuarial analyst (less than 60%)
  * SQL was the most desirable for data engineer (approx. 85%) and then data analyst and business intelligent analyst (approx. 65%) 

![alt text](https://github.com/ensembles4612/analysis_and_modeling_on_data_science_jobs_Canada/blob/master/wordcloud_img/tools_mentioned_in_job_discription(%25).png "tools_mentioned_in_job_discription(%)")

* Top10 company that released most jobs with their company ratings:

<p align="center">
<img width="450" height="550" src="https://github.com/ensembles4612/analysis_and_modeling_on_data_science_jobs_Canada/blob/master/wordcloud_img/top10_company_released_most_jobs_with_ratings.png">
</p>

* Total number of jobs released by sector and by job title and salary distribution:

<p align="center">
<img width="450" height="550" src="https://github.com/ensembles4612/analysis_and_modeling_on_data_science_jobs_Canada/blob/master/wordcloud_img/salary_and_%23jobs_by_sector_by_job_title.png">
</p>

* Total number of jobs released by location and by job title and salary distribution:

<p align="center">
<img width="450" height="550" src="https://github.com/ensembles4612/analysis_and_modeling_on_data_science_jobs_Canada/blob/master/wordcloud_img/salary%20distribution%20and%20count%20of%20jobs%20by%20location%20by%20title.png">
</p>

* Wordcloud for job description regarding Data Analyst(left), Statistician(center) and Data Scientist(right): 

<img align="left" width="250" height="450" src="https://github.com/ensembles4612/analysis_and_modeling_on_data_science_jobs_Canada/blob/master/wordcloud_img/wordcloud_DataAnalyst.png">
<img align="right" width="250" height="450" src="https://github.com/ensembles4612/analysis_and_modeling_on_data_science_jobs_Canada/blob/master/wordcloud_img/wordcloud_DataScientist.png">
<p align="center">
  <img width="250" height="450" src="https://github.com/ensembles4612/analysis_and_modeling_on_data_science_jobs_Canada/blob/master/wordcloud_img/wordcloud_Statistician.png">
</p>

## Model Building 

I did the following before building models:

* Train test set split: I splited the data into training set (80%) and tests set (20%).
* Deciding which predictor variables to include in the model based on correlation matrix: In order to avoid bias (keeping test data unseen by the response variable - average salary), I made the corelation heatmap using only training data. Then, decided the following variables to be included in the model based on the heatmap:  Job title, Seniority, Location, Company Size, Type of Ownership, python, rstudio, sql, aws, pytorch, sas, excel.
The heatmap is shown below:

![alt text](https://github.com/ensembles4612/analysis_and_modeling_on_data_science_jobs_Canada/blob/master/wordcloud_img/heatmap.png "heatmap")

* Transforming the categorical variables into dummy variables.

I tried 4 different models and evaluated them using Mean Absolute Error. They are:
*	**Multiple Linear Regression** – Baseline model
*	**Lasso Regression** – Data was very sparsed due to the many categorical variables, so I tried Lasso
*	**Random Forest** 
* **Support Vector Regressor** 


## Model Performance
I fine-tuned the following 3 models using GridSearchCV to find the best parameters with 10-fold cross validation. The test errors are:
*	**Lasso Regression** : MAE = 16784
*	**Random Forest**: MAE = 17745
*	**Support Vector Regressor**: MAE = 16288

Below was the graph I plotted regarding **test set pred vs. actual average salary** for the 3 models. Test error of SVM outperformed that of the other 2 approaches. 

![alt text](https://github.com/ensembles4612/analysis_and_modeling_on_data_science_jobs_Canada/blob/master/wordcloud_img/pred_vs_actual.png "pred_vs_actual")

* **Reflection on the model performance**: we can tell from the graph above that the 3 models tended to only predict salary between approx. from 50k to 90k well. Reasons for this might be that the majority of the jobs scraped had average salary that fell in this range. We can see this from the below salary distribution graph. What we can do to fix this problem is to scrape more data with salary that fall beyond this range and add them to dataset, then retrain the models.   

![alt text](https://github.com/ensembles4612/analysis_and_modeling_on_data_science_jobs_Canada/blob/master/wordcloud_img/salary_dist.png "salary_dist")

## Productionization 

I built a flask API endpoint using the SVM model and deployed it in Heroku as a client-facing salary prediction website [here](https://salary-estimator-shelley.herokuapp.com/predict). The deployment code is in [this repo](https://github.com/ensembles4612/Salary_prediction_flask_api_heroku). On the website, you can choose values from the drop-down lists and submit. Then, the flask app takes in the request with the values and returns an estimated salary that will be shown on the website for you like below:

![alt text](https://github.com/ensembles4612/analysis_and_modeling_on_data_science_jobs_Canada/blob/master/wordcloud_img/salary_estimator.png "salary_estimator")


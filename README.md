# CaRLMaLP

Cancer-Related LncRNA Machine Learning Predictor App (CaRLMaLP R-shiny App) is a Web-based user interface for predicting cancer-related lncRNAs by integrating manifold features (genomic, expression, epigenetic and network features) with five machine-learning techniques (Random Forest (RF), Naïve bayes (NB), Support Vector Machine (SVM), Logistic Regression (LR) and K-Nearest Neighbors (KNN). With this App, predicting novel cancer-related lncRNA with high accuracy and efficiency is possible. This App was developed using R Shiny and based on the machine learning work from [CRlncRC](https://bmcmedgenomics.biomedcentral.com/articles/10.1186/s12920-018-0436-9).
                

## User manual 
### Create Input
Input should be a csv file with Gene_ID as the first column(can be any form of Gene ID), the rest of the file should contain some of the columns(does not have to be all) in the features file 'feature_set.csv'. Our features file contains 4 types of integrated feature: genomic features, epigenetic features, expression features, network features.
### Run App 
Download the [files] and go to the directory. 
       
run command: 'shiny run --reload app.py'

open browser and go to: 'http://127.0.0.1:8000'

Upload your file for analysis. 

# Pre-requirements 
This contains the scripts for the Machine Learning Analysis. To use these, you will need to install a version of Python 3, and use your preferred Python IDE (Spyder was used in the current paper). You will also need to make sure you have installed the matplotlib, pandas, numpy and sklearn Python modules to use the script. You will also need Rstudio in order to run the growth mixture models. 

Note that this directory also contains the raw pain diary data in "Train/Pain_Diary_Data_train.xlsx" and "Test_shuffled/Pain_Diary_Data_Test_Unshuffled.xlsx". It also contains the Raw Demographic and Covariate Scores in "Test_shuffled/Covariate_Data_test_unshuffled" and "Test_shuffled/Demographics_test_unshuffled" as well as "Train/Demographics_train" and "Train/Covariate_Data_train"

# Using the Scripts

## Preprocessing the Raw Pain Diary Data
The first step is to organise the pain diary data (and impute missing pain diary data) which will then be used for the growth mixture modelling 

"preprocessing_train.py" calls on "Train/Pain_Diary_Data_train.xlsx" to preprocess the raw training set pain diary data with the output being "Train/df_chew_filled_train.csv" and "Train/df_yawn_filled_train.csv" 
NOTE: If you get an error with RPY2 Module, try commenting out Lines 7-8. 

"shuffle_ID.py" calls on raw test set data "Test_unshuffled/Pain_Diary_Data_Test_Unshuffled.xlsx" to shuffle the test set diary data and saves output to "Test_unshuffled/ID_dropped_df.csv" and 'Test_unshuffled/ID_df.csv' as well as 'Test_shuffled/Pain_Diary_Data_Test_Shuffled.xlsx'. 

"preprocessing_test.py" calls on "Test_shuffled/Pain_Diary_Data_Test_Shuffled.xlsx" to process the shuffled test set pain diary data with the output being "Test_shuffled/df_chew_filled_test.csv" and  "Test_shuffled/df_yawn_filled_test.csv"
NOTE: If you get an error with RPY2 Module, try commenting out Lines 7-8. 

## Growth Mixture Modelling
To run the growth mixture modelling, use "R_Script_LGM.R" in RStudio. "R_Script_LGM.R" calls on "Train/df_chew_filled_train.csv", "Train/df_yawn_filled_train.csv", "Test_shuffled/df_chew_filled_test.csv" and "Test_shuffled/df_yawn_filled_test.csv" to run
the growth mixture modelling using data from the first 7 days. 

The script for installing and acquiring the required R packages are included. If you still get any error about the functions not being recognised, you might have to install that function using install.packages. 

The output of the script is saved as "Test_shuffled\\df_ID_LGM_test.csv" and "Train\\df_ID_train_LGM.csv", which contains the IDs of participants labelled as high or low pain sensitive. 

## Training Set Analysis
"parameter_tuning.py" defines the classifier tuner which is necessary to run the machine learning scripts  

"ML_classification_PAF_CME" runs the machine learning scripts on the training set and establishes predicted pain labels for the test set.  It calls on 'Train/df_ID_train_LGM.csv', 'Train/Sensorimotor_Peak_Alpha_Frequency_train.xlsx', 'Train/Map_Volume_train.xlsx'- which are respectively the pain labels established by the R Script GMM, the PAF values (obtained from the Manual_ICA_Pipeline) and the Map Volumes obtained from the TMS Scripts. It then runs and compares 5 differet machine learning  algorithms and saves the output in "fitted_models" folder. The test data is then loaded   from 'Test_shuffled/Map_Volume_test_unshuffled.xlsx' and 'Test_shuffled/Sensorimotor_Peak_Alpha_Frequency_test_unshuffled.xlsx' in order to make predictions about pain labels in the test set. This is saved as 'Test_shuffled/PAF_CME_df_test_predicted.csv' and 'Test_shuffled/PAF_CME_results_test_proba.csv' The performance of the logistic regression (winning) classifier is plotted and saved as  'figures/Training_ROC_curve.png'

## Test Set Analysis 
"compare_results_PAF_CME" uses the predicted pain labels established from the training set analysis and compares against the actual pain labels established by the GMM. It calls on 'Test_shuffled/PAF_CME_df_test_predicted.csv' and 'Test_shuffled/df_ID_LGM_test.csv' as well as 'Test_unshuffled/ID_df.csv', and 'Test_shuffled/PAF_CME_results_test_proba.csv', and plots roc curves of false and true positive rates. Results are plotted as 'figures/PAF_CME_roc_test.png' and 'figures/PAF_CME_pain_severity_test.png'


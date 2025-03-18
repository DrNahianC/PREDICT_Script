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




# Update 18/03/25: Response to Buchel et al. 

On 18/02/25, we received feedback on our analysis approach from Buchel et al. https://github.com/olegolt/PAF_reanalysis/tree/main . Here is our response to this feedback below. 

## Criticism 1 – Use of a single Validation Set

To respond to the Buchel et al’s  criticism on the use of a single validation set in the training set, we followed up on this by varying the random seed (and therefore varying the validation set) between 1 and 100. While our distribution of validation AUCs matches what the authors show in their analysis, what the authors do not show are the results of the winning classifier applied to the test set. Specifically, regardless of the seed that is chosen, when the winning classifier is locked in and applied to the test set, the performance of the model is almost always in the excellent range as shown in Figure 1. Thus, regardless of internal training and validation split, performance of the locked model in the test set was excellent, supporting our conclusions.

![image](https://github.com/user-attachments/assets/e1e62ceb-1e86-4999-84ec-73ddde5980b7)

Figure 1. Distribution of Test AUCs across 100 random seeds for the chosen split in the training set. The Y-axis shows the number of AUCs in a particular AUC range and the x-axis shows the AUC ranges.

## Criticism 2 – Use of a single test set

### The use of a single test set was according to a pre-defined protocol selected a-priori to ensure unbiased and robust assessment of diagnostic performance
To respond to Buchel et al’s second criticism on the use of a single test set, we note that the use of a single data split was a pre-registered approach. Our analysis strictly follows the originally planned (and published) study protocol, which specifies using the first 100 enrolled patients for training and the remaining 50 for testing. By adhering to the pre-registered protocol, we maintained methodological consistency, prevented bias, and ensured a realistic assessment of diagnostic performance, making this the most appropriate approach for our study.

### The protocol was chosen to match with real-world clinical deployment
Our protocol aligns with real-world clinical deployment, where models are trained on historical patient data and applied prospectively to new, unseen cases. Unlike randomized train-test split (the approach proposed by Buchel et al), which mixes training and test data across multiple splits, our approach ensures that the final model is evaluated on a truly independent holdout set, providing an unbiased estimate of real-world performance. While repeated train-test splits with multiple random seeds can be useful during model evaluation, they do not provide a final, independent assessment necessary for validation on truly unseen test data. Thus, Buchel’s approach, which does not evaluate the model in a truly independent test-set, provides a less-rigorous approach to validation that fails to consider the intended real-world application for the biomarker signature. 

## An alternative approach addressing the author’s concern of a single split yields an AUC of 0.88.

Based on the points above, we contend that our pre-defined analysis approach was the most appropriate statistical analyses for analytical validation of our biomarker. However, to directly address the concerns raised by the authors of using a single split, we ran a more generalizable statistical approach using an ensemble of 50 logistic regression models, selected as the top 10% of 500 models based on validation performance. 

To minimize performance overestimation from a single split, we conducted 500 train-test splits in the training set of 100 subjects, with varying seeds, ensuring a stable and unbiased assessment. For model training, we used a selective ensembling strategy. Only logistic regression models were trained on peak-alpha frequency (PAF) and binarized corticomotor excitability (CME), with hyperparameters tuned via 5-fold cross-validation. The top 10% of models, based on validation AUC, were ensembled to generate final test predictions. This approach mitigates overfitting, ensuring that only well-generalizing models contribute to the final prediction.
To prevent information leakage, we revised our imputation procedure. Instead of imputing the entire dataset before splitting, we now fit the imputer only on the training data for each split, applying it separately to the validation and test sets. While the missing data in CME is minimal—suggesting a negligible impact on results—this methodological refinement further strengthens the credibility of our analysis.

Our ensemble, drawn from 500 diverse splits, stabilizes predictions and prevents overfitting by averaging across high-performing models. Logistic regression produces calibrated probabilities, making this averaging approach theoretically sound. Importantly, ensembling the top 10% rather than all 500 models avoids diluting performance with weaker predictors. This balance optimizes robustness, avoiding both the variance of a single model and the noise of a full ensemble.

Our re-analysis yields a test AUC of 0.88 and accuracy of 0.74, supporting our original findings and directly countering the concerns of a single data split raised by Buchel et al. Visualizations, including an ROC curve confirm our model’s ability to distinguish pain sensitivity groups. The analysis can be found in multiple_random_seeds_test.py and compare_results_multiple_random_seeds_test.py of the Machine Learning Scripts folder

## Criticism 3 – protocol deviations

The protocol deviations asserted by Buchel and colleagues reflect critical misunderstandings of our protocol. 
1.	In the authors’ github repository, they state that CME on Day 5 was the “original” predictor but this reflects a fundamental misunderstanding of the theoretical background of our paper.  Our intention was always to examine CME as a difference, not as an absolute value. The introduction clearly presents preliminary evidence of pain sensitivity differing between facilitators (those whose CME increases) and depressors (those whose CME decreases) – showing we were always interested in “change in CME” as a predictor. 

2.	Regarding the deviation of not matching the proportion of high and low pain sensitive individuals in the training and test set, this was justified. Since participants from the training and test sets were drawn from the same population, we applied the same growth mixed model to acquire the probabilities of a subject being high or low pain sensitive. Though this changes the proportion of high and low pain sensitive individuals, this is the best way to classify the test set as it avoids any potential bias in pain sensitivity classification, because if any model other than the trained GMM model was used, it is equivalent to saying that the training and test sets are from different patient populations that have different baseline characteristics and pain profiles. Our analysis method was therefore the most robust approach for our data.

## Criticism 4 – limitations on clinical utility

The authors seem to suggest that an AUC of 0.74 limits the clinical utility of the biomarker, but this is inaccurate. In fact, even if we adopt the authors’ approach (and we strongly contend that it is neither the most robust nor the most appropriate approach) the biomarker is STILL promising. Our paper reports a series of sensitivity analyses using different methodological approaches to analyze our data (e.g. different PAF/CME calculation methods, different GMM classification approaches, inclusion of covariates) in order to account for potential biases in our pre-registered analysis method. The random split approach presented by Buchel et al. is at best, another type of sensitivity analysis. Ultimately, the number of different analysis methods available for large-scale data sets is unlimited making adherence to a pre-registered, a-priori justified protocol essential. We note that because this was an analytical validation study, we selected to run those sensitivity analyses that were most likely to impact future clinical translation (i.e. different PAF/CME calculation methods, inclusion of covariates etc). However, regardless of the sensitivity analyses run (including that of the Buchel group) we found at least acceptable AUCs for the test set performance. Critically, at least acceptable AUC (>0.7) was our a-priori defined hypotheses (see last sentence of introduction) and benchmark for moving to the next stage of biomarker development – clinical validation. Thus, even with the AUC of 0.74 reported by Buchel et al, the PAF/CME signature meets criteria for moving to clinical validation and shows promise as a biomarker. It also retains one of the strongest predictive accuracies of any biomarker so far discovered in the field of pain. Therefore, we strongly dispute the authors’ claim that an AUC of 0.74 limits the biomarker’s clinical utility or that our conclusions are overstated. 

## Summary

There are many ways to analyse large-scale data sets. Pre-registration of robust protocols along with a-priori defined hypotheses and clear benchmarks for what constitutes success is therefore essential. Our study meets these conditions and adheres to the highest standards of scientific conduct and reporting. Based on clear justification of our analysis plan, along with multiple sensitivity analyses, we contend that our findings provide an accurate assessment of the potential of our biomarker signature and are not overstated. The analysis conducted by Buchel’s group does not provide justification for a ‘better’ statistical approach for analytical validation nor does it result in AUC scores that challenge our original analysis methods or interpretation.






Please download this folder containing all the files for MATLAB, R and Python Scripts and other associated files for the PREDICT Study (https://jamanetwork.com/journals/jamaneurology/fullarticle/2829261) 

# Update 18/03/25: Response to Buchel et al. 

On 18/02/25, we received feedback on our analysis approach from Buchel et al. https://github.com/olegolt/PAF_reanalysis/tree/main. Here is our response to this feedback below. 

## Criticism 1 – Use of a single Validation Set

To respond to Buchel et al’s  criticism on the use of a single validation set in the training set, we followed up on this by varying the random seed (and therefore varying the validation set) between 1 and 100. While our distribution of validation AUCs matches what the authors show in their analysis, what the authors do not show are the results of the winning classifier applied to the test set. Specifically, regardless of the seed that is chosen, when the winning classifier is locked in and applied to the test set, the performance of the model is almost always in the excellent range as shown in Figure 1. Thus, the statement that model performance is a “consequence of a particular random seed” is incorrect. Regardless of the internal training and validation split, performance of the locked model in the test set was excellent, supporting our conclusions.

![image](https://github.com/user-attachments/assets/e1e62ceb-1e86-4999-84ec-73ddde5980b7)

Figure 1. Distribution of Test AUCs across 100 random seeds for the chosen split in the training set. The Y-axis shows the number of AUCs in a particular AUC range and the x-axis shows the AUC ranges.

## Criticism 2 – Use of a single test set

### The use of a single test set was according to a pre-defined protocol selected a-priori to ensure unbiased and robust assessment of diagnostic performance
To respond to Buchel et al’s second criticism on the use of a single test set, we note that the use of a single data split was a pre-registered approach, as the authors themselves acknowledge. By adhering to this, we maintained methodological consistency, prevent bias, and ensure a realistic assessment of model performance, making this the most appropriate approach for our study. We therefore encourage the authors to be cautious about replacing the main conclusions from a pre-defined approach with a post-hoc alternative. 

### The protocol was chosen to match with real-world clinical deployment
Our protocol mirrors real-world clinical deployment: models are trained on historical patient data, the best-performing model is selected and locked in, and then applied prospectively to new cases. In contrast, Buchel et al.'s approach is less rigorous in key ways:
- They do not conclude their analyses with a fixed model for future data - If a decision needs to be made on which model to be used, it makes sense to select the one that performs best to be assessed on the test set. Our approach ensures this by selecting the top-performing model from a validation set and locking it in.
- Mixing of Training and Test Data – Their method blends training and test data across multiple splits, preventing evaluation on a truly independent holdout set. In contrast, we performed model selection on the first 100 participants, locked in the best model, and then evaluated it on the next 50 participants i.e. the future unseen test set.
By locking in the best model and testing it on an independent holdout set, our approach provides an unbiased estimate of real-world performance. In contrast, Buchel et al.'s method lacks this rigor and fails to align with real-world biomarker deployment. 

## An alternative approach addressing the author’s concern of a single split yields an AUC of 0.88.

Based on the points above, we contend that our analysis approach was the most appropriate statistical analyses for analytical validation of our biomarker. Nonetheless, to directly address the concerns raised by the authors of using a single split, as well as the other "code inconstencies" in their github repository, we ran a more generalizable statistical approach using an ensemble of 50 logistic regression models, selected as the top 10% of 500 models based on validation performance. 

To minimize performance overestimation from a single split, we conducted 500 train-test splits in the training set of 100 subjects, with varying seeds, ensuring a stable and unbiased assessment. For model training, we used a selective ensembling strategy. Only logistic regression models were trained on peak-alpha frequency (PAF) and binarized corticomotor excitability (CME), with hyperparameters tuned via 5-fold cross-validation. The top 10% of models, based on validation AUC, were ensembled to generate final test predictions. This approach mitigates overfitting, ensuring that only well-generalizing models contribute to the final prediction.

To prevent information leakage, we revised our imputation procedure. Instead of imputing the entire dataset before splitting, we now fit the imputer only on the training data for each split, applying it separately to the validation and test sets. While the missing data in CME is minimal—suggesting a negligible impact on results—this methodological refinement further strengthens the credibility of our analysis.

Our ensemble, drawn from 500 diverse splits, stabilizes predictions and prevents overfitting by averaging across high-performing models. Logistic regression produces calibrated probabilities, making this averaging approach theoretically sound. Importantly, ensembling the top 10% rather than all 500 models avoids diluting performance with weaker predictors. This balance optimizes robustness, avoiding both the variance of a single model and the noise of a full ensemble.

Our re-analysis yields a test AUC of 0.88 and accuracy of 0.74, supporting our original findings and directly countering the concerns of a single data split raised by Buchel et al. Visualizations, including an ROC curve confirm our model’s ability to distinguish pain sensitivity groups. The analysis can be found in multiple_random_seeds_test.py and compare_results_multiple_random_seeds_test.py of the Machine Learning Scripts folder

![image](https://github.com/user-attachments/assets/a4076e0c-587d-4af4-b675-29735948e095)

Figure 2. Re-analysis using a more generalizable statistical approach using an ensemble of 50 logistic regression models, selected as the top 10% of 500 models based on validation performance 

## Criticism 3 – protocol deviations

The protocol deviations asserted by Buchel and colleagues reflect misunderstandings of our protocol. 

1.	In the authors’ github repository, they state that CME on Day 5 was the “original” predictor but this is likely a misunderstanding of the theoretical background of our paper.  Our intention was always to examine CME as a binary facilitator vs. depressor split, not as an absolute value - as written in both the grant application and in the preliminary studies https://www.sciencedirect.com/science/article/pii/S1526590019307448 and https://www.biorxiv.org/content/10.1101/278598v1.abstract. The introduction of both the protocol paper and the main paper also clearly presents preliminary evidence of pain sensitivity differing between facilitators (those whose CME increases) and depressors (those whose CME decreases) – showing we were always interested in “change in CME” as a predictor. CME on Day 5 in the protocol is referring to CME status on Day 5 (facilitator or depressor). Therefore we contend that our operationalization of CME in the main paper was not a protocol deviation

2.	Regarding the deviation of not matching the proportion of high and low pain sensitive individuals in the training and test set, this was justified. Since participants from the training and test sets were drawn from the same population, we applied the same growth mixed model to acquire the probabilities of a subject being high or low pain sensitive. Though this changes the proportion of high and low pain sensitive individuals, this is the best way to classify the test set as it avoids any potential bias in pain sensitivity classification, because if any model other than the trained GMM model was used, it is equivalent to saying that the training and test sets are from different patient populations that have different baseline characteristics and pain profiles. Our analysis method was therefore the most robust approach for our data.

## Criticism 4 – Claims about limited clinical utility

The authors seem to suggest that an AUC of 0.74 limits the clinical utility of the biomarker, but this is inaccurate in two ways. 

- Even if we were to adopt the authors’ approach—despite our strong stance that it is neither the most robust nor the most appropriate—the biomarker remains promising. Large-scale datasets offer unlimited analytical methods, making adherence to the pre-registered single split approach essential. That said, we still conducted several sensitivity analyses in our paper. As an analytical validation study, we focused on sensitivity analyses most relevant to future clinical translation (e.g., different PAF/CME calculation methods, inclusion of covariates). The random split method used by Buchel et al. is, at best,  another form of sensitivity analysis. Crucially, regardless of the sensitivity analyses—including Buchel et al.’s approach—we consistently found at least acceptable AUCs for test set performance. Our hypothesis explicitly stated that an AUC > 0.7 would indicate acceptable performance (see final sentence of the introduction). Even with the AUC of 0.74 reported by Buchel et al., the PAF/CME signature has acceptable performance and remains a promising biomarker candidate. In fact, it retains one of the highest predictive accuracies of any biomarker discovered in the field of pain to date.
- A further comment - this study was an analytical, not a clinical, validation study and consequently, the aim was to determine biomarker potential not clinical utility. Thus we encourage the authors not make inferences about clinical utility from this study

To conclude, we strongly dispute the authors’ claim that an AUC of 0.74 limits the biomarker’s clinical utility or that our conclusions are overstated. 

## Summary

There are many ways to analyse large-scale data sets. Pre-registration of protocols along with hypotheses and clear benchmarks for what constitutes acceptable model performance is therefore essential. Our study meets these conditions and adheres to the highest standards of scientific conduct and reporting. Based on clear justification of our analysis plan, along with multiple sensitivity analyses, we contend that our findings provide an accurate assessment of the potential of our biomarker signature and are not overstated. The analysis conducted by Buchel’s group does not provide justification for a ‘better’ statistical approach for analytical validation nor does it result in AUC scores that challenge our original analysis methods or interpretation.


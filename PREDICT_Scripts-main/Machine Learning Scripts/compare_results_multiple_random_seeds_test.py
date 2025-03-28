import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.metrics import auc
from matplotlib import pyplot as plt

# load predicted test labels
df_test = pd.read_csv('Test_shuffled/df_test_predicted_multiple_seeds.csv')

# read test labels
df_GT = pd.read_csv('Test_shuffled/df_ID_LGM_test.csv')
# rename ID to shuffled_ID
df_GT = df_GT.rename(columns={'ID':'ID_shuffled'})
# read mapper
df_mapper = pd.read_csv('Test_unshuffled/ID_df.csv')
# merge mapper and df_GT
df_GT = pd.merge(df_mapper, df_GT, on='ID_shuffled')
# merge df_GT and df_test
df_test = df_test.reset_index()
df_test = df_test.rename(columns={'id':'ID'})
df_final_test = pd.merge(df_GT, df_test, on='ID')

# calculate the number of matching elements
matches = np.sum(df_final_test['class'] == df_final_test['label'])

# calculate the accuracy
accuracy = matches / len(df_final_test['class'])

print('Accuracy:', accuracy)

# load test probabilities
df_test_proba = pd.read_csv('Test_shuffled/lr_results_test_proba_multiple_seeds.csv')
# merge df_test_proba and df_final_test
df_final_test = pd.merge(df_final_test, df_test_proba, on='ID')

# plot the auc-roc curve and display auc on the legend
from sklearn.metrics import roc_curve
lr_fpr_test, lr_tpr_test, lr_thresholds_test = roc_curve(df_final_test['class'], df_final_test['high'])
# plot the ROC curve for the voting classifier, including auc in the legend, and accuracy in the title
plt.figure(figsize=(10, 8))
plt.plot(lr_fpr_test, lr_tpr_test, label='AUC = {:.2f}'.format(auc(lr_fpr_test, lr_tpr_test)), linewidth=4)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate', fontsize=28)
plt.ylabel('True Positive Rate', fontsize=28)
plt.legend(loc='best', fontsize=20)
plt.savefig('figures/roc_test_multiple_seeds.png')
plt.show()
# save figures


# plot the pain diary data based on the predicted labels
# load the diary
df_yawn = pd.read_csv('Test_shuffled/df_yawn_filled_test.csv', index_col='ID')
df_chew = pd.read_csv('Test_shuffled/df_chew_filled_test.csv', index_col='ID')
df_diary = df_yawn + df_chew
# high_pain_severity
ID_high = df_final_test.loc[df_final_test['label'] == 1, 'ID_shuffled'].to_list()
df_high_pain_severity = df_diary.loc[df_diary.index.isin(ID_high), :]
ID_low = df_final_test.loc[df_final_test['label'] == 0, 'ID_shuffled'].to_list()
df_low_pain_severity = df_diary.loc[df_diary.index.isin(ID_low), :]
# plot the diary according to the df_test.label
plt.figure(figsize=(10, 8))
# mean values of the diary
plt.plot(df_high_pain_severity.mean(), label='High pain severity', linewidth=3)
plt.plot(df_low_pain_severity.mean(), label='Low pain severity', linewidth=3)
# add confidence interval
plt.fill_between(df_high_pain_severity.columns, df_high_pain_severity.mean() - df_high_pain_severity.std(),
                    df_high_pain_severity.mean() + df_high_pain_severity.std(), alpha=0.2)
plt.fill_between(df_low_pain_severity.columns, df_low_pain_severity.mean() - df_low_pain_severity.std(),
                    df_low_pain_severity.mean() + df_low_pain_severity.std(), alpha=0.2)
# vertical xlabels
plt.xlabel('Time', fontsize=28, labelpad=20)
plt.xticks(rotation=90)
plt.ylabel('Pain severity', fontsize=28)
plt.legend(loc='best', fontsize=20)
plt.savefig('figures/pain_severity_test_multiple_seeds.png')
plt.show()



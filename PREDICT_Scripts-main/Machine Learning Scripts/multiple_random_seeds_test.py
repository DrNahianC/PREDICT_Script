import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
import numpy as np
from parameter_tuning import ClassifierTuner
import warnings
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.makedirs('new_results', exist_ok=True)


# Data Preparation Function (Load raw data only)
def prepare_data(label_file, paf_file, map_volume_file, is_train=True):
    # For training, load labels and filter by IDs; for test, skip filtering
    df_paf = pd.read_excel(paf_file)
    df_map_volume = pd.read_excel(map_volume_file)

    if is_train:
        df_labels = pd.read_csv(label_file)
        ids = df_labels.low.to_list() + df_labels.high.to_list()
        df_paf = df_paf[df_paf.id.isin(ids)]
        df_map_volume = df_map_volume[df_map_volume.id.isin(ids)]

    df_map_volume['CME'] = ((df_map_volume['Volume_Day5'] - df_map_volume['Volume_Day0']) >= 0).astype(int)
    df = pd.merge(df_paf, df_map_volume[['id', 'CME']], on='id')

    if is_train:
        df['label'] = df.id.isin(df_labels['high']).astype(int)
        df = df.drop('id', axis=1)
    else:
        df = df.set_index('id')

    return df



# Prepare Training Data (raw, no imputation)
df_train = prepare_data(
    'Train/df_ID_train_LGM.csv',
    'Train/Sensorimotor_Peak_Alpha_Frequency_train.xlsx',
    'Train/Map_Volume_train.xlsx',
    is_train=True
)
X = df_train.drop('label', axis=1)
y = df_train['label']

# Set parameters for logistic regression tuning
cv = 5
lr_param_grid = {
    'C': np.logspace(-3, 3, 50),
    'solver': ['newton-cg', 'lbfgs'],
    'max_iter': [100, 500, 1000, 2500, 5000, 20000]
}

results_list = []
models_dict = {}


# Train and Evaluate Models over 500 Random Seeds (Manual Imputation)

for seed in range(500):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    # Fit imputer only on training data
    imputer = IterativeImputer(max_iter=100, random_state=seed)
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)

    # Tune logistic regression on imputed training data
    tuner = ClassifierTuner(X_train_imp, y_train, random_state=seed, cv=cv)
    lr_best_estimator = tuner.tune_logistic_regression(lr_param_grid)

    # Save the (imputer, classifier) pair
    models_dict[seed] = (imputer, lr_best_estimator)

    # Evaluate on imputed validation data
    lr_results_proba = lr_best_estimator.predict_proba(X_val_imp)
    lr_fpr, lr_tpr, _ = roc_curve(y_val, lr_results_proba[:, 1])
    lr_auc = auc(lr_fpr, lr_tpr)

    results_list.append({'random_seed': seed, 'lr_val_auc': lr_auc})
    print(f"Seed {seed}: Validation AUC = {lr_auc:.3f}")

# Save performance results and plot histogram
df_results = pd.DataFrame(results_list)
df_results.to_csv('new_results/lr_val_auc.csv', index=False)
plt.figure(figsize=(10, 6))
plt.hist(df_results['lr_val_auc'], bins=20, edgecolor='black', alpha=0.75)
plt.xlabel('Validation AUC')
plt.ylabel('Frequency')
plt.title('Distribution of Logistic Regression Validation AUCs over 500 Random Seeds')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('new_results/lr_val_auc_histogram.png', dpi=300, bbox_inches='tight')

# Select Top 10% Models and Build Ensemble
df_results_sorted = df_results.sort_values('lr_val_auc', ascending=False)
top_n = int(0.10 * len(df_results_sorted))
top_seeds = df_results_sorted.head(top_n)['random_seed'].tolist()
best_models = [models_dict[s] for s in top_seeds]


class EnsembleClassifier:
    def __init__(self, model_tuples):
        """
        model_tuples: list of tuples (imputer, classifier)
        """
        self.model_tuples = model_tuples

    def predict_proba(self, X):
        probas_list = []
        for imputer, classifier in self.model_tuples:
            # Transform test data with the imputer fitted on training data
            X_imp = imputer.transform(X)
            probas_list.append(classifier.predict_proba(X_imp))
        probas_array = np.array(probas_list)
        return np.mean(probas_array, axis=0)

    def predict(self, X):
        avg_probas = self.predict_proba(X)
        return (avg_probas[:, 1] > 0.5).astype(int)


ensemble_model = EnsembleClassifier(best_models)
joblib.dump(ensemble_model, 'new_results/best_ensemble_model.pkl')

# Prepare Test Data and Make Predictions
df_test = prepare_data(
    None,
    'Test_shuffled/Sensorimotor_Peak_Alpha_Frequency_test_unshuffled.xlsx',
    'Test_shuffled/Map_Volume_test_unshuffled.xlsx',
    is_train=False
)
df_test = df_test.loc[~df_test.index.isin([107, 112, 120, 134])]

lr_results_test = ensemble_model.predict(df_test)
lr_results_test_proba = ensemble_model.predict_proba(df_test)

df_test['label'] = lr_results_test
df_test.to_csv('Test_shuffled/df_test_predicted_multiple_seeds.csv')

lr_results_test_proba_df = pd.DataFrame(lr_results_test_proba, columns=['low', 'high'])
lr_results_test_proba_df['ID'] = df_test.index
lr_results_test_proba_df.to_csv('Test_shuffled/lr_results_test_proba_multiple_seeds.csv', index=False)

print("Complete model training, ensemble creation, and prediction completed.")
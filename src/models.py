from preprocess import X_train, X_test, y_train, y_test , plt,cross_val_score, sns ,pd ,X_over, y_over 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
import joblib

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = [acc , prec , rec , f1]
   

    cm= confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(4,3))
    # sns.heatmap(cm, annot=True, fmt='d')
    # plt.title(f"Matrice de Confusion - {name}")
    # plt.xlabel("Prédictions")
    # plt.ylabel("Vérités")
    # plt.show()

print("=" *40)    
results_df = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1-Score"]).T
print("\n Model Performance Table:")
print(results_df)
print("=" *40)

for name, model in models.items():

    cv_scores = cross_val_score(model, X_over, y_over, cv=5, scoring='f1')
    print(f"Cross-validated F1 score ({name}): {cv_scores.mean():.3f}")




best_models = {}
results = {}

# 1️⃣ Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=rf_param_grid,
    n_iter=10,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)
rf_grid.fit(X_over, y_over)
best_models['Random Forest'] = rf_grid.best_estimator_
results['Random Forest'] = rf_grid.best_score_

# 2️⃣ Decision Tree
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt_grid = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=dt_param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
dt_grid.fit(X_over, y_over)
best_models['Decision Tree'] = dt_grid.best_estimator_
results['Decision Tree'] = dt_grid.best_score_

# Logistic Regression
lr_param_grid = {
    'penalty': ['l2'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear']
}

lr_grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    param_grid=lr_param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
lr_grid.fit(X_over, y_over)
best_models['Logistic Regression'] = lr_grid.best_estimator_
results['Logistic Regression'] = lr_grid.best_score_

# Support Vector Machine (SVM)
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

svm_grid = GridSearchCV(
    estimator=SVC(random_state=42),
    param_grid=svm_param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
svm_grid.fit(X_over, y_over)
best_models['SVM'] = svm_grid.best_estimator_
results['SVM'] = svm_grid.best_score_

# Gradient Boosting
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

gb_grid = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=gb_param_grid,
    n_iter=10,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)
gb_grid.fit(X_over, y_over)
best_models['Gradient Boosting'] = gb_grid.best_estimator_
results['Gradient Boosting'] = gb_grid.best_score_

# XGBoost
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

xgb_grid = RandomizedSearchCV(
    estimator=XGBClassifier(random_state=42, eval_metric='logloss'),
    param_distributions=xgb_param_grid,
    n_iter=10,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)
xgb_grid.fit(X_over, y_over)
best_models['XGBoost'] = xgb_grid.best_estimator_
results['XGBoost'] = xgb_grid.best_score_


# Compare all models
print("\n=== Best Cross-Validation F1 Scores ===")
for model_name, score in results.items():
    print(f"{model_name}: {score:.4f}")

# Select best model
best_model_name = max(results, key=results.get)
final_model = best_models[best_model_name]
print(f"\n Best model: {best_model_name}")

joblib.dump(final_model, "model/best_model.joblib")



# results_df["Mean_Score"] = results_df.mean(axis=1)
# best_model_name = results_df["Mean_Score"].idxmax()
# print(f"Best Model Selected: {best_model_name}")
# best_model = models[best_model_name]
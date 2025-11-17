import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, log_loss, accuracy_score, precision_recall_fscore_support
import optuna
import shap
from sklearn.model_selection import StratifiedKFold



def paper_model(features):

    df = pd.DataFrame(features)

    feature_names = [
        'Mean_PP', 'Std_PP', 'Mean_HR', 'Std_HR', 'SD2',
        'Mean_BVP', 'Median_BVP', 'Mode_BVP', 'Min_BVP', 'Max_BVP', 'Std_BVP',
        'M_d1', 'Std_d1', 'M_d2', 'Std_d2', 'HF',
        'Mean_EDA', 'Median_EDA', 'Mode_EDA', 'Max_EDA', 'Min_EDA', 'Std_EDA',
        'N_PEAKS', 'M_Amp', 'M_RT', 'M_D'
    ]
    X = df[feature_names]
    y = df['label']
    # Initialize a random seed for reproducibility
    random_seed = 42

    # Define the objective function for hyperparameter optimization
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }

        rf = RandomForestClassifier(random_state=random_seed, **params)
        
        # Create a StratifiedKFold object for 10-fold cross-validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)
        
        # Lists to store accuracy and loss values for each fold
        fold_accuracies = []
        fold_log_losses = []

        for train_index, val_index in cv.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # Fit the model on the training data for each fold
            rf.fit(X_train, y_train)

            # Calculate training accuracy and loss for the fold
            y_train_pred = rf.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)

            y_train_pred_proba = rf.predict_proba(X_train)
            train_loss = log_loss(y_train, y_train_pred_proba)
            
            # Calculate validation accuracy and loss for the fold
            y_val_pred = rf.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)

            y_val_pred_proba = rf.predict_proba(X_val)
            val_loss = log_loss(y_val, y_val_pred_proba)
            
            # Append to fold lists
            fold_accuracies.append((train_accuracy, val_accuracy))
            fold_log_losses.append((train_loss, val_loss))

        # Calculate the mean training and validation accuracy and log loss across folds
        mean_train_accuracy = np.mean([acc[0] for acc in fold_accuracies])
        mean_val_accuracy = np.mean([acc[1] for acc in fold_accuracies])
        mean_train_log_loss = np.mean([loss[0] for loss in fold_log_losses])
        mean_val_log_loss = np.mean([loss[1] for loss in fold_log_losses])

        # Calculate negative log loss for Optuna (to maximize)
        return -mean_val_log_loss

    # Create an Optuna study for hyperparameter optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50,n_jobs=4)

    # Get the best hyperparameters from Optuna
    best_params = study.best_params

    # Create lists to store accuracy and loss values for each fold
    fold_accuracies = []
    fold_log_losses = []

    # Create a StratifiedKFold object for 10-fold cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)

    for train_index, val_index in cv.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Initialize the model with the best hyperparameters
        best_rf = RandomForestClassifier(random_state=random_seed, **best_params)
        
        # Fit the model on the training data for each fold
        best_rf.fit(X_train, y_train)

        # Calculate training accuracy and loss for the fold
        y_train_pred = best_rf.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        y_train_pred_proba = best_rf.predict_proba(X_train)
        train_loss = log_loss(y_train, y_train_pred_proba)
        
        # Calculate validation accuracy and loss for the fold
        y_val_pred = best_rf.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        y_val_pred_proba = best_rf.predict_proba(X_val)
        val_loss = log_loss(y_val, y_val_pred_proba)
        
        # Append to fold lists
        fold_accuracies.append((train_accuracy, val_accuracy))
        fold_log_losses.append((train_loss, val_loss))

    # Calculate the mean training and validation accuracy and log loss across folds
    mean_train_accuracy = np.mean([acc[0] for acc in fold_accuracies])
    mean_val_accuracy = np.mean([acc[1] for acc in fold_accuracies])
    mean_train_log_loss = np.mean([loss[0] for loss in fold_log_losses])
    mean_val_log_loss = np.mean([loss[1] for loss in fold_log_losses])

    print(f'Mean Training Accuracy: {mean_train_accuracy:.4f}')
    print(f'Mean Validation Accuracy: {mean_val_accuracy:.4f}')
    print(f'Mean Training Log Loss: {mean_train_log_loss:.4f}')
    print(f'Mean Validation Log Loss: {mean_val_log_loss:.4f}')

    # Rest of the code remains the same for evaluating and explaining the model

    # Calculate ROC curve and AUC for the entire dataset (you can add this code if needed)
    y_prob = best_rf.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = roc_auc_score(y, y_prob)

    # Plot ROC curve
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    # Plot accuracy and loss for training and validation
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot([acc[0] for acc in fold_accuracies], marker='o', label='Training Accuracy', color='red')
    plt.plot([acc[1] for acc in fold_accuracies], marker='o', label='Validation Accuracy', color='blue')
    plt.title('Accuracy')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot([loss[0] for loss in fold_log_losses], marker='o', label='Training Binary Cross-Entropy Loss', color='red')
    plt.plot([loss[1] for loss in fold_log_losses], marker='o', label='Validation Binary Cross-Entropy Loss', color='blue')
    plt.title('Binary Cross-Entropy Loss')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plots
    plt.tight_layout()

    # Plot confusion matrix for the entire dataset
    y_pred = best_rf.predict(X)
    conf_matrix = confusion_matrix(y, y_pred)

    plt.figure(figsize=(6, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Entire Dataset)')
    plt.colorbar()
    plt.xticks([0, 1], ['Predicted 0', 'Predicted 1'])
    plt.yticks([0, 1], ['Actual 0', 'Actual 1'])
    plt.xlabel('True')
    plt.ylabel('Predicted')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', verticalalignment='center', color='black', fontsize=12)

    # Calculate and plot validation confusion matrix (overall)
    overall_val_true_labels = []  # List to store true labels for all folds
    overall_val_predictions = []  # List to store predicted labels for all folds

    for train_index, val_index in cv.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Initialize the model with the best hyperparameters
        best_rf = RandomForestClassifier(random_state=random_seed, **best_params)
        
        # Fit the model on the training data for each fold
        best_rf.fit(X_train, y_train)

        # Predict on the validation data
        val_pred = best_rf.predict(X_val)

        # Append true labels and predictions for this fold
        overall_val_true_labels.extend(y_val)
        overall_val_predictions.extend(val_pred)

    # Calculate the validation confusion matrix (overall)
    val_confusion_matrix = confusion_matrix(overall_val_true_labels, overall_val_predictions)

    # Plot the validation confusion matrix (overall)
    plt.figure(figsize=(6, 6))
    plt.imshow(val_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Validation Confusion Matrix (Overall)')
    plt.colorbar()
    plt.xticks([0, 1], ['Predicted 0', 'Predicted 1'])
    plt.yticks([0, 1], ['Actual 0', 'Actual 1'])
    plt.xlabel('True')
    plt.ylabel('Predicted')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(val_confusion_matrix[i, j]), horizontalalignment='center', verticalalignment='center', color='black', fontsize=12)

    # Show the validation confusion matrix
    plt.show()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import matplotlib
from lime.lime_tabular import LimeTabularExplainer

matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


class StockTrendPredictor:
    def __init__(self):
        self.models = {
            'XGBoost': XGBClassifier(
                objective='multi:softprob',  # for multi-class probability
                num_class=3,
                learning_rate=0.1,
                max_depth=6,
                n_estimators=100,
                eval_metric='mlogloss',  # changed from list to string
                use_label_encoder=False,  # important for newer versions
                random_state=42
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'NaiveBayes': GaussianNB()
        }

        self.param_grids = {
            'XGBoost': {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'n_estimators': [100, 200],
                'min_child_weight': [1, 3],
                'gamma': [0, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            },
            'NaiveBayes': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            }
        }

    def train_and_evaluate(self, X, y):
        """Train and evaluate all models with LIME explanations"""
        results = {}

        # Existing feature names setup
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [
                'sentiment_score', 'RSI', 'MACD', 'Returns_5d', 'combined_trend_influence',
                'sentiment_trend_up_interaction', 'sentiment_trend_neutral_interaction',
                'sentiment_trend_down_interaction', 'day_of_week', 'month',
                'influence_Trend_Up', 'influence_Trend_Neutral', 'influence_Trend_Down'
            ]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Convert to numpy arrays if needed
        X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

        for model_name, model in self.models.items():
            try:
                print(f"\nTraining {model_name}...")
                lime_explanations = []

                # Existing grid search code...
                if model_name == 'XGBoost':
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=self.param_grids[model_name],
                        cv=5,
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=1
                    )
                    y_train = y_train.astype(int)
                    y_test = y_test.astype(int)
                else:
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=self.param_grids[model_name],
                        cv=5,
                        scoring='accuracy',
                        n_jobs=-1
                    )

                # Fit the model
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_

                # Make predictions
                y_pred = best_model.predict(X_test)

                # LIME Explanations
                try:
                    class_names = [0, 1, 2]  # For three classes: Downward, Neutral, Upward

                    # Initialize LIME explainer
                    lime_explainer = LimeTabularExplainer(
                        X_test.values,  # Using test data instead of training data
                        class_names=class_names,
                        feature_names=X_test.columns,
                        discretize_continuous=True
                    )

                    idx = 0

                    predicted_class = best_model.predict(X_test.values[idx].reshape(1, -1))[0]
                    explanation = lime_explainer.explain_instance(
                        X_test.values[idx],
                        best_model.predict_proba,
                        num_features=len(feature_names),
                        labels=[predicted_class]  # Only explain the predicted class
                    )

                    predicted_proba = best_model.predict_proba(X_test.values[idx].reshape(1, -1))[0][predicted_class]
                    fig = explanation.as_pyplot_figure(label=predicted_class)
                    plt.title(
                        f'Explanation - {model_name} - Predicted Class: {predicted_class}$_{{{predicted_proba:.2f}}}$',
                        fontsize=8)
                    plt.tight_layout()
                    plt.savefig(f'lime_explanation_{model_name}_instance_{idx}_predicted_class.png')
                    plt.close()

                except Exception as e:
                    print(f"Error generating LIME explanations for {model_name}: {str(e)}")

                # Store all results
                results[model_name] = {
                    'model': best_model,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'test_predictions': y_pred,
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test
                }

                # Continue with existing evaluation code...
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred,
                                            target_names=['Downward', 'Neutral', 'Upward']))

                accuracy = np.mean(y_pred == y_test)
                print(f"\nTest Set Accuracy: {accuracy:.4f}")

                # Generate existing plots...
                self.plot_confusion_matrix(y_test, y_pred, model_name)
                self.plot_feature_importance(best_model, X_train, model_name)
                self.plot_performance_metrics(y_test, y_pred, model_name)

            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        return results

    def load_data(self, news_stock_file, features_file):
        """Load and prepare the data"""
        try:
            news_stock_data = pd.read_csv(news_stock_file)
            features_data = pd.read_csv(features_file)

            threshold = 0.001
            price_change_pct = (news_stock_data['Close'] - news_stock_data['Open']) / news_stock_data['Open']
            target = np.where(price_change_pct > threshold, 2,
                              np.where(price_change_pct < -threshold, 0, 1))

            return features_data, target
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None

    def save_plot(self, fig, filename):
        """Safely save plot to file"""
        try:
            fig.savefig(filename)
            plt.close(fig)
        except Exception as e:
            print(f"Error saving plot {filename}: {str(e)}")

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot and save confusion matrix"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Downward', 'Neutral', 'Upward'],
                        yticklabels=['Downward', 'Neutral', 'Upward'], ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix - {model_name}')
            plt.tight_layout()
            self.save_plot(fig, f'confusion_matrix_{model_name}.png')
        except Exception as e:
            print(f"Error plotting confusion matrix for {model_name}: {str(e)}")

    def plot_feature_importance(self, model, X_train, model_name):
        """Plot feature importance if available"""
        try:
            if hasattr(model, 'feature_importances_'):
                fig, ax = plt.subplots(figsize=(12, 6))
                importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                })
                importance_df = importance_df.sort_values('importance', ascending=False)
                top_15_features = importance_df.head(15)

                sns.barplot(x='importance', y='feature', data=top_15_features, ax=ax)
                ax.set_title(f'Top 15 Feature Importance - {model_name}')
                ax.set_xlabel('Importance Score')
                ax.set_ylabel('Features')
                plt.tight_layout()
                self.save_plot(fig, f'feature_importance_{model_name}.png')
        except Exception as e:
            print(f"Error plotting feature importance for {model_name}: {str(e)}")



    def plot_performance_metrics(self, y_test, y_pred, model_name):
        """Plot performance metrics heatmap"""
        try:
            performance_metrics = classification_report(y_test, y_pred,
                                                        target_names=['Downward', 'Neutral', 'Upward'],
                                                        output_dict=True)
            metrics_df = pd.DataFrame(performance_metrics).transpose()

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(metrics_df.iloc[:-3][['precision', 'recall', 'f1-score']],
                        annot=True, cmap='YlOrRd', ax=ax)
            ax.set_title(f'Model Performance Metrics - {model_name}')
            plt.tight_layout()
            self.save_plot(fig, f'performance_metrics_{model_name}.png')
        except Exception as e:
            print(f"Error plotting performance metrics for {model_name}: {str(e)}")


def main():
    try:
        # Initialize predictor
        predictor = StockTrendPredictor()

        # Load data
        X, y = predictor.load_data('News+Stock data.csv', 'final_features.csv')

        if X is None or y is None:
            print("Failed to load data. Exiting...")
            return

        # Train and evaluate all models
        results = predictor.train_and_evaluate(X, y)

        # Compare models
        comparison_df = pd.DataFrame({
            model_name: {
                'Best CV Score': results[model_name]['best_score'],
                'Best Parameters': str(results[model_name]['best_params'])
            }
            for model_name in results.keys()
        }).transpose()

        print("\nModel Comparison:")
        print(comparison_df)

        # Save comparison results
        comparison_df.to_csv('model_comparison.csv')

    except Exception as e:
        print(f"An error occurred in main: {str(e)}")


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('eda_plots', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('model_results', exist_ok=True)

class CricketWinPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = []
        
    def load_data(self, train_path='data/cricket_dataset.csv', test_path='data/cricket_dataset_test.csv'):
        """Load training and test datasets"""
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        print(f"Data Loaded - Train: {self.train_df.shape}, Test: {self.test_df.shape}")
        return self
    
    def exploratory_data_analysis(self):
        """Comprehensive EDA with insights documentation"""
        print("="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic info
        print("\n1. DATASET OVERVIEW")
        print(f"Training data shape: {self.train_df.shape}")
        print(f"Features: {list(self.train_df.columns)}")
        
        # Data quality assessment
        print("\n2. DATA QUALITY ASSESSMENT")
        null_counts = self.train_df.isnull().sum()
        print("Null values per column:")
        print(null_counts[null_counts > 0])
        
        # Duplicate check
        duplicates = self.train_df.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")
        
        # Statistical summary
        print("\n3. STATISTICAL SUMMARY")
        print(self.train_df.describe())
        
        # Target variable analysis
        print("\n4. TARGET VARIABLE ANALYSIS")
        if 'won' in self.train_df.columns:
            win_dist = self.train_df['won'].value_counts(normalize=True)
            print("Class distribution:")
            print(win_dist)
            print(f"Dataset balance ratio: {win_dist.min():.3f}")
        
        # Key insights from correlation
        print("\n5. FEATURE CORRELATIONS WITH TARGET")
        if 'won' in self.train_df.columns:
            correlations = self.train_df.corr()['won'].abs().sort_values(ascending=False)
            print("Top correlated features with winning:")
            print(correlations.head(6))
        
        # Create visualizations
        self._create_eda_visualizations()
        
        # Document key insights
        self._document_insights()
        
        return self
    
    def _create_eda_visualizations(self):
        """Create comprehensive EDA visualizations"""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Target distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Win/Loss distribution
        self.train_df['won'].value_counts().plot(kind='bar', ax=axes[0,0], color=['red', 'green'])
        axes[0,0].set_title('Win/Loss Distribution', fontsize=14)
        axes[0,0].set_xlabel('Outcome (0=Loss, 1=Win)')
        axes[0,0].set_ylabel('Count')
        
        # Runs vs Balls Left (colored by outcome)
        scatter = axes[0,1].scatter(self.train_df['total_runs'], 
                                   self.train_df['balls_left'], 
                                   c=self.train_df['won'], 
                                   cmap='RdYlGn', alpha=0.6)
        axes[0,1].set_title('Total Runs vs Balls Left', fontsize=14)
        axes[0,1].set_xlabel('Total Runs')
        axes[0,1].set_ylabel('Balls Left')
        plt.colorbar(scatter, ax=axes[0,1], label='Won')
        
        # Wickets distribution by outcome
        self.train_df.groupby('won')['wickets'].hist(alpha=0.7, bins=10, ax=axes[1,0])
        axes[1,0].set_title('Wickets Distribution by Outcome', fontsize=14)
        axes[1,0].set_xlabel('Wickets Lost')
        axes[1,0].legend(['Lost', 'Won'])
        
        # Target vs Current Score
        axes[1,1].scatter(self.train_df['target'], 
                         self.train_df['total_runs'], 
                         c=self.train_df['won'], 
                         cmap='RdYlGn', alpha=0.6)
        axes[1,1].plot([0, 200], [0, 200], 'k--', alpha=0.5, label='Equal line')
        axes[1,1].set_title('Target vs Current Score', fontsize=14)
        axes[1,1].set_xlabel('Target Score')
        axes[1,1].set_ylabel('Current Score')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('eda_plots/comprehensive_eda.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.train_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Heatmap', fontsize=16)
        plt.tight_layout()
        plt.savefig('eda_plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Box plots for key features
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        key_features = ['total_runs', 'wickets', 'balls_left']
        for i, feature in enumerate(key_features):
            self.train_df.boxplot(column=feature, by='won', ax=axes[i])
            axes[i].set_title(f'{feature.replace("_", " ").title()} by Outcome')
            axes[i].set_xlabel('Won (0=Loss, 1=Win)')
        
        plt.suptitle('Feature Distributions by Match Outcome', fontsize=16)
        plt.tight_layout()
        plt.savefig('eda_plots/feature_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ EDA visualizations saved to 'eda_plots/' directory")
    
    def _document_insights(self):
        """Document key insights from EDA"""
        insights = []
        
        # Class balance insight
        if 'won' in self.train_df.columns:
            win_rate = self.train_df['won'].mean()
            if win_rate < 0.4 or win_rate > 0.6:
                insights.append(f"⚠️  Dataset is imbalanced (win rate: {win_rate:.1%})")
            else:
                insights.append(f"✓ Dataset is reasonably balanced (win rate: {win_rate:.1%})")
        
        # Data quality insights
        null_count = self.train_df.isnull().sum().sum()
        if null_count > 0:
            insights.append(f"⚠️  Found {null_count} null values requiring handling")
        else:
            insights.append("✓ No missing values detected")
        
        # Feature insights
        if 'total_runs' in self.train_df.columns and 'target' in self.train_df.columns:
            avg_chase_success = self.train_df[self.train_df['total_runs'] >= self.train_df['target']]['won'].mean()
            insights.append(f"📊 Teams reaching target score win {avg_chase_success:.1%} of matches")
        
        print("\n6. KEY INSIGHTS:")
        for insight in insights:
            print(f"   {insight}")
    
    def engineer_features(self):
        """Enhanced feature engineering with domain knowledge"""
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)
        
        def enhance_dataframe(df, is_test=False):
            df = df.copy()
            
            # Clean data
            df['balls_left'] = df['balls_left'].clip(lower=0)
            
            # Basic engineered features
            df['required_runs'] = df['target'] - df['total_runs']
            df['wickets_remaining'] = 10 - df['wickets']
            
            # Rate calculations
            overs_left = df['balls_left'] / 6
            overs_bowled = (120 - df['balls_left']) / 6
            overs_bowled = overs_bowled.replace(0, 1e-5)  # Avoid division by zero
            
            df['required_rr'] = np.where(
                df['balls_left'] > 0,
                df['required_runs'] / overs_left,
                np.where(df['required_runs'] > 0, np.inf, 0)
            )
            df['current_rr'] = df['total_runs'] / overs_bowled
            
            # BONUS FEATURES (New sophisticated features)
            
            # 1. Pressure Index: Combines required run rate with wickets situation
            df['pressure_index'] = (df['required_rr'] * (11 - df['wickets_remaining'])) / 10
            
            # 2. Game State Feature: Categorizes the match situation
            df['run_rate_difference'] = df['required_rr'] - df['current_rr']
            
            # 3. Balls per wicket remaining (batting depth indicator)
            df['balls_per_wicket_remaining'] = np.where(
                df['wickets_remaining'] > 0,
                df['balls_left'] / df['wickets_remaining'],
                0
            )
            
            # 4. Win probability based on runs needed vs balls left (simple heuristic)
            max_realistic_rr = 12  # Maximum realistic run rate
            df['theoretical_max_runs'] = df['balls_left'] * (max_realistic_rr / 6)
            df['win_feasibility'] = np.where(
                df['required_runs'] <= df['theoretical_max_runs'],
                1 - (df['required_runs'] / df['theoretical_max_runs']),
                0
            )
            
            # 5. Match phase (early, middle, death overs)
            total_overs = 20
            overs_completed = (120 - df['balls_left']) / 6
            df['match_phase'] = np.where(
                overs_completed <= 6, 0,  # Powerplay
                np.where(overs_completed <= 15, 1, 2)  # Middle overs, Death overs
            )
            
            # Handle infinite and extreme values
            df = df.replace([np.inf, -np.inf], np.nan)
            df['required_rr'] = df['required_rr'].fillna(0).clip(0, 36)  # Max 36 runs per over
            df['pressure_index'] = df['pressure_index'].fillna(0).clip(0, 100)
            
            return df
        
        # Apply feature engineering
        self.train_df = enhance_dataframe(self.train_df)
        self.test_df = enhance_dataframe(self.test_df, is_test=True)
        
        # Define feature set
        self.feature_names = [
            'total_runs', 'wickets', 'target', 'balls_left',
            'required_runs', 'wickets_remaining', 'required_rr', 'current_rr',
            'pressure_index', 'run_rate_difference', 'balls_per_wicket_remaining',
            'win_feasibility', 'match_phase'
        ]
        
        print(f"✓ Feature engineering completed. Total features: {len(self.feature_names)}")
        print("New features created:")
        new_features = ['pressure_index', 'run_rate_difference', 'balls_per_wicket_remaining', 
                       'win_feasibility', 'match_phase']
        for feature in new_features:
            print(f"  • {feature}")
        
        return self
    
    def prepare_data(self):
        """Clean and prepare data for modeling"""
        # Remove null values and duplicates
        initial_shape = self.train_df.shape[0]
        self.train_df = self.train_df.dropna().drop_duplicates()
        self.test_df = self.test_df.dropna().drop_duplicates()
        
        print(f"\nData cleaning: {initial_shape - self.train_df.shape[0]} rows removed")
        print(f"Final training data shape: {self.train_df.shape}")
        
        # Prepare features and target
        X = self.train_df[self.feature_names]
        y = self.train_df['won']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data with stratification
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Validation set: {self.X_val.shape}")
        
        return self
    
    def train_models(self):
        """Train multiple models with hyperparameter tuning"""
        print("\n" + "="*50)
        print("MODEL TRAINING & COMPARISON")
        print("="*50)
        
        # Define models with explanations
        model_configs = {
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'class_weight': ['balanced', None],
                    'solver': ['liblinear', 'lbfgs']
                },
                'reasoning': 'Linear baseline model, interpretable coefficients, good for balanced datasets'
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'class_weight': ['balanced', None]
                },
                'reasoning': 'Ensemble method, handles non-linear relationships, feature importance available'
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0],
                    'scale_pos_weight': [1, 2]  # For imbalanced data
                },
                'reasoning': 'Gradient boosting, excellent performance, handles missing values well'
            }
        }
        
        # Cross-validation strategy
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print(f"Cross-validation strategy: {cv_strategy.__class__.__name__} with {cv_strategy.n_splits} folds")
        
        results = {}
        
        for name, config in model_configs.items():
            print(f"\n🔄 Training {name}...")
            print(f"Reasoning: {config['reasoning']}")
            
            # Hyperparameter tuning
            grid_search = GridSearchCV(
                config['model'], 
                config['params'],
                cv=cv_strategy,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                best_model, self.X_train, self.y_train, 
                cv=cv_strategy, scoring='roc_auc'
            )
            
            # Validation predictions
            y_pred = best_model.predict(self.X_val)
            y_prob = best_model.predict_proba(self.X_val)[:, 1]
            
            # Calculate metrics
            metrics = {
                'Best_Params': grid_search.best_params_,
                'CV_AUC_Mean': cv_scores.mean(),
                'CV_AUC_Std': cv_scores.std(),
                'Val_Accuracy': accuracy_score(self.y_val, y_pred),
                'Val_AUC': roc_auc_score(self.y_val, y_prob),
                'Val_LogLoss': log_loss(self.y_val, y_prob)
            }
            
            results[name] = metrics
            self.models[name] = best_model
            
            print(f"✓ {name} completed:")
            print(f"  Best params: {grid_search.best_params_}")
            print(f"  CV AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"  Validation AUC: {metrics['Val_AUC']:.4f}")
        
        # Store results and find best model
        self.results_df = pd.DataFrame(results).T
        self.best_model_name = self.results_df['Val_AUC'].idxmax()
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🏆 Best Model: {self.best_model_name} (AUC: {self.results_df.loc[self.best_model_name, 'Val_AUC']:.4f})")
        
        return self
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Results comparison table
        print("\nModel Comparison:")
        comparison_cols = ['CV_AUC_Mean', 'CV_AUC_Std', 'Val_Accuracy', 'Val_AUC', 'Val_LogLoss']
        print(self.results_df[comparison_cols].round(4))
        
        # Detailed evaluation of best model
        print(f"\n📊 Detailed Evaluation - {self.best_model_name}:")
        
        y_pred = self.best_model.predict(self.X_val)
        y_prob = self.best_model.predict_proba(self.X_val)[:, 1]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_val, y_pred))
        
        # Create evaluation plots
        self._create_evaluation_plots(y_prob)
        
        # Feature importance for tree-based models
        if hasattr(self.best_model, 'feature_importances_'):
            self._plot_feature_importance()
        
        return self
    
    def _create_evaluation_plots(self, y_prob):
        """Create comprehensive evaluation visualizations"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(self.y_val, y_prob)
        auc_score = roc_auc_score(self.y_val, y_prob)
        
        axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title(f'ROC Curve - {self.best_model_name}')
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)
        
        # 2. Confusion Matrix
        cm = confusion_matrix(self.y_val, self.best_model.predict(self.X_val))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
        axes[1].set_title('Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        # 3. Probability Distribution
        win_probs = y_prob[self.y_val == 1]
        loss_probs = y_prob[self.y_val == 0]
        
        axes[2].hist(loss_probs, bins=30, alpha=0.7, label='Actual Loss', color='red', density=True)
        axes[2].hist(win_probs, bins=30, alpha=0.7, label='Actual Win', color='green', density=True)
        axes[2].set_xlabel('Predicted Win Probability')
        axes[2].set_ylabel('Density')
        axes[2].set_title('Probability Distribution by Actual Outcome')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_results/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Evaluation plots saved to 'model_results/model_evaluation.png'")
    
    def _plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        
        importances = self.best_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
        plt.title(f'Feature Importance - {self.best_model_name}', fontsize=14)
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('model_results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Feature importance plot saved")
        print("\nTop 5 Most Important Features:")
        top_features = feature_importance_df.tail().iloc[::-1]
        for _, row in top_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    def save_model(self):
        """Save the best model and preprocessing components"""
        joblib.dump(self.best_model, 'best_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        
        # Save model metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'validation_auc': self.results_df.loc[self.best_model_name, 'Val_AUC'],
            'model_params': self.best_model.get_params()
        }
        
        import json
        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\n✅ Model artifacts saved:")
        print("  • best_model.pkl")
        print("  • scaler.pkl") 
        print("  • model_metadata.json")
        
        return self
    
    def predict_test_data(self):
        """Generate predictions for test data"""
        X_test = self.test_df[self.feature_names]
        X_test_scaled = self.scaler.transform(X_test)
        
        predictions = self.best_model.predict(X_test_scaled)
        probabilities = self.best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Add predictions to test dataframe
        results_df = self.test_df.copy()
        results_df['predicted_won'] = predictions
        results_df['win_probability'] = probabilities
        
        # Save predictions
        results_df.to_csv('data/predictions.csv', index=False)
        print(f"\n✅ Test predictions saved to 'data/predictions.csv'")
        print(f"Predicted win rate: {predictions.mean():.1%}")
        
        return results_df

# Main execution
if __name__ == "__main__":
    # Initialize predictor
    predictor = CricketWinPredictor()
    
    # Execute full pipeline
    (predictor
     .load_data()
     .exploratory_data_analysis()
     .engineer_features()
     .prepare_data()
     .train_models()
     .evaluate_models()
     .save_model()
     .predict_test_data())
    
    print("\n🎉 Cricket Win Prediction Pipeline Completed Successfully!")
    print("\nGenerated Artifacts:")
    print("📁 eda_plots/ - Exploratory data analysis visualizations")
    print("📁 model_results/ - Model evaluation plots")  
    print("📄 predictions.csv - Test set predictions")
    print("🤖 best_model.pkl - Trained model")
    print("⚙️  scaler.pkl - Feature scaler")
    print("📋 model_metadata.json - Model details")
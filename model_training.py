"""Model training module with ensemble learning."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Flatten, Dense
import pickle
import os
from config import RANDOM_STATE, MODEL_SAVE_DIR


class EmotionEnsembleModel:
    """Ensemble model for emotion recognition with multi-layer architecture."""
    
    def __init__(self, random_state=RANDOM_STATE):
        
        self.random_state = random_state
        self.rf_clf = None
        self.xgb_clf = None
        self.rnn_model = None
        self.rf_layer2 = None
        self.nb_layer2 = None
        self.final_meta_clf = None
        
    def _build_base_classifiers(self, X_train, y_train):
        
        print("Training base classifiers...")
        
        # 1. Random Forest
        print("  - Training Random Forest...")
        self.rf_clf = RandomForestClassifier(n_estimators=150, random_state=self.random_state)
        self.rf_clf.fit(X_train, y_train)
        
        # 2. XGBoost
        print("  - Training XGBoost...")
        self.xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                                     random_state=self.random_state)
        self.xgb_clf.fit(X_train, y_train)
        
        # 3. Recurrent Neural Network
        print("  - Training RNN...")
        
        # Determine number of classes
        n_classes = len(np.unique(y_train))
        
        self.rnn_model = Sequential([
            SimpleRNN(64, activation='relu', return_sequences=True, 
                     input_shape=(X_train.shape[1], 1)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(n_classes, activation='softmax')
        ])
        self.rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
                              metrics=['accuracy'])
        
        # Reshape data for RNN
        X_train_rnn = np.expand_dims(X_train, axis=-1)
        self.rnn_model.fit(X_train_rnn, y_train, epochs=15, batch_size=32, verbose=0)
        
        print("Base classifiers trained successfully!")
    
    def _get_base_predictions(self, X_test):
        """Get predictions from base classifiers.
        
        Args:
            X_test: Test features
        
        Returns:
            Dictionary of predictions from each base classifier
        """
        rf_preds = self.rf_clf.predict(X_test)
        xgb_preds = self.xgb_clf.predict(X_test)
        
        X_test_rnn = np.expand_dims(X_test, axis=-1)
        rnn_preds = np.argmax(self.rnn_model.predict(X_test_rnn, verbose=0), axis=1)
        
        return {
            'rf': rf_preds,
            'xgb': xgb_preds,
            'rnn': rnn_preds
        }
    
    def _build_meta_classifiers(self, base_preds, y_test):
        
        print("\nTraining meta classifiers...")
        
        # First-level majority voting
        ensemble_preds = np.array([base_preds['rf'], base_preds['xgb'], 
                                   base_preds['rnn']])
        final_preds = np.round(np.mean(ensemble_preds, axis=0)).astype(int)
        
        # Create meta features
        meta_features = np.column_stack((base_preds['rf'], base_preds['xgb'], 
                                        base_preds['rnn'], final_preds))
        
        # Random Forest as meta classifier
        print("  - Training Random Forest meta-classifier...")
        self.rf_layer2 = RandomForestClassifier(n_estimators=200, random_state=self.random_state)
        self.rf_layer2.fit(meta_features, y_test)
        rf_layer2_preds = self.rf_layer2.predict(meta_features)
        
        # Naïve Bayes as meta classifier
        print("  - Training Naïve Bayes meta-classifier...")
        self.nb_layer2 = GaussianNB()
        self.nb_layer2.fit(meta_features, y_test)
        nb_layer2_preds = self.nb_layer2.predict(meta_features)
        
        # Final meta classifier (Voting)
        print("  - Training final voting classifier...")
        meta_input = np.column_stack((rf_layer2_preds, nb_layer2_preds))
        
        self.final_meta_clf = VotingClassifier(
            estimators=[('rf', RandomForestClassifier(n_estimators=200, random_state=self.random_state)), 
                       ('nb', GaussianNB())], 
            voting='hard'
        )
        self.final_meta_clf.fit(meta_input, y_test)
        
        print("Meta classifiers trained successfully!")
    
    def fit(self, X_train, X_test, y_train, y_test):
        
        # Train base classifiers
        self._build_base_classifiers(X_train, y_train)
        
        # Get base predictions on test set
        base_preds = self._get_base_predictions(X_test)
        
        # Train meta classifiers
        self._build_meta_classifiers(base_preds, y_test)
    
    def predict(self, X_test):
        
        # Get base predictions
        base_preds = self._get_base_predictions(X_test)
        
        # First-level majority voting
        ensemble_preds = np.array([base_preds['rf'], base_preds['xgb'], 
                                   base_preds['rnn']])
        final_preds = np.round(np.mean(ensemble_preds, axis=0)).astype(int)
        
        # Create meta features
        meta_features = np.column_stack((base_preds['rf'], base_preds['xgb'], 
                                        base_preds['rnn'], final_preds))
        
        # Get meta predictions
        rf_layer2_preds = self.rf_layer2.predict(meta_features)
        nb_layer2_preds = self.nb_layer2.predict(meta_features)
        
        # Final prediction
        meta_input = np.column_stack((rf_layer2_preds, nb_layer2_preds))
        return self.final_meta_clf.predict(meta_input)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: True labels (encoded)
        
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\nFinal Ensemble Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        
        # Try to get label names for better readability
        try:
            from data_preprocessing import decode_labels, label_encoder
            if hasattr(label_encoder, 'classes_'):
                target_names = label_encoder.classes_
                print(classification_report(y_test, predictions, target_names=target_names))
            else:
                print(classification_report(y_test, predictions))
        except:
            print(classification_report(y_test, predictions))
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'classification_report': classification_report(y_test, predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, predictions)
        }
    
    def save(self, directory=MODEL_SAVE_DIR):
        
        os.makedirs(directory, exist_ok=True)
        
        # Save sklearn models
        with open(os.path.join(directory, 'rf_clf.pkl'), 'wb') as f:
            pickle.dump(self.rf_clf, f)
        with open(os.path.join(directory, 'xgb_clf.pkl'), 'wb') as f:
            pickle.dump(self.xgb_clf, f)
        with open(os.path.join(directory, 'rf_layer2.pkl'), 'wb') as f:
            pickle.dump(self.rf_layer2, f)
        with open(os.path.join(directory, 'nb_layer2.pkl'), 'wb') as f:
            pickle.dump(self.nb_layer2, f)
        with open(os.path.join(directory, 'final_meta_clf.pkl'), 'wb') as f:
            pickle.dump(self.final_meta_clf, f)
        
        # Save Keras model
        self.rnn_model.save(os.path.join(directory, 'rnn_model.h5'))
        
        print(f"\nModels saved to {directory}")
    
    def load(self, directory=MODEL_SAVE_DIR):
        
        from tensorflow.keras.models import load_model
        
        # Load sklearn models
        with open(os.path.join(directory, 'rf_clf.pkl'), 'rb') as f:
            self.rf_clf = pickle.load(f)
        with open(os.path.join(directory, 'xgb_clf.pkl'), 'rb') as f:
            self.xgb_clf = pickle.load(f)
        with open(os.path.join(directory, 'rf_layer2.pkl'), 'rb') as f:
            self.rf_layer2 = pickle.load(f)
        with open(os.path.join(directory, 'nb_layer2.pkl'), 'rb') as f:
            self.nb_layer2 = pickle.load(f)
        with open(os.path.join(directory, 'final_meta_clf.pkl'), 'rb') as f:
            self.final_meta_clf = pickle.load(f)
        
        # Load Keras model
        self.rnn_model = load_model(os.path.join(directory, 'rnn_model.h5'))
        
        print(f"\nModels loaded from {directory}")


def train_and_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test):
    
    print("=" * 60)
    print("Starting Ensemble Model Training")
    print("=" * 60)
    
    # Initialize model
    model = EmotionEnsembleModel()
    
    # Train model
    model.fit(X_train, X_test, y_train, y_test)
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    results = model.evaluate(X_test, y_test)
    
    # Save model
    model.save()
    
    return model, results


if __name__ == "__main__":
    from data_preprocessing import prepare_data_pipeline
    from config import OUTPUT_CSV_ORIGINAL, OUTPUT_CSV_BALANCED
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_pipeline(
        OUTPUT_CSV_ORIGINAL, 
        OUTPUT_CSV_BALANCED
    )
    
    # Train and evaluate
    model, results = train_and_evaluate_model(
        X_train, X_val, X_test, 
        y_train, y_val, y_test
    )

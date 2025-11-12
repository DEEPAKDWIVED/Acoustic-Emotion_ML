"""Main execution script for emotion recognition pipeline."""
import librosa
import argparse
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from config import OUTPUT_CSV_ORIGINAL, OUTPUT_CSV_BALANCED, MODEL_SAVE_DIR
from feature_extraction import extract_and_save_features
from data_preprocessing import prepare_data_pipeline
from model_training import train_and_evaluate_model, EmotionEnsembleModel


def main(args):
    
    print("=" * 80)
    print(" " * 20 + "EMOTION RECOGNITION PIPELINE")
    print("=" * 80)
    
    # Step 1: Feature Extraction
    if args.extract_features:
        print("\n" + "=" * 80)
        print("STEP 1: Feature Extraction")
        print("=" * 80)
        extract_and_save_features(OUTPUT_CSV_ORIGINAL)
    else:
        print("\nSkipping feature extraction (use --extract-features to enable)")
    
    # Step 2: Data Preprocessing
    print("\n" + "=" * 80)
    print("STEP 2: Data Preprocessing")
    print("=" * 80)
    
    if not os.path.exists(OUTPUT_CSV_ORIGINAL):
        print(f"Error: {OUTPUT_CSV_ORIGINAL} not found!")
        print("Please run with --extract-features flag first.")
        return
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_pipeline(
        OUTPUT_CSV_ORIGINAL, 
        OUTPUT_CSV_BALANCED,
        balance=args.balance
    )
    
    # Step 3: Model Training
    if args.train:
        print("\n" + "=" * 80)
        print("STEP 3: Model Training")
        print("=" * 80)
        model, results = train_and_evaluate_model(
            X_train, X_val, X_test,
            y_train, y_val, y_test
        )
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nFinal Accuracy: {results['accuracy']:.4f}")
        print(f"Models saved to: {MODEL_SAVE_DIR}")
        print(f"Balanced data saved to: {OUTPUT_CSV_BALANCED}")
    else:
        print("\nSkipping model training (use --train to enable)")
    
    # Step 4: Prediction (if model exists and prediction mode is enabled)
    if args.predict and os.path.exists(os.path.join(MODEL_SAVE_DIR, 'rf_clf.pkl')):
        print("\n" + "=" * 80)
        print("STEP 4: Making Predictions")
        print("=" * 80)
        
        model = EmotionEnsembleModel()
        model.load(MODEL_SAVE_DIR)
        
        predictions = model.predict(X_test)
        print(f"\nPredictions generated for {len(predictions)} samples")
        print(f"Sample predictions: {predictions[:10]}")


def parse_arguments():
    
    parser = argparse.ArgumentParser(
        description='Emotion Recognition Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --extract-features --balance --train
  
  # Only preprocessing and training
  python main.py --balance --train
  
  # Only training (assuming data is preprocessed)
  python main.py --train
  
  # Load model and make predictions
  python main.py --predict
        """
    )
    
    parser.add_argument(
        '--extract-features',
        action='store_true',
        help='Extract features from audio files'
    )
    
    parser.add_argument(
        '--balance',
        action='store_true',
        default=True,
        help='Balance dataset using SMOTE (default: True)'
    )
    
    parser.add_argument(
        '--no-balance',
        action='store_false',
        dest='balance',
        help='Do not balance dataset'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the ensemble model'
    )
    
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Make predictions using trained model'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # If no arguments provided, show help
    if not any([args.extract_features, args.train, args.predict]):
        print("No action specified. Use --help for usage information.")
        print("\nQuick start: python main.py --train")
        sys.exit(1)
    
    main(args)

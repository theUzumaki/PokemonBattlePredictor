"""
Example usage of the training pipeline for Pokemon Battle Predictor.

This script demonstrates how to:
1. Load battleline data (from battleline_extractor)
2. Train a model using PCA features
3. Evaluate the model
4. Make predictions on new battles
"""

import train
import battleline_extractor as be
from test_battleline_struct import example_battleline


def example_training():
    """Example: Train a model on battleline data."""
    
    # In production, you would get battleline from battleline_extractor
    # For this example, we use the test data
    battleline = example_battleline
    
    # Option 1: Use the full training pipeline (recommended)
    print("\n" + "="*80)
    print("OPTION 1: Full Training Pipeline")
    print("="*80)
    
    results = train.train_pipeline(
        battleline=battleline,
        n_components=10,              # Number of PCA components
        max_moves=4,                   # Max moves per Pokemon
        use_individual_pokemon=False,  # False = aggregated, True = individual
        model_type='ridge',            # 'linear', 'ridge', or 'lasso'
        alpha=1.0,                     # Regularization strength
        test_size=0.2,                 # 20% test split
        random_state=42,               # For reproducibility
        perform_cv=True,               # Perform cross-validation
        cv_folds=5,                    # 5-fold CV
        save_path='models/trained_model.pkl'
    )
    
    return results


def example_step_by_step():
    """Example: Step-by-step training process."""
    
    print("\n" + "="*80)
    print("OPTION 2: Step-by-Step Training")
    print("="*80)
    
    battleline = example_battleline
    
    # Step 1: Prepare data
    labels, battle_ids = train.prepare_data_from_battleline(battleline)
    print(f"Prepared {len(labels)} battle labels")
    
    # Step 2: Apply PCA
    pca_features, pca_model, scaler, original_features, feature_names = train.apply_pca_transformation(
        battleline=battleline,
        n_components=10,
        use_individual_pokemon=False
    )
    print(f"PCA features shape: {pca_features.shape}")
    
    # Step 3: Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        pca_features, labels, test_size=0.2, random_state=42
    )
    
    # Step 4: Train model
    model = train.train_linear_regression(X_train, y_train, model_type='linear')
    
    # Step 5: Evaluate
    train_metrics = train.evaluate_model(model, X_train, y_train, "Training")
    test_metrics = train.evaluate_model(model, X_test, y_test, "Test")
    
    # Step 6: Save
    train.save_model(model, pca_model, scaler, feature_names, 'models/manual_model.pkl')
    
    return {
        'model': model,
        'pca_model': pca_model,
        'scaler': scaler,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }


def example_prediction():
    """Example: Load a trained model and make predictions."""
    
    print("\n" + "="*80)
    print("OPTION 3: Making Predictions with Trained Model")
    print("="*80)
    
    # Load the saved model
    try:
        model_package = train.load_model('models/trained_model.pkl')
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model not found. Please run training first.")
        return
    
    # Get new battles to predict (in production, from battleline_extractor)
    battleline = example_battleline
    
    # Make predictions
    predictions = train.predict_battle_outcome(
        battleline=battleline,
        model_package=model_package,
        use_individual_pokemon=False,
        threshold=0.5
    )
    
    # Display predictions
    print("\nPredictions for battles:")
    for battle_id, pred in list(predictions.items())[:10]:  # Show first 10
        print(f"Battle {battle_id}:")
        print(f"  Win Probability: {pred['prediction_probability']:.4f}")
        print(f"  Predicted Outcome: {'WIN' if pred['prediction_binary'] == 1 else 'LOSS'}")
        print(f"  Confidence: {pred['confidence']:.4f}")
        
        # Compare with actual if available
        actual = battleline.battles[battle_id].win
        correct = "✓" if pred['prediction_binary'] == actual else "✗"
        print(f"  Actual: {'WIN' if actual == 1 else 'LOSS'} {correct}")
        print()


def compare_model_types():
    """Example: Compare different regression models."""
    
    print("\n" + "="*80)
    print("OPTION 4: Comparing Different Model Types")
    print("="*80)
    
    battleline = example_battleline
    
    model_configs = [
        {'name': 'Linear Regression', 'type': 'linear', 'alpha': None},
        {'name': 'Ridge (α=0.1)', 'type': 'ridge', 'alpha': 0.1},
        {'name': 'Ridge (α=1.0)', 'type': 'ridge', 'alpha': 1.0},
        {'name': 'Ridge (α=10.0)', 'type': 'ridge', 'alpha': 10.0},
        {'name': 'Lasso (α=0.01)', 'type': 'lasso', 'alpha': 0.01},
    ]
    
    results_summary = []
    
    for config in model_configs:
        print(f"\nTraining {config['name']}...")
        
        result = train.train_pipeline(
            battleline=battleline,
            n_components=10,
            model_type=config['type'],
            alpha=config['alpha'] if config['alpha'] else 1.0,
            test_size=0.2,
            random_state=42,
            perform_cv=False,  # Skip CV to save time
            save_path=None  # Don't save intermediate models
        )
        
        results_summary.append({
            'name': config['name'],
            'train_accuracy': result['train_metrics']['accuracy'],
            'test_accuracy': result['test_metrics']['accuracy'],
            'train_r2': result['train_metrics']['r2'],
            'test_r2': result['test_metrics']['r2'],
            'train_rmse': result['train_metrics']['rmse'],
            'test_rmse': result['test_metrics']['rmse']
        })
    
    # Print comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'Test R²':<12} {'Test RMSE':<12}")
    print("-"*80)
    for res in results_summary:
        print(f"{res['name']:<20} {res['train_accuracy']:<12.4f} {res['test_accuracy']:<12.4f} "
              f"{res['test_r2']:<12.4f} {res['test_rmse']:<12.4f}")


def compare_pca_components():
    """Example: Compare models with different numbers of PCA components."""
    
    print("\n" + "="*80)
    print("OPTION 5: Comparing Different PCA Component Counts")
    print("="*80)
    
    battleline = example_battleline
    
    component_counts = [3, 5, 10, 15, 20]
    results_summary = []
    
    for n_comp in component_counts:
        print(f"\nTraining with {n_comp} PCA components...")
        
        try:
            result = train.train_pipeline(
                battleline=battleline,
                n_components=n_comp,
                model_type='ridge',
                alpha=1.0,
                test_size=0.2,
                random_state=42,
                perform_cv=False,
                save_path=None
            )
            
            results_summary.append({
                'n_components': n_comp,
                'train_accuracy': result['train_metrics']['accuracy'],
                'test_accuracy': result['test_metrics']['accuracy'],
                'test_r2': result['test_metrics']['r2'],
                'variance_explained': result['pca_model'].explained_variance_ratio_.sum()
            })
        except Exception as e:
            print(f"Error with {n_comp} components: {e}")
    
    # Print comparison
    print("\n" + "="*80)
    print("PCA COMPONENTS COMPARISON")
    print("="*80)
    print(f"{'Components':<12} {'Train Acc':<12} {'Test Acc':<12} {'Test R²':<12} {'Var. Explained':<15}")
    print("-"*80)
    for res in results_summary:
        print(f"{res['n_components']:<12} {res['train_accuracy']:<12.4f} {res['test_accuracy']:<12.4f} "
              f"{res['test_r2']:<12.4f} {res['variance_explained']:<15.4f}")


if __name__ == "__main__":
    import sys
    
    print("\nPokemon Battle Predictor - Training Examples\n")
    print("Choose an option:")
    print("1. Full training pipeline (recommended)")
    print("2. Step-by-step training")
    print("3. Make predictions with trained model")
    print("4. Compare different model types")
    print("5. Compare different PCA component counts")
    print("6. Run all examples")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter option (1-6): ").strip()
    
    if choice == '1':
        example_training()
    elif choice == '2':
        example_step_by_step()
    elif choice == '3':
        example_prediction()
    elif choice == '4':
        compare_model_types()
    elif choice == '5':
        compare_pca_components()
    elif choice == '6':
        print("\nRunning all examples...")
        example_training()
        example_step_by_step()
        example_prediction()
        compare_model_types()
        compare_pca_components()
    else:
        print("Invalid option. Running default (option 1)...")
        example_training()

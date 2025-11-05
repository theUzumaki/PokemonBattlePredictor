"""
Plotting utilities for Pokemon Battle Predictor training visualization.
Provides functions to plot confusion matrices, ROC curves, feature importance,
prediction distributions, and other useful training metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    average_precision_score
)
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import chronicle.logger as logger


# Set default style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix with percentages and counts.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save the figure
        title: Plot title
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    logger.log_info(f"Generating confusion matrix: {title}")
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both count and percentage
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    # Plot heatmap
    sns.heatmap(
        cm, 
        annot=annotations, 
        fmt='', 
        cmap='Blues', 
        cbar=True,
        square=True,
        ax=ax,
        xticklabels=['Loss (0)', 'Win (1)'],
        yticklabels=['Loss (0)', 'Win (1)']
    )
    
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.log_success(f"Confusion matrix saved to: {save_path}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        save_path: Optional path to save the figure
        title: Plot title
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    logger.log_info(f"Generating ROC curve: {title}")
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    logger.log_debug(f"ROC AUC score: {roc_auc:.4f}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.log_success(f"ROC curve saved to: {save_path}")
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot Precision-Recall curve with average precision score.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        save_path: Optional path to save the figure
        title: Plot title
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    logger.log_info(f"Generating precision-recall curve: {title}")
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    logger.log_debug(f"Average precision score: {avg_precision:.4f}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot PR curve
    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'PR curve (AP = {avg_precision:.4f})')
    
    # Calculate baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    ax.plot([0, 1], [baseline, baseline], color='navy', lw=2, 
            linestyle='--', label=f'Random Classifier (AP = {baseline:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.log_success(f"Precision-Recall curve saved to: {save_path}")
    
    return fig


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    save_path: Optional[Path] = None,
    title: str = "Prediction Probability Distribution",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot distribution of prediction probabilities for wins and losses.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        threshold: Classification threshold
        save_path: Optional path to save the figure
        title: Plot title
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    logger.log_info(f"Generating prediction distribution: {title}")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate probabilities by true label
    loss_proba = y_proba[y_true == 0]
    win_proba = y_proba[y_true == 1]
    logger.log_debug(f"Distribution - Losses: {len(loss_proba)}, Wins: {len(win_proba)}, Threshold: {threshold}")
    
    # Plot histograms
    ax.hist(loss_proba, bins=50, alpha=0.6, color='red', 
            label=f'True Losses (n={len(loss_proba)})', density=True)
    ax.hist(win_proba, bins=50, alpha=0.6, color='green', 
            label=f'True Wins (n={len(win_proba)})', density=True)
    
    # Add threshold line
    ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold}')
    
    ax.set_xlabel('Predicted Probability (Win)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper center', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.log_success(f"Prediction distribution saved to: {save_path}")
    
    return fig


def plot_feature_importance(
    feature_importance: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    save_path: Optional[Path] = None,
    title: str = "Top Feature Importances",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot top N most important features.
    
    Args:
        feature_importance: Array of feature importance scores
        feature_names: Optional list of feature names
        top_n: Number of top features to display
        save_path: Optional path to save the figure
        title: Plot title
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    logger.log_info(f"Generating feature importance plot: {title} (top {top_n})")
    
    # Get top N features
    indices = np.argsort(feature_importance)[::-1][:top_n]
    top_importance = feature_importance[indices]
    
    logger.log_debug(f"Top feature importance: {top_importance[0]:.4f}, Lowest in top {top_n}: {top_importance[-1]:.4f}")
    
    # Create feature names if not provided
    if feature_names is None:
        top_names = [f'Feature {i}' for i in indices]
    else:
        top_names = [feature_names[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_importance)))
    bars = ax.barh(range(len(top_importance)), top_importance, color=colors)
    
    ax.set_yticks(range(len(top_importance)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Highest importance at top
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_importance)):
        ax.text(val, i, f' {val:.4f}', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.log_success(f"Feature importance plot saved to: {save_path}")
    
    return fig


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Threshold Analysis",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot how accuracy, precision, recall change with different thresholds.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        save_path: Optional path to save the figure
        title: Plot title
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    logger.log_info(f"Generating threshold analysis: {title}")
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        # Calculate metrics
        tn = np.sum((y_true == 0) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot metrics
    ax.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
    ax.plot(thresholds, precisions, label='Precision', linewidth=2)
    ax.plot(thresholds, recalls, label='Recall', linewidth=2)
    ax.plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
    
    # Find optimal F1 threshold
    optimal_idx = np.argmax(accuracies)
    optimal_thresh = thresholds[optimal_idx]
    logger.log_debug(f"Optimal threshold: {optimal_thresh:.3f} with accuracy: {accuracies[optimal_idx]:.4f}")
    ax.axvline(x=optimal_thresh, color='red', linestyle='--', linewidth=2,
               label=f'Optimal Threshold = {optimal_thresh:.3f}')
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.log_success(f"Threshold analysis saved to: {save_path}")
    
    return fig


def plot_training_summary(
    result: Dict[str, Any],
    save_dir: Optional[Path] = None,
    model_name: str = "Model"
) -> Dict[str, plt.Figure]:
    """
    Create a comprehensive summary of all training metrics in one call.
    
    Args:
        result: Training result dictionary containing:
            - model: Trained model
            - X_train, X_test: Feature arrays
            - y_train, y_test: Label arrays
            - threshold: Classification threshold
            - feature_indices (optional): Selected feature indices
        save_dir: Optional directory to save all plots
        model_name: Name for the model (used in titles)
        
    Returns:
        Dictionary of figure objects
    """
    logger.log_section_header(f"Generating comprehensive training summary for: {model_name}")
    
    figures = {}
    
    # Get predictions
    model = result['model']
    X_train = result['X_train']
    X_test = result.get('X_test')
    y_train = result['y_train']
    y_test = result.get('y_test')
    threshold = result.get('threshold', 0.5)
    
    logger.log_debug(f"Training samples: {len(X_train)}, Test samples: {len(X_test) if X_test is not None else 0}", indentation_tabs=1)
    logger.log_debug(f"Classification threshold: {threshold}", indentation_tabs=1)
    
    # Train set predictions
    train_proba = model.predict_proba(X_train)[:, 1]
    train_pred = (train_proba >= threshold).astype(int)
    
    # Create save directory if specified
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.log_info(f"Saving plots to directory: {save_dir}", indentation_tabs=1)
    
    # 1. Training Confusion Matrix
    logger.log_subsection("Generating training set plots")
    figures['confusion_train'] = plot_confusion_matrix(
        y_train, train_pred,
        save_path=save_dir / 'confusion_matrix_train.png' if save_dir else None,
        title=f"{model_name} - Training Confusion Matrix"
    )
    
    # 2. Training ROC Curve
    figures['roc_train'] = plot_roc_curve(
        y_train, train_proba,
        save_path=save_dir / 'roc_curve_train.png' if save_dir else None,
        title=f"{model_name} - Training ROC Curve"
    )
    
    # 3. Training Precision-Recall Curve
    figures['pr_train'] = plot_precision_recall_curve(
        y_train, train_proba,
        save_path=save_dir / 'precision_recall_train.png' if save_dir else None,
        title=f"{model_name} - Training Precision-Recall Curve"
    )
    
    # 4. Training Prediction Distribution
    figures['dist_train'] = plot_prediction_distribution(
        y_train, train_proba, threshold=threshold,
        save_path=save_dir / 'prediction_distribution_train.png' if save_dir else None,
        title=f"{model_name} - Training Prediction Distribution"
    )
    
    # 5. Threshold Analysis
    figures['threshold'] = plot_threshold_analysis(
        y_train, train_proba,
        save_path=save_dir / 'threshold_analysis_train.png' if save_dir else None,
        title=f"{model_name} - Threshold Analysis"
    )
    
    # Test set plots (if available)
    if X_test is not None and len(X_test) > 0:
        logger.log_subsection("Generating test set plots")
        test_proba = model.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= threshold).astype(int)
        
        figures['confusion_test'] = plot_confusion_matrix(
            y_test, test_pred,
            save_path=save_dir / 'confusion_matrix_test.png' if save_dir else None,
            title=f"{model_name} - Test Confusion Matrix"
        )
        
        figures['roc_test'] = plot_roc_curve(
            y_test, test_proba,
            save_path=save_dir / 'roc_curve_test.png' if save_dir else None,
            title=f"{model_name} - Test ROC Curve"
        )
        
        figures['pr_test'] = plot_precision_recall_curve(
            y_test, test_proba,
            save_path=save_dir / 'precision_recall_test.png' if save_dir else None,
            title=f"{model_name} - Test Precision-Recall Curve"
        )
        
        figures['dist_test'] = plot_prediction_distribution(
            y_test, test_proba, threshold=threshold,
            save_path=save_dir / 'prediction_distribution_test.png' if save_dir else None,
            title=f"{model_name} - Test Prediction Distribution"
        )
    
    # 6. Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        logger.log_subsection("Generating feature importance plot")
        figures['importance'] = plot_feature_importance(
            model.feature_importances_,
            top_n=20,
            save_path=save_dir / 'feature_importance.png' if save_dir else None,
            title=f"{model_name} - Top 20 Feature Importances"
        )
    elif hasattr(model, 'estimators_'):
        # For ensemble models, average feature importances
        logger.log_subsection("Generating ensemble feature importance plot")
        importances = []
        for estimator in model.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                importances.append(estimator.feature_importances_)
        if importances:
            avg_importance = np.mean(importances, axis=0)
            logger.log_debug(f"Averaged feature importances from {len(importances)} estimators", indentation_tabs=1)
            figures['importance'] = plot_feature_importance(
                avg_importance,
                top_n=20,
                save_path=save_dir / 'feature_importance.png' if save_dir else None,
                title=f"{model_name} - Average Top 20 Feature Importances"
            )
    
    logger.log_newline()
    logger.log_success(f"Generated {len(figures)} plots successfully!", newline_before=1, newline_after=1)
    
    # Close all figures to free memory (optional - comment out if you want to display)
    # plt.close('all')
    
    return figures
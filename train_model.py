"""
Music Popularity Regression - Model Training & Evaluation
==========================================================
Trains a Multiple Linear Regression model on music features
to predict popularity score (0-100).

Pipeline:
  1. Load CSV data
  2. EDA with visualizations
  3. Feature scaling (StandardScaler)
  4. Train/Test split (80/20)
  5. Train LinearRegression model
  6. Evaluate (R², Adjusted R², MAE, MSE, RMSE)
  7. Save model + scaler
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
import joblib


def load_data(path="data/music_data.csv"):
    """Load the music dataset."""
    df = pd.read_csv(path)
    print(f"✅ Loaded {len(df)} samples from {path}")
    print(f"   Columns: {list(df.columns)}")
    return df


def perform_eda(df, output_dir="static/images"):
    """Generate EDA visualizations and save as PNG."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("darkgrid")
    plt.rcParams.update({
        'figure.facecolor': '#1a1a2e',
        'axes.facecolor': '#16213e',
        'text.color': '#e0e0e0',
        'axes.labelcolor': '#e0e0e0',
        'xtick.color': '#e0e0e0',
        'ytick.color': '#e0e0e0',
    })
    
    features = ['duration_min', 'tempo_bpm', 'energy', 'danceability', 'loudness_db']
    
    # 1. Correlation Heatmap
    print("   📈 Creating correlation heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt='.2f',
        cmap='coolwarm', center=0, square=True,
        linewidths=1, ax=ax,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Distributions
    print("   📊 Creating feature distributions...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    all_cols = features + ['popularity']
    colors = ['#00d2ff', '#7928ca', '#ff0080', '#ff6b35', '#00f5d4', '#ffd700']
    
    for i, (col, color) in enumerate(zip(all_cols, colors)):
        axes[i].hist(df[col], bins=25, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
    
    fig.suptitle('Feature Distributions', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Scatter plots: Each feature vs Popularity
    print("   📉 Creating scatter plots...")
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    for i, (feat, color) in enumerate(zip(features, colors[:5])):
        axes[i].scatter(df[feat], df['popularity'], alpha=0.4, s=20, color=color, edgecolors='white', linewidth=0.3)
        # Add trend line
        z = np.polyfit(df[feat], df['popularity'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
        axes[i].plot(x_line, p(x_line), color='#ffd700', linewidth=2, linestyle='--')
        axes[i].set_xlabel(feat, fontsize=11)
        axes[i].set_ylabel('Popularity', fontsize=11)
        axes[i].set_title(f'{feat} vs Popularity', fontsize=12, fontweight='bold')
    
    fig.suptitle('Features vs Popularity', fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Actual vs Predicted (will be updated after training)
    print("   ✅ EDA visualizations saved!")


def train_and_evaluate(df):
    """Train multiple linear regression and evaluate."""
    features = ['duration_min', 'tempo_bpm', 'energy', 'danceability', 'loudness_db']
    target = 'popularity'
    
    X = df[features].values
    y = df[target].values
    
    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n📐 Train/Test Split:")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   ✅ Features scaled with StandardScaler")
    
    # Train Model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    print("   ✅ Multiple Linear Regression model trained!")
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Evaluation Metrics
    n = X_test.shape[0]
    p = X_test.shape[1]
    
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    adj_r2 = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    
    # Model coefficients
    coef_dict = {feat: round(float(coef), 4) for feat, coef in zip(features, model.coef_)}
    
    evaluation = {
        'r2_train': round(float(r2_train), 4),
        'r2_test': round(float(r2_test), 4),
        'adjusted_r2': round(float(adj_r2), 4),
        'mae': round(float(mae), 4),
        'mse': round(float(mse), 4),
        'rmse': round(float(rmse), 4),
        'intercept': round(float(model.intercept_), 4),
        'coefficients': coef_dict,
        'features': features,
        'n_train': int(X_train.shape[0]),
        'n_test': int(X_test.shape[0])
    }
    
    print("\n📊 Model Evaluation:")
    print("-" * 50)
    print(f"   R² (Train):     {r2_train:.4f}")
    print(f"   R² (Test):      {r2_test:.4f}")
    print(f"   Adjusted R²:    {adj_r2:.4f}")
    print(f"   MAE:            {mae:.4f}")
    print(f"   MSE:            {mse:.4f}")
    print(f"   RMSE:           {rmse:.4f}")
    print(f"\n   Intercept: {model.intercept_:.4f}")
    print("   Coefficients:")
    for feat, coef in coef_dict.items():
        print(f"     {feat:15s}: {coef:+.4f}")
    
    # Generate Actual vs Predicted plot
    print("\n   📈 Creating Actual vs Predicted plot...")
    os.makedirs("static/images", exist_ok=True)
    
    plt.rcParams.update({
        'figure.facecolor': '#1a1a2e',
        'axes.facecolor': '#16213e',
        'text.color': '#e0e0e0',
        'axes.labelcolor': '#e0e0e0',
        'xtick.color': '#e0e0e0',
        'ytick.color': '#e0e0e0',
    })
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_test_pred, alpha=0.6, s=30, color='#00d2ff', edgecolors='white', linewidth=0.3, label='Test Data')
    ax.scatter(y_train, y_train_pred, alpha=0.2, s=15, color='#7928ca', label='Train Data')
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Popularity', fontsize=13)
    ax.set_ylabel('Predicted Popularity', fontsize=13)
    ax.set_title(f'Actual vs Predicted (R² = {r2_test:.4f})', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('static/images/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Residual plot
    fig, ax = plt.subplots(figsize=(10, 5))
    residuals = y_test - y_test_pred
    ax.scatter(y_test_pred, residuals, alpha=0.6, s=30, color='#ff0080', edgecolors='white', linewidth=0.3)
    ax.axhline(y=0, color='#ffd700', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Popularity', fontsize=13)
    ax.set_ylabel('Residuals', fontsize=13)
    ax.set_title('Residual Plot', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/images/residual_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return model, scaler, evaluation


def save_model(model, scaler, evaluation, model_dir="model"):
    """Save model, scaler, and evaluation metrics."""
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    joblib.dump(model, f'{model_dir}/regression_model.joblib')
    print(f"\n💾 Model saved to: {model_dir}/regression_model.joblib")
    
    # Save scaler
    joblib.dump(scaler, f'{model_dir}/scaler.joblib')
    print(f"💾 Scaler saved to: {model_dir}/scaler.joblib")
    
    # Save evaluation metrics
    with open(f'{model_dir}/evaluation.json', 'w') as f:
        json.dump(evaluation, f, indent=2)
    print(f"💾 Evaluation saved to: {model_dir}/evaluation.json")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("🎵 Music Popularity Regression - Model Training")
    print("=" * 60)
    
    # 1. Load data
    print("\n📂 Step 1: Loading Data...")
    df = load_data()
    
    # 2. EDA
    print("\n🔍 Step 2: Exploratory Data Analysis...")
    perform_eda(df)
    
    # 3. Train & Evaluate
    print("\n🧠 Step 3: Training & Evaluating Model...")
    model, scaler, evaluation = train_and_evaluate(df)
    
    # 4. Save
    print("\n💾 Step 4: Saving Model & Artifacts...")
    save_model(model, scaler, evaluation)
    
    print("\n" + "=" * 60)
    print("✅ Training pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

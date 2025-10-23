import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
'''
    Features:
        - Stats
        - Weaknesses table
    Dynamic features:
        - Move Effectivness 
        - Effects
'''

def meanCenterFeature(X):
    X_mean = X.mean()
    return X - X_mean 
    
def NormalizeFeature(X):
    scaler = StandardScaler(with_mean=True, with_std=True) #this operation already subtracts the mean 
    X_std = scaler.fit_transform(X) #fit_trasform computes the mean and std of each feature in X and then applies the scaling formula (X - Xmean)/std
    return X_std
    
def performPca(n_components, normalized_X):
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=0)
    Z = pca.fit_transform(normalized_X) # performs the Pca riduction trasformation 
    evr = pca.explained_variance_ratio_
    return Z, evr, pca
def identifyLoadings(pca):
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_ratio_)
    return loadings
def plot(loadings, feature_names):
    n_components = loadings.shape[1]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        loadings,
        annot=True,
        cmap='coolwarm',
        xticklabels=[f'PC{i+1}' for i in range(n_components)],
        yticklabels=feature_names
    )
    plt.title('Feature Loadings (Importance of Features in Principal Components)')
    plt.xlabel('Principal Components')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
    
print("Mean Centering the dataset...")
train_df = "" # creates simple feature 

numeric_features = train_df.columns.difference(['battle_id', 'player_won'])
numerical_df = train_df[numeric_features]  # returns only numeric columns
print(numerical_df.head(2))

normalized = NormalizeFeature(numerical_df)

# Convert normalized array back to DataFrame
normalized_df = pd.DataFrame(normalized, columns=numerical_df.columns)
normalized_df = pd.concat(
    [normalized_df, train_df[['battle_id', 'player_won']]],
    axis=1
)

n_components = 2
print(f"Performing pca...")
Z, evr, pca = performPca(n_components,normalized)
loadings = identifyLoadings(pca)

# doing list(normalized_df) returns the lables of columns in the dataframe
print(list(normalized_df))
plot(loadings, list(normalized_df))

pca_df = pd.DataFrame(Z, columns=["PC1", "PC2"])
pca_df["player_won"] = train_df["player_won"].values

# --- 4️⃣ Plot PC1 vs PC2 ---
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=pca_df,
    x="PC1", y="PC2",
    hue="player_won",
    palette={0: "red", 1: "green"},
    alpha=0.7,
    s=80
)
plt.title("PCA Projection — PC1 vs PC2 (Colored by Battle Outcome)", fontsize=14)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend(title="Player Won", labels=["Loss", "Win"])
plt.grid(True)
plt.tight_layout()
plt.show()




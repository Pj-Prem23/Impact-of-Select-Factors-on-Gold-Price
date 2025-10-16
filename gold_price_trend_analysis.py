# ===============================================================
# 📊 A Study on Impact of Select Factors on the Price of Gold
# ===============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.seasonal import seasonal_decompose

# -----------------------------------------------
# 1️⃣ Load Dataset
# -----------------------------------------------
df = pd.read_csv('data/gold_price_factors.csv')

# Convert Month-Year column to datetime if present
if 'Month-Year' in df.columns:
    df['Month-Year'] = pd.to_datetime(df['Month-Year'], errors='coerce')

print("✅ Data Loaded Successfully.")
print("\nData Overview:")
print(df.head())

# -----------------------------------------------
# 2️⃣ Handle Missing Values
# -----------------------------------------------
print("\nMissing Values in Dataset:")
print(df.isnull().sum())
df = df.dropna()

# -----------------------------------------------
# 3️⃣ Basic Statistics Summary
# -----------------------------------------------
print("\nSummary Statistics:\n", df.describe())

# -----------------------------------------------
# 4️⃣ Correlation Heatmap
# -----------------------------------------------
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='YlGnBu', linewidths=0.5)
plt.title("Correlation Between Gold Price and Economic Factors", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 5️⃣ Pair Plot for Visual Relationships
# -----------------------------------------------
sns.pairplot(df[['Gold_Price','USD_INR','EUR_INR','Crude_Oil','Repo_Rate','BSE_Sensex']])
plt.suptitle("Pairwise Relationships Between Variables", y=1.02)
plt.show()

# -----------------------------------------------
# 6️⃣ Regression Analysis & Scatter Plots
# -----------------------------------------------
def regression_analysis(x_col):
    X = df[[x_col]].values
    y = df['Gold_Price'].values
    
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    
    print(f"\n📈 Regression: Gold Price vs {x_col}")
    print(f"R² Score: {r2_score(y, y_pred):.3f}")
    print(f"Slope: {model.coef_[0]:.3f}, Intercept: {model.intercept_:.3f}")
    
    # Plot regression line
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=x_col, y='Gold_Price', data=df, alpha=0.6)
    plt.plot(X, y_pred, color='red', linewidth=2)
    plt.title(f"Gold Price vs {x_col}", fontsize=12, fontweight='bold')
    plt.xlabel(x_col)
    plt.ylabel("Gold Price (per 10 grams)")
    plt.grid(alpha=0.3)
    plt.show()
    
    return model.coef_[0], model.intercept_, r2_score(y, y_pred)

# Run analysis for each factor
factors = ['USD_INR', 'EUR_INR', 'Crude_Oil', 'Repo_Rate', 'BSE_Sensex']
results = []

for f in factors:
    slope, intercept, r2 = regression_analysis(f)
    results.append([f, slope, intercept, round(r2,3)])

# Show summary table
summary = pd.DataFrame(results, columns=['Factor', 'Slope', 'Intercept', 'R²'])
print("\n🔍 Regression Summary:")
print(summary)

# -----------------------------------------------
# 7️⃣ Distribution of Variables
# -----------------------------------------------
plt.figure(figsize=(10,6))
for i, col in enumerate(['Gold_Price','USD_INR','EUR_INR','Crude_Oil','Repo_Rate','BSE_Sensex'], start=1):
    plt.subplot(3,2,i)
    sns.histplot(df[col], kde=True, color='gold')
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 8️⃣ Time Series Trend & Seasonal Analysis
# -----------------------------------------------
if 'Month-Year' in df.columns:
    df = df.set_index('Month-Year').sort_index()
    
    # Plot gold price trend
    plt.figure(figsize=(12,5))
    plt.plot(df.index, df['Gold_Price'], color='darkorange')
    plt.title("Gold Price Trend (2001–2020)", fontsize=13, fontweight='bold')
    plt.xlabel("Year")
    plt.ylabel("Gold Price (per 10g)")
    plt.grid(alpha=0.4)
    plt.show()

    # Seasonal decomposition
    result = seasonal_decompose(df['Gold_Price'], model='additive', period=12)
    result.plot()
    plt.suptitle("Gold Price Trend, Seasonality, and Residuals", fontsize=12)
    plt.show()

print("\n✅ Statistical and Graphical Analysis Completed Successfully.")

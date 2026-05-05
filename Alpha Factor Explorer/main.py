import sys
import subprocess
import importlib

REQUIRED_PACKAGES = [
    'numpy',
    'pandas',
    'matplotlib',
    'seaborn',
    'scikit-learn',
]

def install_and_import(package):
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        importlib.invalidate_caches()

def auto_setup():
    for pkg in REQUIRED_PACKAGES:
        install_and_import(pkg)


# Ensure all dependencies are installed before proceeding
if __name__ == "__main__":
    auto_setup()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.linear_model import LinearRegression



# --- 2. DATA GENERATION ---
np.random.seed(42)
N = 200
mu = 0.0005
sigma = 0.01
prices = [100]
for _ in range(N):
    prices.append(prices[-1] * np.exp(np.random.normal(mu, sigma)))
prices = np.array(prices)
df = pd.DataFrame({'price': prices})
df['returns'] = np.log(df['price']).diff()
df['rolling_mean'] = df['price'].rolling(window=20).mean()
df['rolling_vol'] = df['returns'].rolling(window=20).std()

# --- 2. ALPHA FACTORS ---
df['momentum'] = df['returns'].rolling(window=10).mean()
df['mean_reversion'] = df['price'] - df['rolling_mean']
df['volatility_factor'] = df['rolling_vol']

# --- 3. MODELING ---
df['future_return'] = df['returns'].shift(-1)
factor_cols = ['momentum', 'mean_reversion', 'volatility_factor']
df_model = df.dropna(subset=factor_cols + ['future_return'])
X = df_model[factor_cols].values
y = df_model['future_return'].values
model = LinearRegression()
model.fit(X, y)
df_model['predicted_return'] = model.predict(X)



# --- 3D PLOTS ---
from matplotlib import cm

# 1. 3D Surface: Time vs. Price vs. Rolling Mean
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection='3d')
xs = np.arange(len(df_model))
ys = df_model['price'].values
zs = df_model['rolling_mean'].values
ax1.plot(xs, ys, zs, label='Price vs. Rolling Mean', color='b', alpha=0.7)
ax1.set_xlabel('Time')
ax1.set_ylabel('Price')
ax1.set_zlabel('Rolling Mean')
ax1.set_title('3D Line: Price & Rolling Mean')

# 2. 3D Scatter: Factors vs. Future Return
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(df_model['momentum'], df_model['mean_reversion'], df_model['future_return'],
            c=df_model['volatility_factor'], cmap=cm.coolwarm, s=20)
ax2.set_xlabel('Momentum')
ax2.set_ylabel('Mean Reversion')
ax2.set_zlabel('Future Return')
ax2.set_title('3D Scatter: Factors vs. Future Return')

# 3. 3D Surface: Factor Mesh vs. Predicted Return
from scipy.interpolate import griddata
num = 40
xi = np.linspace(df_model['momentum'].min(), df_model['momentum'].max(), num)
yi = np.linspace(df_model['mean_reversion'].min(), df_model['mean_reversion'].max(), num)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((df_model['momentum'], df_model['mean_reversion']), df_model['predicted_return'], (xi, yi), method='cubic', fill_value=0)
fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111, projection='3d')
surf = ax3.plot_surface(xi, yi, zi, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.8)
ax3.set_xlabel('Momentum')
ax3.set_ylabel('Mean Reversion')
ax3.set_zlabel('Predicted Return')
ax3.set_title('3D Surface: Predicted Return by Factors')
fig3.colorbar(surf, shrink=0.5, aspect=10, label='Predicted Return')

# --- 5. CRITICAL DISPLAY FIX ---
plt.tight_layout()
plt.show()

print("Done. Alpha analysis complete.")

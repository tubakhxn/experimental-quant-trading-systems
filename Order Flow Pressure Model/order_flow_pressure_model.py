import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# 1. DATA GENERATION
np.random.seed(42)
N = 300  # Number of time steps
time = np.arange(N)

# Simulate buy/sell volumes with some autocorrelation
buy_volume = np.abs(np.random.normal(100, 20, N) + np.convolve(np.random.normal(0, 10, N), np.ones(5)/5, mode='same'))
sell_volume = np.abs(np.random.normal(100, 20, N) + np.convolve(np.random.normal(0, 10, N), np.ones(5)/5, mode='same'))

# Imbalance
imbalance = buy_volume - sell_volume

# Simulate price as a random walk with drift from imbalance
price = [100.0]
for i in range(1, N):
    drift = 0.02 * np.tanh(imbalance[i-1]/100)  # Nonlinear drift from imbalance
    noise = np.random.normal(0, 0.5)
    price.append(price[-1] + drift + noise)
price = np.array(price)

# 2. MODELING: Regression to predict next price move from imbalance
X = imbalance[:-1].reshape(-1, 1)
y = price[1:] - price[:-1]
reg = LinearRegression().fit(X, y)
pred_move = reg.predict(imbalance.reshape(-1, 1))

# 3. TRADING LOGIC
threshold = np.percentile(np.abs(imbalance), 80)  # Only trade on strong signals
positions = np.zeros(N)
for i in range(N):
    if imbalance[i] > threshold:
        positions[i] = 1  # Buy
    elif imbalance[i] < -threshold:
        positions[i] = -1  # Sell
    else:
        positions[i] = 0  # Flat

# Calculate PnL
pnl = np.zeros(N)
for i in range(1, N):
    pnl[i] = pnl[i-1] + positions[i-1] * (price[i] - price[i-1])

# 4. VISUALIZATION
plt.style.use('seaborn-v0_8-darkgrid')
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Price
axs[0].plot(time, price, color='royalblue', linewidth=2, label='Price')
axs[0].set_ylabel('Price')
axs[0].set_title('Order Flow Pressure Trading Model')
axs[0].legend()

# Imbalance
axs[1].plot(time, imbalance, color='darkorange', linewidth=2, label='Imbalance')
axs[1].axhline(threshold, color='green', linestyle='--', alpha=0.5, label='Buy Threshold')
axs[1].axhline(-threshold, color='red', linestyle='--', alpha=0.5, label='Sell Threshold')
axs[1].set_ylabel('Imbalance')
axs[1].legend()

# PnL
axs[2].plot(time, pnl, color='purple', linewidth=2, label='PnL')
axs[2].set_ylabel('PnL')
axs[2].set_xlabel('Time')
axs[2].legend()

plt.tight_layout()
plt.show()

# 5. 3D VISUALIZATION
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Smooth scatter
ax.plot_trisurf(time, imbalance, price, cmap=cm.viridis, alpha=0.8, linewidth=0.2, antialiased=True)
ax.set_xlabel('Time')
ax.set_ylabel('Imbalance')
ax.set_zlabel('Price')
ax.set_title('3D Surface: Time vs Imbalance vs Price')

plt.tight_layout()
plt.show()

# 6. Print regression stats
print(f"Regression coefficient (imbalance → price move): {reg.coef_[0]:.4f}")
print(f"Regression intercept: {reg.intercept_:.4f}")
print(f"Final PnL: {pnl[-1]:.2f}")

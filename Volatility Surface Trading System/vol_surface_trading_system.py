import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.interpolate import griddata

# 1. DATA GENERATION
np.random.seed(42)
N = 200
T = 50

time = np.linspace(0, T, N)
volatility = np.abs(np.random.normal(0.2, 0.05, N))
price = np.zeros(N)
price[0] = 100
for i in range(1, N):
    drift = 0.05 * (T / N)
    shock = volatility[i] * np.random.normal(0, 1)
    price[i] = price[i-1] * np.exp(drift - 0.5 * volatility[i]**2 + shock * np.sqrt(T/N))

# 2. MODELING (Polynomial Regression)
X = np.column_stack((time, volatility))
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
reg = LinearRegression()
reg.fit(X_poly, price)

# For surface: create meshgrid
grid_t = np.linspace(time.min(), time.max(), 40)
grid_v = np.linspace(volatility.min(), volatility.max(), 40)
T_grid, V_grid = np.meshgrid(grid_t, grid_v)
X_grid = np.column_stack((T_grid.ravel(), V_grid.ravel()))
X_grid_poly = poly.transform(X_grid)
P_grid = reg.predict(X_grid_poly).reshape(T_grid.shape)

# 3. 3D VISUALIZATION
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(121, projection='3d')
surf = ax.plot_surface(T_grid, V_grid, P_grid, cmap=cm.coolwarm, alpha=0.85, edgecolor='none', antialiased=True)

# Contour projection on bottom
contour = ax.contourf(T_grid, V_grid, P_grid, zdir='z', offset=P_grid.min()-10, cmap=cm.coolwarm, alpha=0.6)

ax.set_xlabel('Time')
ax.set_ylabel('Volatility')
ax.set_zlabel('Price')
ax.set_title('Volatility Surface (Price = f(Time, Volatility))')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

# 4. TRADING LOGIC
# Compute gradient
dP_dt, dP_dv = np.gradient(P_grid, grid_t, grid_v, edge_order=2)
# Use the gradient at the actual data points
from scipy.spatial import cKDTree
kdtree = cKDTree(np.column_stack((T_grid.ravel(), V_grid.ravel())))
_, idxs = kdtree.query(np.column_stack((time, volatility)))
slope = dP_dt.ravel()[idxs]
signals = np.where(slope > 0, 1, -1)  # 1=buy, -1=sell

# Simulate PnL
pnl = np.zeros(N)
position = 0
for i in range(1, N):
    if signals[i] != signals[i-1]:
        position = signals[i]
    pnl[i] = pnl[i-1] + position * (price[i] - price[i-1])

# 5. ADDITIONAL PLOTS
ax2 = fig.add_subplot(222)
ax2.plot(time, price, label='Price', color='tab:blue')
ax2.set_xlabel('Time')
ax2.set_ylabel('Price')
ax2.set_title('Price vs Time')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(224)
ax3.plot(time, pnl, label='PnL', color='tab:green')
ax3.set_xlabel('Time')
ax3.set_ylabel('PnL')
ax3.set_title('PnL Curve')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

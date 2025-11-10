import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize

# load the data file
data = pd.read_csv("xy_data.csv")
x_obs, y_obs = data["x"].values, data["y"].values
N = len(x_obs)

# create a uniform t range (6 to 60) matching number of points
t = np.linspace(6, 60, N)

# define the parametric model
def model(params):
    theta_deg, M, X = params
    theta = np.deg2rad(theta_deg)
    x_fit = t * np.cos(theta) - np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(theta) + X
    y_fit = 42 + t * np.sin(theta) + np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(theta)
    return x_fit, y_fit

# loss function (L1 = sum of absolute errors)
def l1_loss(params):
    x_fit, y_fit = model(params)
    return np.sum(np.abs(x_obs - x_fit) + np.abs(y_obs - y_fit))

# parameter search ranges
bounds = [
    (0.1, 179.9),   # theta (in degrees)
    (-0.05, 0.05),  # exponential decay/growth term
    (0.0, 120.0),   # x offset
]

print("Running global search (this may take a minute)...")
res_global = differential_evolution(
    l1_loss,
    bounds,
    maxiter=60,
    popsize=20,
    tol=1e-7,
    polish=False
)
print("Global search result:", res_global.x, "L1 =", res_global.fun)

print("\nRefining locally with Nelder–Mead...")
res = minimize(l1_loss, res_global.x, method="Nelder-Mead", options={"maxiter": 10000})
theta_opt, M_opt, X_opt = res.x
L1_sum = res.fun

# calculate fitted curve and residuals
x_fit, y_fit = model(res.x)
abs_err = np.abs(x_obs - x_fit) + np.abs(y_obs - y_fit)
df_resid = pd.DataFrame({
    "t": t,
    "x_obs": x_obs,
    "y_obs": y_obs,
    "x_fit": x_fit,
    "y_fit": y_fit,
    "abs_err": abs_err
})
df_resid.to_csv("fit_residuals.csv", index=False)

# plot observed and fitted data
plt.figure(figsize=(8, 6))
plt.scatter(x_obs, y_obs, s=8, label="Observed data", color="dodgerblue")
plt.plot(x_fit, y_fit, color="crimson", linewidth=2, label="Fitted curve")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Parametric Curve Fitting (L1 Minimization)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("param_fit.png", dpi=300)
plt.close()

# print final results
print("\n--- FINAL RESULTS ---")
print(f"theta (deg): {theta_opt:.6f}")
print(f"M          : {M_opt:.8f}")
print(f"X          : {X_opt:.6f}")
print(f"L1 sum     : {L1_sum:.6f}")

# desmos format output
desmos = (
    f"( t*cos({theta_opt}*pi/180) - e^({M_opt}*abs(t))*sin(0.3t)*sin({theta_opt}*pi/180) + {X_opt}, "
    f"42 + t*sin({theta_opt}*pi/180) + e^({M_opt}*abs(t))*sin(0.3t)*cos({theta_opt}*pi/180) )"
)
print("\nPaste this into Desmos (domain 6 ≤ t ≤ 60):\n")
print(desmos)

# LaTeX version for report
latex_str = (
    f"\\left(t\\cos({theta_opt:.6f}) - e^{{{M_opt:.8f}|t|}}\\sin(0.3t)\\sin({theta_opt:.6f}) + {X_opt:.6f},\\;"
    f"42 + t\\sin({theta_opt:.6f}) + e^{{{M_opt:.8f}|t|}}\\sin(0.3t)\\cos({theta_opt:.6f})\\right)"
)
print("\nSubmission string (LaTeX style):\n")
print(latex_str)

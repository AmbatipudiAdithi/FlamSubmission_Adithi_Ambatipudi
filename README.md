# FlamSubmission_Adithi_Ambatipudi
Parameter fitting tool for trajectory optimization. Uses global + local search (Nelder–Mead) to estimate best-fit values for θ, M, and X that minimize error (L1 norm). Includes plotting utilities and Desmos-ready output for visualization.
Overview

This project fits a nonlinear parametric model to the given xy_data.csv using an L1 loss minimization approach.
The goal is to estimate three parameters — θ (rotation angle), M (exponential scale), and X (horizontal offset) — that best match the observed curve.

The optimization is done in two stages:

Global search using Differential Evolution (to avoid local minima).

Local refinement using the Nelder–Mead algorithm for final fine-tuning.
| File                | Description                                                                             |
| ------------------- | --------------------------------------------------------------------------------------- |
| `fit_params.py`     | Main Python script that performs the parameter fitting, plotting, and output.           |
| `xy_data.csv`       | Provided dataset containing observed x–y points.                                        |
| `fit_residuals.csv` | Output file containing observed and fitted points along with absolute error per sample. |
| `param_fit.png`     | Generated plot comparing observed and fitted curves.                                    |
| `README.md`         | This documentation file.                                                                |

| Parameter         | Symbol | Value       |
| ----------------- | ------ | ----------- |
| Rotation Angle    | θ      | 28.118425°  |
| Exponential Scale | M      | 0.02138895  |
| Horizontal Offset | X      | 54.899815   |
| L1 Sum            | —      | 37865.09384 |

Desmos equation
( t*cos(28.118424567925153*pi/180) - exp(0.021388946565839576*abs(t))*sin(0.3*t)*sin(28.118424567925153*pi/180) + 54.899815445379964 , 42 + t*sin(28.118424567925153*pi/180) + exp(0.021388946565839576*abs(t))*sin(0.3*t)*cos(28.118424567925153*pi/180) )
set:

t-min = 6

t-max = 60


Summary of Method

Load the data from xy_data.csv.

Define the parametric model equations for x(t) and y(t).

Compute the L1 loss (sum of absolute differences) between observed and fitted data.

Perform global optimization (Differential Evolution).

Refine locally with Nelder–Mead to minimize the L1 loss further.

Plot and export final fitted curve, residuals, and parameters.

Key Takeaways

L1 loss is more robust than L2 when dealing with non-Gaussian noise or outliers.

The two-step optimization (global + local) ensures reliable convergence.

The final parameters produce a visually and numerically accurate fit to the data.

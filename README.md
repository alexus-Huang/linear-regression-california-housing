# Linear Regression from Scratch — California Housing

A mini project implementing linear regression and mini-batch stochastic gradient descent (SGD) 
from scratch using NumPy, without relying on sklearn's built-in model. Built to practice the 
core mechanics behind gradient descent: loss functions, gradients, learning rates, batching, 
and convergence.

## Dataset
California Housing dataset (from `sklearn.datasets`), using a single feature (`MedInc`, median 
income) to predict `MedHouseVal` (median house value).

## What's covered
- Linear regression equation and manual derivation of gradients
- MSE, MAE, and RMSE loss functions implemented by hand
- Mini-batch SGD training loop with epochs
- Feature scaling and its effect on gradient descent stability
- Loss curve visualization showing convergence
- Learning rate comparison (stable vs. unstable vs. divergent)
- Convexity visualization of the MSE loss surface
- Final fitted regression line overlaid on the data

## Setup
\`\`\`bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
\`\`\`

Open `notebooks/linear_regression.ipynb` in Jupyter or VS Code.

## What I learned
- Unscaled features can cause gradient descent to become unstable or diverge, even at 
  moderate learning rates.
- A single feature's predictive power has a "floor" — the model converges almost 
  immediately once it hits the best fit achievable with that one feature.
- Learning rate selection involves a tradeoff between convergence speed and stability.
- 

## Main Purpose
- Build linear regression from scratch to understand how gradient descent works because it is the foundation under every ML model.
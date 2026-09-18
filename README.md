# ames-house-price-regression
KNIME workflow comparing Linear Regression vs. a tuned Regression Tree on 1,300 Ames, Iowa properties: Linear Regression wins the test set (Adj. R² 0.763 vs 0.747) despite the Tree scoring better on cross-validation, plus a fixed reproducibility bug where the shipped tree config didn't match its own saved results.

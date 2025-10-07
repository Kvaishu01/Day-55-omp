import streamlit as st
from sklearn.datasets import make_regression
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Streamlit UI
st.set_page_config(page_title="Orthogonal Matching Pursuit (OMP)", page_icon="🧠", layout="centered")

st.title("🧠 Orthogonal Matching Pursuit (OMP) Regression")
st.write("""
Orthogonal Matching Pursuit (OMP) is a sparse linear model that selects features one at a time to best predict the target.
""")

# Generate synthetic regression data
X, y = make_regression(n_samples=200, n_features=100, n_informative=10, noise=15, random_state=42)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Sidebar controls
st.sidebar.header("Model Hyperparameters")
n_nonzero_coefs = st.sidebar.slider("Number of non-zero coefficients", 1, 30, 10)
fit_intercept = st.sidebar.checkbox("Fit Intercept", True)

# Train model
model = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, fit_intercept=fit_intercept)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Results
st.subheader("📊 Model Performance")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Plot actual vs predicted
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(y_test, y_pred, alpha=0.7, color='teal')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Actual vs Predicted Values")
st.pyplot(fig)

# Display non-zero coefficients
st.subheader("🔍 Selected Features (Non-Zero Coefficients)")
coef_indices = np.where(model.coef_ != 0)[0]
st.write(f"Model selected **{len(coef_indices)}** features out of 100.")
st.write("Feature indices:", coef_indices.tolist())

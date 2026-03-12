import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io

# ============================================================
# Streamlit Page Config
# ============================================================
st.set_page_config(page_title="Polymath SMR Model", layout="wide")
st.title("SMR Model: Catalyst Weight Optimization")
st.markdown("Calculates the required catalyst weight based on Temperature and Conversion targets.")

# ============================================================
# Sidebar Inputs
# ============================================================
st.sidebar.header("Model Parameters")
FA0 = st.sidebar.number_input("FA0 (mol/s)", value=1.059977, format="%.6f")
ebs = st.sidebar.number_input("Epsilon (ebs)", value=0.15, format="%.2f")
P_CH40 = st.sidebar.number_input("Inlet P_CH4 (atm)", value=2.25, format="%.2f")

st.sidebar.header("Reference Points")
T_ref = st.sidebar.number_input("Reference Temp (K)", value=1173)
X_ref = st.sidebar.slider("Reference Conversion (X)", 0.1, 0.99, 0.78)

st.sidebar.header("Grid Settings")
T_min = st.sidebar.number_input("Min Temp (K)", value=900)
T_max = st.sidebar.number_input("Max Temp (K)", value=1500)

st.sidebar.header("Visualization Settings")
# This slider prevents the 3D plot from being ruined by infinite spikes
max_w_plot = st.sidebar.number_input("Max Weight for 3D Plot (kg)", value=5000, step=500, help="Clips the visual spike at low temps/high conversions.")

# ============================================================
# Core Math Functions
# ============================================================
def rate(X, T):
    k1 = (1.17e12) * np.exp(-2.401e5 / (8.314 * T))
    k_CH4 = (6.65e-4) * np.exp(0.38280 / (8.314e-5 * T))
    k_CO  = (8.23e-5) * np.exp(0.7065  / (8.314e-5 * T))
    k_H2  = (6.12e-9) * np.exp(0.829   / (8.314e-5 * T))
    k_H2O = (1.77e5) * np.exp(-0.8868  / (8.314e-5 * T))
    K_eq  = np.exp((-26830 / T) + 30.114)

    N = 1 + ebs * X

    P_CH4 = P_CH40 * (1 - X) / N
    P_H2O = P_CH40 * (3.12 - X) / N
    P_H2  = P_CH40 * (3.0 * X) / N
    P_CO  = P_CH40 * X / N

    P_H2 = np.maximum(P_H2, 1e-10) # Protect denominator

    Omega = 1 + k_CO * P_CO + k_H2 * P_H2 + k_CH4 * P_CH4 + (k_H2O * P_H2O) / P_H2
    numerator = P_CH4 * P_H2O - (P_H2**3 * P_CO) / K_eq
    denominator = (P_H2**2.5) * (Omega**2)

    rA = k1 * numerator / denominator
    rA = np.where(np.abs(rA) < 1e-20, 1e-20, rA) # Numerical protection

    return rA

def catalyst_weight_curve(X_array, T):
    rA = rate(X_array, T)
    dwdx = FA0 / rA
    w = np.zeros_like(X_array)
    dx = np.diff(X_array)
    w[1:] = np.cumsum(0.5 * (dwdx[1:] + dwdx[:-1]) * dx)
    return w / 0.01

# ============================================================
# Generate Data
# ============================================================
T_range = np.linspace(T_min, T_max, 50)
X_array = np.linspace(1e-4, X_ref, 250)

T_grid, X_grid = np.meshgrid(T_range, X_array)
W_surface = np.zeros_like(T_grid)

rows = []
for j, T in enumerate(T_range):
    W_curve = catalyst_weight_curve(X_array, T)
    W_surface[:, j] = W_curve
    for i, x_val in enumerate(X_array):
        rows.append({
            "Temperature_K": float(T),
            "Conversion_X": float(x_val),
            "Catalyst_Weight_kg": float(W_curve[i])
        })

surface_df = pd.DataFrame(rows)

slice_temperatures = [1123, 1143, 1173, 1223, 1273]
slice_df = pd.DataFrame({"Conversion_X": X_array})
for T in slice_temperatures:
    slice_df[f"W_kg_at_{T}K"] = catalyst_weight_curve(X_array, T)

W_ref_curve = catalyst_weight_curve(X_array, T_ref)
W_ref = W_ref_curve[-1]

# ============================================================
# Helper Function for Excel Downloads
# ============================================================
def to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# ============================================================
# Web Dashboard Output
# ============================================================
col1, col2 = st.columns(2)
with col1:
    st.metric("Reference Temperature", f"{T_ref} K")
with col2:
    st.metric("Required Catalyst Weight at Ref Point", f"{W_ref:.3f} kg")

st.divider()

# --- 3D Plot ---
st.subheader("Interactive 3D Surface: T vs X vs W")

# We clip the visual data here to stop the massive rate spikes from ruining the scale
W_surface_clipped = np.clip(W_surface, 0, max_w_plot)

fig_3d = go.Figure(data=[go.Surface(
    x=T_grid, 
    y=X_grid, 
    z=W_surface_clipped, 
    colorscale="Plasma",
    cmin=0,
    cmax=max_w_plot
)])

# Add Reference Point
fig_3d.add_trace(go.Scatter3d(
    x=[T_ref], y=[X_ref], z=[min(W_ref, max_w_plot)],
    mode='markers+text', marker=dict(size=8, color='red'),
    text=[f"Ref: {W_ref:.2f} kg"], textposition="top center", name="Reference Point"
))

fig_3d.update_layout(
    scene=dict(
        xaxis_title="Temperature (K)", 
        yaxis_title="Conversion, X", 
        zaxis_title="Catalyst Weight (kg)",
        zaxis=dict(range=[0, max_w_plot])
    ),
    height=700, margin=dict(l=0, r=0, b=0, t=30)
)
st.plotly_chart(fig_3d, use_container_width=True)

st.download_button(
    label="Download Raw 3D Data (Unclipped Excel)",
    data=to_excel_bytes(surface_df),
    file_name="smr_3d_surface_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.divider()

# --- 2D Slices Plot ---
st.subheader("Reference Temperature Slices")
fig_2d, ax = plt.subplots(figsize=(10, 6))
for T in slice_temperatures:
    # Also clip the 2D slices so they remain readable
    w_curve = catalyst_weight_curve(X_array, T)
    ax.plot(X_array, np.clip(w_curve, 0, max_w_plot), linewidth=2, label=f"{T} K")

ax.scatter([X_ref], [min(W_ref, max_w_plot)], c="red", s=70, zorder=5, label="Reference Point")
ax.set_ylim(0, max_w_plot)
ax.set_xlabel("Conversion, X")
ax.set_ylabel("Catalyst Weight (kg)")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig_2d)

st.download_button(
    label="Download 2D Slices Data (Excel)",
    data=to_excel_bytes(slice_df),
    file_name="smr_reference_temperature_slices.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

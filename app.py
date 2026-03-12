import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import pandas as pd
import io

# ============================================================
# Streamlit Page Config
# ============================================================
st.set_page_config(page_title="SMR Intrinsic Kinetics Model", layout="wide")
st.title("SMR Reactor Operational Surface (Xu & Froment Kinetics)")
st.markdown("Interactive 3D surface using `scipy.integrate.solve_ivp`.")

# ============================================================
# Sidebar Inputs
# ============================================================
st.sidebar.header("Model Parameters")
FA0 = st.sidebar.number_input("FA0 (kmol/s)", value=1.059977, format="%.6f")
P_CH40 = st.sidebar.number_input("Inlet P_CH4 (bar)", value=2.25, format="%.2f")
ebs = st.sidebar.number_input("Expansion Factor (ebs)", value=0.15, format="%.2f")
eta = st.sidebar.number_input("Effectiveness Factor (eta)", value=0.01, format="%.3f")

st.sidebar.header("Design Target")
T_op_C = st.sidebar.number_input("Design Temp (°C)", value=900)
Target_X = st.sidebar.slider("Target Conversion (X)", 0.10, 0.99, 0.78)
W_max = st.sidebar.number_input("Max Catalyst Weight (kg)", value=500000, step=10000)

# ============================================================
# Core Math Functions (Xu & Froment)
# ============================================================
def calculate_intrinsic_rate(X, T, ebs, P_CH40):
    X = np.clip(X, 1e-6, 0.9999) 
    R = 8.314
    
    N = 1.0 + (ebs * X)
    P_CH4 = P_CH40 * (1.0 - X) / N
    P_H2O = P_CH40 * (3.12 - X) / N
    P_H2  = P_CH40 * (3.0 * X) / N
    P_CO  = P_CH40 * (X) / N
    
    K1_eq = np.exp((-26830.0 / T) + 30.114)
    k1 = (1.17e12) * np.exp(-240100 / (R * T))
    
    k_CH4 = (6.65e-4) * np.exp(38280 / (R * T))
    k_CO  = (8.23e-5) * np.exp(70650 / (R * T))
    k_H2  = (6.12e-9) * np.exp(82900 / (R * T))
    k_H2O = (1.77e5)  * np.exp(-88680 / (R * T))
    
    Omega = 1.0 + k_CO*P_CO + k_H2*P_H2 + k_CH4*P_CH4 + (k_H2O*P_H2O) / P_H2
    
    numerator = P_CH4 * P_H2O - ((P_H2**3 * P_CO) / K1_eq)
    denominator = (P_H2**2.5) * (Omega**2)
    
    return k1 * numerator / denominator

def dX_dW(W, X_array, T, FA0, eta, ebs, P_CH40):
    X = X_array[0]
    if X >= 1.0: return [0.0]
    
    rA_intrinsic = calculate_intrinsic_rate(X, T, ebs, P_CH40)
    if rA_intrinsic <= 0: return [0.0]
    
    return [(eta * rA_intrinsic) / FA0]

# Terminal event to stop integration if conversion reaches 99.9% (Speeds up web app)
def hit_max_conversion(W, y, *args):
    return y[0] - 0.999
hit_max_conversion.terminal = True

# ============================================================
# Execute Calculations (Cached for Web Speed)
# ============================================================
@st.cache_data
def calculate_surface(T_op_C, Target_X, W_max, FA0, eta, ebs, P_CH40):
    T_op_K = T_op_C + 273.15
    
    # 1. Calculate specific operating point
    sol_op = solve_ivp(
        fun=dX_dW, t_span=(0, W_max), y0=[1e-6], 
        args=(T_op_K, FA0, eta, ebs, P_CH40), method='BDF',
        dense_output=True, events=hit_max_conversion, rtol=1e-4, atol=1e-7
    )
    
    W_op_profile = sol_op.t
    X_op_profile = sol_op.y[0]
    W_op_target = None
    if np.max(X_op_profile) >= Target_X:
        W_op_target = float(np.interp(Target_X, X_op_profile, W_op_profile))

    # 2. Generate 3D Surface
    T_range_C = np.linspace(500, 1000, 30)  
    W_eval = np.linspace(0, W_max, 60) 
    T_mesh, W_mesh = np.meshgrid(T_range_C, W_eval)
    X_mesh = np.zeros_like(T_mesh)

    for i, T_C in enumerate(T_range_C):
        T_K = T_C + 273.15
        sol = solve_ivp(
            fun=dX_dW, t_span=(0, W_max), y0=[1e-6], t_eval=W_eval,      
            args=(T_K, FA0, eta, ebs, P_CH40), method='BDF',
            events=hit_max_conversion, rtol=1e-4, atol=1e-7
        )
        X_mesh[:, i] = sol.y[0] * 100 

    return W_op_target, T_mesh, W_mesh, X_mesh

with st.spinner('Solving Differential Equations...'):
    W_op_target, T_mesh, W_mesh, X_mesh = calculate_surface(T_op_C, Target_X, W_max, FA0, eta, ebs, P_CH40)

# ============================================================
# Web Dashboard Output
# ============================================================
if W_op_target is not None:
    st.success(f"**Target Reached:** A catalyst weight of **{W_op_target:,.1f} kg** is required to hit {Target_X*100}% conversion at {T_op_C}°C.")
else:
    st.error(f"**Target Not Reached:** The reactor cannot reach {Target_X*100}% conversion at {T_op_C}°C within {W_max} kg of catalyst.")

st.divider()

# --- Interactive 3D Plot ---
st.subheader("Interactive 3D Surface: T vs W vs X")

fig = go.Figure(data=[go.Surface(
    x=T_mesh, y=W_mesh, z=X_mesh, 
    colorscale="Magma", 
    colorbar=dict(title="Conversion (%)")
)])

if W_op_target is not None:
    fig.add_trace(go.Scatter3d(
        x=[T_op_C], y=[W_op_target], z=[Target_X * 100],
        mode='markers+text', marker=dict(size=8, color='cyan', line=dict(color='black', width=2)),
        text=[f"Design Point<br>{T_op_C}°C<br>{W_op_target/1000:.1f} tons<br>{Target_X*100}%"], 
        textposition="top center", name="Design Point"
    ))
    # Vertical drop line to the floor
    fig.add_trace(go.Scatter3d(
        x=[T_op_C, T_op_C], y=[W_op_target, W_op_target], z=[0, Target_X * 100],
        mode='lines', line=dict(color='cyan', width=4, dash='dash'), showlegend=False
    ))

fig.update_layout(
    scene=dict(
        xaxis_title="Temperature (°C)", 
        yaxis_title="Catalyst Weight (kg)", 
        zaxis_title="Methane Conversion (%)",
        zaxis=dict(range=[0, 100])
    ),
    height=800, margin=dict(l=0, r=0, b=0, t=30)
)
st.plotly_chart(fig, use_container_width=True)

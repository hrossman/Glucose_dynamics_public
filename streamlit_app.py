import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from config import COLOR_G, COLOR_X, COLOR_RA
from glucose_ode_models import GOM_ODE

#-- STREAMLIT
# Init model
model = GOM_ODE()

param_min_val = -2000.
param_max_val = 2000.
new_params = {}

st.set_page_config(layout="wide")
with st.sidebar:
    # Title & Equation
    st.header(model.title)
    st.latex(model.equations)
    # params inputs
    for param_name, param_default_val in model.params.items():
        param_step = model.params_steps[param_name]
        new_params[param_name] = st.number_input(
            label=param_name, min_value=param_min_val, max_value=param_max_val,
            value=float(param_default_val), step=float(param_step), key=param_name, format="%f")


## SIMULATE
model.params = new_params
model.simulate()
dynamics_df = model.dynamics_df
t = model.t_eval

# # Plot outputs
# fig, axes = plt.subplots(3, 1, figsize=(6,4), sharex=True)
# ax = axes[0]
# ax.plot(t, dynamics_df['Ra'], lw=1, color=COLOR_RA, alpha=0.5, label='Glucose values')
# ax.fill_between(t, 0, dynamics_df['Ra'], color=COLOR_RA, alpha=0.4)
# ax.set(ylabel='Ra(t)')
# ax.grid()
# ax = axes[1]
# ax.plot(t, dynamics_df['G'], ls='-', lw=4, color=COLOR_G, alpha=0.5, label='Glucose values')
# ax.set(ylabel='G(t)')
# ax.grid()
# ax = axes[2]
# ax.plot(t, dynamics_df['X'], lw=4, color=COLOR_X, alpha=0.5, label='Glucose values')
# ax.set(ylabel='X(t)', xlabel='t')
# ax.grid()
# fig.tight_layout()
# st.write(print((model.dynamic_vars)))
fig, axes = model.plot_dynamic_vars()
st.pyplot(fig)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from config import COLOR_G, COLOR_X, COLOR_RA
from glucose_ode_models import GOM_ODE, GOM_ODE_OLD

#-- STREAMLIT
# Init model
model = GOM_ODE()
# model = GOM_ODE_OLD()


param_min_val = -99999.
param_max_val = 99999.
new_params = {}

st.set_page_config(layout="wide")

with st.sidebar:
    st.header(model.title)
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
fig, axes = model.plot_dynamic_vars()
st.pyplot(fig)

# Title & Equation
st.header(model.title)
st.latex(model.equations)

st.header('Auxilary variables')
fig, axes = model.plot_aux_vars()
st.pyplot(fig)

st.write(model.dynamics_df)
st.write(model.aux_vars_df)
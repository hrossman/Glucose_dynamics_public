import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import streamlit as st


def Ra_func_gaussian(t: float, H: float, T: float, W: float) -> float:
    return (H*np.exp(-((t-T)**2)/W))


def GOM_ODE(t, ys, Ra, Gb, theta1, theta2, theta3):
    G, X = ys
    dG = -G*X - theta3*(G-Gb) + Ra(t)
    dX = -theta1*X + theta2*(G-Gb)
    return [dG, dX]


def run_ode_sim(params, t_start, t_end, t_points=1000):
    # Unpack model params
    Gb = params['Gb']
    G0 = params['G0']
    X0 = params['X0']
    theta1 = params['theta1']
    theta2 = params['theta2']
    theta3 = params['theta3']
    H1 = params['H1']
    T1 = params['T1']
    W1 = params['W1']
    H2 = params['H2']
    T2 = params['T2']
    W2 = params['W2']

    y0 = [G0, X0]
    t_span = [t_start, t_end]
    t_eval = np.linspace(t_start, t_end, t_points)

    # Compute Ra input
    Ra_value = Ra_func_gaussian(t_eval, H1, T1, W1) + \
        Ra_func_gaussian(t_eval, H2, T2, W2)
    Ra = interp1d(x=t_eval, y=Ra_value, bounds_error=False)

    # Solve ODE
    sol = solve_ivp(GOM_ODE, t_span=t_span, y0=y0, t_eval=t_eval,
                    args=(Ra, Gb, theta1, theta2, theta3), dense_output=True)

    # Output to pd Dataframe
    t = t_eval
    z = sol.sol(t).T
    result_df = pd.DataFrame(z, columns=['G', 'X'])
    result_df['Ra'] = Ra(t)
    result_df['t'] = t

    return result_df


#-- STREAMLIT
# model params
t_mult = 0.01
def_params_L = {
      'theta1':1.5*t_mult,
      'theta2':0.5*t_mult**2,
      'theta3':0.1*t_mult,
      'G0': 5.7,
      'X0': 0,
      'Gb': 5.7,
}
def_params_R = {
      'H1':0.5,
      'T1':100,
      'W1':300, 
      'H2':0.05,
      'T2':100,
      'W2':120
}

step_params_L = {
      'theta1':0.1*t_mult,
      'theta2':0.01*t_mult**2,
      'theta3':0.01*t_mult,
      'G0': 0.1,
      'X0': 0.1,
      'Gb': 0.1,
}
step_params_R = {
      'H1':0.1,
      'T1':10,
      'W1':10, 
      'H2':0.01,
      'T2':10,
      'W2':10
}


param_min_val = -1000.
param_max_val = 1000.
params = {}

st.set_page_config(layout="wide")
col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    for param_name, param_default_val in def_params_L.items():
        param_step = step_params_L[param_name]
        params[param_name] = st.number_input(
            label=param_name, min_value=param_min_val, max_value=param_max_val,
            value=float(param_default_val), step=float(param_step), key=param_name, format="%f")

with col2:
    for param_name, param_default_val in def_params_R.items():
        param_step = step_params_R[param_name]
        params[param_name] = st.number_input(
            label=param_name, min_value=param_min_val, max_value=param_max_val,
            value=float(param_default_val), step=float(param_step), key=param_name, format="%f")


with col3:
    # simulation params
    t_start = 0
    t_end = 700
    t_points = 1000
    t = np.linspace(t_start, t_end, t_points)

    df = run_ode_sim(params, t_start, t_end, t_points)
    # st.write(df)
    # Plot outputs
    fig, axes = plt.subplots(3, 1)

    ax = axes[0]
    ax.plot(t, df['Ra'])
    ax.set(ylabel='Ra(t)', xlabel='t')
    ax.grid()

    ax = axes[1]
    ax.plot(t, df['G'])
    ax.set(ylabel='G(t)', xlabel='t')
    ax.grid()

    ax = axes[2]
    ax.plot(t, df['X'])
    ax.set(ylabel='X(t)', xlabel='t')
    ax.grid()

    fig.tight_layout()
    st.pyplot(fig)

    
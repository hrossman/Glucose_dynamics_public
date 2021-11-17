import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import streamlit as st
from config import SEED, FONTSZ, COLOR_G, COLOR_X, COLOR_RA, COLOR_FIT, COLOR_FOOD, COLOR_OGTT

def Ra_func_gaussian(t: float, H: float, T: float, W: float) -> float:
    return (H*np.exp(-((t-T)**2)/W))

def Ra_double_gaussian(t, params):
    # Unpack model params
    H1 = params['H1']
    T1 = params['T1']
    W1 = params['W1']
    H2 = params['H2']
    T2 = params['T2']
    W2 = params['W2']

    # Compute Ra input
    Ra_value = Ra_func_gaussian(t, H1, T1, W1) + \
        Ra_func_gaussian(t, H2, T2, W2)
    Ra = interp1d(x=t, y=Ra_value, bounds_error=False)
    return Ra


def GOM_ODE_func(t, ys, Ra, params):
    Gb = params['Gb']
    theta1 = params['theta1']
    theta2 = params['theta2']
    theta3 = params['theta3']
    G, X = ys
    dG = -G*X - theta3*(G-Gb) + Ra(t)
    dX = -theta1*X + theta2*(G-Gb)
    return [dG, dX]


#-- ODE class
class Glucose_ODE:
    def __init__(self) -> None:
        self.name = 'GOM'
        self.title = 'Glucose Only Model (GOM)'
        self.equations = r"""
        dG/dt = -G(t)X(t)-\theta_3\left(G(t)-G_b\right) \\
        dX/dt = -\theta_1X(t) +\theta_2X\left(G(t)-G_b\right) \\
        Ra(t) = H_1exp\left(\frac{-(t-T_1)^2}{W_1}\right) \\
              + H_2exp\left(\frac{-(t-T_2)^2}{W_2}\right)
        """
        # model params
        t_mult = 0.01
        self.params = {
            'theta1':1.5*t_mult,
            'theta2':0.5*t_mult**2,
            'theta3':0.1*t_mult,
            'G0': 5.7,
            'X0': 0,
            'Gb': 5.7,
            'H1':0.5,
            'T1':100,
            'W1':300, 
            'H2':0.05,
            'T2':100,
            'W2':120
        }

        # inital conditions vec
        self.y0 = [self.params['G0'], self.params['X0']]

        # simulation params
        self.t_start = 0
        self.t_end = 700
        self.t_points = 1000
        self.t_span = [self.t_start, self.t_end]
        self.t_eval = np.linspace(self.t_start, self.t_end, self.t_points)

        self.ode_func = GOM_ODE_func
        self.Ra_func = Ra_double_gaussian 

        # # Compute dynamics upon init
        # self.simulate() 


    def run_ode_sim(self):    
        # Solve ODE
        sol = solve_ivp(self.ode_func, t_span=self.t_span, y0=self.y0, t_eval=self.t_eval,
                        args=(self.Ra, self.params), dense_output=True)
        # Output to pd Dataframe
        z = sol.sol(self.t_eval).T
        dynamics_df = pd.DataFrame(z, columns=['G', 'X'])
        dynamics_df['Ra'] = self.Ra(self.t_eval)
        dynamics_df['t'] = self.t_eval
        return dynamics_df


    def simulate(self):
        # Compute Ra input
        self.Ra = self.Ra_func(self.t_eval, self.params)
        # Solve dynamics
        self.dynamics_df = self.run_ode_sim()


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

# col1, col2, col3 = st.columns([1, 1, 3])
# col1, col2= st.columns([2, 3])
# with col1:
with st.sidebar:
    # Init model
    model = Glucose_ODE()
    # # Title & Equation
    st.header(model.title)
    st.latex(model.equations)

    for param_name, param_default_val in def_params_L.items():
        param_step = step_params_L[param_name]
        params[param_name] = st.number_input(
            label=param_name, min_value=param_min_val, max_value=param_max_val,
            value=float(param_default_val), step=float(param_step), key=param_name, format="%f")

    for param_name, param_default_val in def_params_R.items():
        param_step = step_params_R[param_name]
        params[param_name] = st.number_input(
            label=param_name, min_value=param_min_val, max_value=param_max_val,
            value=float(param_default_val), step=float(param_step), key=param_name, format="%f")

# with col2:
#     for param_name, param_default_val in def_params_R.items():
#         param_step = step_params_R[param_name]
#         params[param_name] = st.number_input(
#             label=param_name, min_value=param_min_val, max_value=param_max_val,
#             value=float(param_default_val), step=float(param_step), key=param_name, format="%f")


# with col2:
## SIMULATE
model.params = params
model.simulate()
dynamics_df = model.dynamics_df
t = model.t_eval

# Plot outputs
fig, axes = plt.subplots(3, 1, figsize=(6,4), sharex=True)
ax = axes[0]
ax.plot(t, dynamics_df['Ra'], lw=1, color=COLOR_RA, alpha=0.5, label='Glucose values')
ax.fill_between(t, 0, dynamics_df['Ra'], color=COLOR_RA, alpha=0.4)
ax.set(ylabel='Ra(t)')
ax.grid()
ax = axes[1]
ax.plot(t, dynamics_df['G'], ls='-', lw=4, color=COLOR_G, alpha=0.5, label='Glucose values')
ax.set(ylabel='G(t)')
ax.grid()
ax = axes[2]
ax.plot(t, dynamics_df['X'], lw=4, color=COLOR_X, alpha=0.5, label='Glucose values')
ax.set(ylabel='X(t)', xlabel='t')
ax.grid()
fig.tight_layout()
st.pyplot(fig)

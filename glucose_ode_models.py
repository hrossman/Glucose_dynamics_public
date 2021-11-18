import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
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

# -- ODE class


class Glucose_ODE(object):
    def __init__(self) -> None:
        self.name = None
        self.title = None
        self.equations = None
        self.dynamic_vars = None
        self.dynamic_vars_labels = None
        self.dynamic_vars_colors = None
        self.params = None
        # initial condions
        self.y0 = None

        # simulation params
        self.t_start = 0
        self.t_end = 700
        self.t_points = 1000
        self.t_span = [self.t_start, self.t_end]
        self.t_eval = np.linspace(self.t_start, self.t_end, self.t_points)

        self.ode_func = None
        self.Ra_func = None

    def run_ode_sim(self):
        # Solve ODE
        sol = solve_ivp(self.ode_func, t_span=self.t_span, y0=self.y0, t_eval=self.t_eval,
                        args=(self.Ra, self.params), dense_output=True)
        # Output to pd Dataframe
        z = sol.sol(self.t_eval).T
        dynamics_df = pd.DataFrame(z, columns=self.dynamic_vars)
        dynamics_df['Ra'] = self.Ra(self.t_eval)
        dynamics_df['t'] = self.t_eval
        return dynamics_df

    def simulate(self):
        # Compute Ra input
        self.Ra = self.Ra_func(self.t_eval, self.params)
        # Solve dynamics
        self.dynamics_df = self.run_ode_sim()

    def plot_dynamic_vars(self):
        # Plot outputs
        n_vars = len(self.dynamic_vars)+1
        fig, axes = plt.subplots(n_vars, 1, figsize=(6, 0.6+n_vars), sharex=True)
        # Plot Ra input
        ax = axes[0]
        ax.plot(self.t_eval, self.dynamics_df['Ra'],
                lw=1, color=COLOR_RA, alpha=0.5, label='Ra')
        ax.fill_between(
            self.t_eval, 0, self.dynamics_df['Ra'], color=COLOR_RA, alpha=0.4)
        ax.set(ylabel='Ra(t)')
        ax.grid()
        # plot dynamic vars
        for i, var in enumerate(self.dynamic_vars):
            ax = axes[i+1]
            ax.plot(self.t_eval, self.dynamics_df[var], ls='-', lw=4,
                    color=self.dynamic_vars_colors[i], alpha=0.5, label=self.dynamic_vars_labels[i])
            ax.set(ylabel=f'{var}(t)')
            ax.grid()
        fig.tight_layout()
        return fig, axes


class GOM_ODE(Glucose_ODE):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'GOM'
        self.title = 'Glucose Only Model (GOM)'
        self.equations = r"""
        dG/dt = -G(t)X(t)-\theta_3\left(G(t)-G_b\right) \\
        dX/dt = -\theta_1X(t) +\theta_2X\left(G(t)-G_b\right) \\
        Ra(t) = H_1exp\left(\frac{-(t-T_1)^2}{W_1}\right) \\
                + H_2exp\left(\frac{-(t-T_2)^2}{W_2}\right)
        """
        self.dynamic_vars = ['G', 'X']
        self.dynamic_vars_labels = ['Glucose values', 'X values']
        self.dynamic_vars_colors = [COLOR_G, COLOR_X]
        # model params
        t_mult = 0.01
        self.params = {
            'theta1': 1.5*t_mult,
            'theta2': 0.5*t_mult**2,
            'theta3': 0.1*t_mult,
            'G0': 5.7,
            'X0': 0,
            'Gb': 5.7,
            'H1': 0.5,
            'T1': 100,
            'W1': 300,
            'H2': 0.05,
            'T2': 100,
            'W2': 120
        }
        # step sizes for streamlit
        self.params_steps = {
            'theta1': 0.2*t_mult,
            'theta2': 0.1*t_mult**2,
            'theta3': 0.1*t_mult,
            'G0': 0.5,
            'X0': 0.5,
            'Gb': 0.5,
            'H1': 0.5,
            'T1': 100,
            'W1': 50,
            'H2': 0.1,
            'T2': 100,
            'W2': 50
        }

        # inital conditions
        self.y0 = [self.params['G0'], self.params['X0']]

        self.ode_func = GOM_ODE_func
        self.Ra_func = Ra_double_gaussian

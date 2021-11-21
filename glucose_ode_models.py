import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from config import SEED, FONTSZ, COLOR_G, COLOR_X, COLOR_RA, COLOR_FIT, COLOR_FOOD, COLOR_OGTT


def Ra_double_gaussian(t, params):
    # Unpack model params
    H1 = params['H1']
    T1 = params['T1']
    W1 = params['W1']
    H2 = params['H2']
    T2 = params['T2']
    W2 = params['W2']

    # Compute Ra input
    def Ra_func_gaussian(t, H, T, W):
        return (H*np.exp(-((t-T)**2)/W))

    Ra_value = Ra_func_gaussian(t, H1, T1, W1) + \
        Ra_func_gaussian(t, H2, T2, W2)
    Ra = interp1d(x=t, y=Ra_value, bounds_error=False)
    return Ra


# -- Parent ODE class
class Glucose_ODE(object):
    def __init__(self, t_start=1, t_end=60*7) -> None:
        self.name = None
        self.title = None
        self.equations = None
        self.dynamic_vars = None
        self.dynamic_vars_labels = None
        self.dynamic_vars_colors = None
        self.params = None
        # initial condions
        self.y0 = None
        # ode function
        self.ode_func = None
        # input function
        self.Ra_func = None

        # simulation params
        self.t_start = t_start
        self.t_end = t_end
        self.t_points = self.t_end
        self.t_span = [self.t_start, self.t_end]
        self.t_eval = np.linspace(self.t_start, self.t_end, self.t_points)

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
        # add measurement error
        self.measured_df = pd.DataFrame({'t':self.t_eval,'G_measured':(self.dynamics_df['G'] + np.random.normal(loc=0,scale=self.params['lambda'], size=len(self.dynamics_df)))})
        self.measured_df = self.measured_df.iloc[::15, :]
        self.dynamics_df = self.dynamics_df.merge(self.measured_df, on='t', how='left')

    def plot_dynamic_vars(self):
        # Plot outputs
        n_vars = len(self.dynamic_vars)+1
        fig, axes = plt.subplots(n_vars, 1, figsize=(6, 0.8+n_vars), sharex=True)
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
        ax.set(xlabel='t (minutes)', xticks=np.arange(self.t_start-1,self.t_end+1, 60))

        # add measurements
        ax = axes[1]
        ax.scatter(self.t_eval, self.dynamics_df['G_measured'], s=20,
                    color=self.dynamic_vars_colors[0], alpha=0.5, label = 'G measured')

        fig.tight_layout()
        return fig, axes


#-- Specific model classes
class GOM_ODE_OLD(Glucose_ODE):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'GOM'
        self.title = 'Glucose Only Model (GOM)'
        self.equations = r"""
        dG/dt = -G(t)X(t)-\theta_3\left(G(t)-G_b\right) + Ra(t) \\
        dX/dt = -\theta_1X(t) +\theta_2X\left(G(t)-G_b\right) \\
        Ra(t) = H_1exp\left(\frac{-(t-T_1)^2}{W_1}\right)
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

        # ode function
        def ODE_func(t, ys, Ra, params):
            Gb = params['Gb']
            theta1 = params['theta1']
            theta2 = params['theta2']
            theta3 = params['theta3']
            G, X = ys
            dG = -G*X - theta3*(G-Gb) + Ra(t)
            dX = -theta1*X + theta2*(G-Gb)
            return [dG, dX]
        self.ode_func = ODE_func
      
        # input function
        self.Ra_func = Ra_double_gaussian

        
class GOM_ODE(Glucose_ODE):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'GOM'
        self.title = 'Glucose Only Model (GOM)'
        self.equations = r"""
        dG/dt = -G(t)X(t)-p_1[G(t)-G_b] \\
                + Ra(t)/V \\
        dX/dt = -p_2[X(t)-S_GZ(t)] \\
        Z(t) = \frac{G(t)-G_b}{1+exp[-\alpha(G(t)-G_b)]} + \beta Ra/V
        """
        self.dynamic_vars = ['G', 'X']
        self.dynamic_vars_labels = ['Glucose values', 'X values']
        self.dynamic_vars_colors = [COLOR_G, COLOR_X]
        # model params
        self.params = {
            'p1': 25e-3, #'[1/min]',
            'p2': 20e-3, #'[1/min]',
            'SG': 10e-4, # '[10^(-4)1/min per mg/dL]
            'alpha': 0.1, #'[dL/mg]'
            'beta': 5, #'[min]'
            'T1': 20, #'[min]'
            'T2': 120, #'[min]'
            'W1': 0.6,
            'W2': 0.4,
            'RH': 0.7,
            'V': 1.45, #'[dL/kg]',
            'lambda': 10,
            'f': 0.9,
            # 'bodyweight_kg': 75,
            # 'input_mg': 75_000,
            'D': 75_000/75, #'[mg/kg]',
            'A': 0.9*(75_000/75), #'[mg/kg]',
            'Gb': 90, #'[mg/dL]',
            'G0': 91, #'[mg/dL]',
            'X0': 0, #'[1/min]',
        }

        #### TODO ####
        # Calculated params
        # self.calced_params = ['D', 'A']
        # self.params['D'] = self.params['input_mg']/self.params['bodyweight_kg'] 
        # self.params['A'] = self.params['D']*self.params['f']
        
        
        # step sizes for streamlit
        self.params_steps = {
            'p1': 0.01, #'[1/min]',
            'p2': 0.01, #'[1/min]',
            'SG': 2, # '[10^(-4)1/min per mg/dL]
            'alpha': 0.1, #'[dL/mg]'
            'beta': 1, #'[min]'
            'T1': 10, #'[min]'
            'T2': 20, #'[min]'
            'W1': 0.2,
            'W2': 0.2,
            'RH': 0.2,
            'V': 1, #'[dL/kg]',
            'lambda': 5, #'[mg/dL]',
            'f': 0.1,
            # 'bodyweight_kg': 10,
            # 'input_mg': 10_000,
            'D': 100, #'[mg/kg]',
            'A': 100, #'[mg/kg]',
            'Gb': 90, #'[mg/dL]',
            'G0': 90, #'[mg/dL]',
            'X0': 0, #'[1/min]',
        }
        # inital conditions
        self.y0 = [self.params['G0'], self.params['X0']]

        # ode function
        def ODE_func(t, ys, Ra, params):
            Gb = params['Gb']
            p1 = params['p1']
            alpha = params['alpha']
            beta = params['beta']
            p2 = params['p2']
            SG = params['SG']
            V = params['V']
            G, X = ys

            dG = -G*X - p1*(G-Gb) + Ra(t)/V
            Zpos = 1+np.exp(-alpha*(G-Gb))
            Z = (G-Gb)/Zpos + beta*Ra(t)/V
            dX = -p2*(X-SG*Z)
            return [dG, dX]
        self.ode_func = ODE_func
      
        # input function
        def Ra_Func(t, params):
            # Unpack model params
            A = params['A']
            RH = params['RH']
            T1 = params['T1']
            W1 = params['W1']
            T2 = params['T2']
            W2 = params['W2']

            def f_LN(t, T, W):
                return (np.exp(-((np.log(t/T)-W/2)**2)/W))/(t*np.sqrt(np.pi*W))
            Ra_value = A*(1-RH)*f_LN(t, T1, W1) + A*(RH)*f_LN(t, T2, W2)
            Ra_value = Ra_value
            Ra = interp1d(x=t, y=Ra_value, bounds_error=False)
            return Ra
        self.Ra_func = Ra_Func

    def calc_aux_vars(self):
        if self.dynamic_vars is None:
            self.simlulate()

        Gb = self.params['Gb']
        alpha = self.params['alpha']
        beta = self.params['beta']
        SG = self.params['SG']
        V = self.params['V']
        G = self.dynamics_df['G']
        X = self.dynamics_df['X']
        Ra = self.Ra
        t = self.t_eval

        G_minus_Gb = G-Gb
        Zpos = 1 + np.exp(-alpha*(G-Gb))
        Z = (G-Gb)/Zpos + beta*Ra(t)/V
        Y_GOM = SG*Z

        aux_vars_df = pd.DataFrame({'t':t, 'G_minus_Gb':G_minus_Gb, 'Zpos':Zpos, 'Y_GOM':Y_GOM})
        self.aux_vars_df = aux_vars_df
        self.aux_vars = aux_vars_df.drop('t', axis=1).columns

    def plot_aux_vars(self):
        self.calc_aux_vars()
        # Plot outputs
        n_vars = len(self.aux_vars)+1
        fig, axes = plt.subplots(n_vars, 1, figsize=(6, 0.6+n_vars), sharex=True)
        # Plot Ra input
        ax = axes[0]
        ax.plot(self.t_eval, self.dynamics_df['Ra']/self.params['V'],
                lw=1, color=COLOR_RA, alpha=0.5, label='Ra/V')
        ax.fill_between(
            self.t_eval, 0, self.dynamics_df['Ra']/self.params['V'], color=COLOR_RA, alpha=0.4)
        ax.set(ylabel='Ra(t)/V')
        ax.grid()
        # plot aux vars
        for i, var in enumerate(self.aux_vars):
            ax = axes[i+1]
            ax.plot(self.t_eval, self.aux_vars_df[var], ls='-', lw=2,
            color='k',alpha=0.5, label=var)
            ax.set(ylabel=f'{var}(t)')
            ax.grid()
        ax.set(xlabel='t (minutes)', xticks=np.arange(self.t_start-1,self.t_end, 60))
        fig.tight_layout()
        return fig, axes


if __name__ == "__main__":
    model = GOM_ODE()

    # model.params = {
    #         'p1': 25e-3, #'[1/min]',
    #         'p2': 20e-3, #'[1/min]',
    #         'SG': 10e-4, # '[10^(-4)1/min per mg/dL]
    #         'alpha': 0.1, #'[dL/mg]'
    #         'beta': 5, #'[min]'
    #         'T1': 20, #'[min]'
    #         'T2': 120, #'[min]'
    #         'W1': 0.5,
    #         'W2': 0.5,
    #         'RH': 0.8,
    #         'V': 1.45, #'[dL/kg]',
    #         'lambda': 0,
    #         'f': 0.9,
    #         'D': 75_000/75, #'[mg/kg]',
    #         'A': None, #'[mg/kg]',
    #         'Gb': 90, #'[mg/dL]',
    #         'G0': 91, #'[mg/dL]',
    #         'X0': 10, #'[1/min]',
    #     }
    # model.params['A'] = model.params['D']*model.params['f']

    model.simulate()
    fig, axes = model.plot_aux_vars()
    fig, axes = model.plot_dynamic_vars()
    plt.show()
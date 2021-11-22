import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from config import SEED, FONTSZ, COLOR_G, COLOR_X, COLOR_RA, COLOR_FIT, COLOR_FOOD, COLOR_OGTT


# -- Parent param class
class model_param(object):
    def __init__(self, name=None, latex_str=None, units=None, default_val=None,
                 step_size=None, is_calculated=False, explanation=None, max_val=9999, min_val=-9999, value=None) -> None:
        self.name = name
        self.latex_str = latex_str  # latex str for printing
        self.units = units  # units str e.g. '[1/min]'
        self.default_val = default_val  # default value
        self.step_size = step_size  # step size for streamlit
        self.is_calculated = is_calculated  # is the param calculated or given
        self.explanaton = explanation  # explanatin str
        self.max_val = max_val
        self.min_val = min_val

        self.value = self.default_val
        # TODO - distribution for sampling
        # self.dist = None


# -- Parent input func class
class input_func(object):
    # TODO
    pass


# -- Parent ODE class
class Glucose_ODE(object):
    def __init__(self, t_start=1, t_end=60*7) -> None:
        self.name = None
        self.title = None
        self.equations = None
        self.explainer_text = None

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
        self.measured_df = pd.DataFrame({'t': self.t_eval, 'G_measured': (
            self.dynamics_df['G'] + np.random.normal(loc=0, scale=self.params['lambda'].value, size=len(self.dynamics_df)))})
        self.measured_df = self.measured_df.iloc[::15, :]
        self.dynamics_df = self.dynamics_df.merge(
            self.measured_df, on='t', how='left')

    def plot_dynamic_vars(self):
        # Plot outputs
        n_vars = len(self.dynamic_vars)+1
        fig, axes = plt.subplots(
            n_vars, 1, figsize=(6, 0.8+n_vars), sharex=True)
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
        ax.set(xlabel='t (minutes)', xticks=np.arange(
            self.t_start-1, self.t_end+1, 60))

        # add measurements
        ax = axes[1]
        ax.scatter(self.t_eval, self.dynamics_df['G_measured'], s=20,
                   color=self.dynamic_vars_colors[0], alpha=0.5, label='G measured')

        fig.tight_layout()
        return fig, axes


class GOM_ODE(Glucose_ODE):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'GOM'
        self.title = 'Glucose Only Model (GOM)'
        self.main_text = """
        A biological-inspired model of glucose-insulin dynamics for analysing cgm data using only observed glucose data.
        De-expand details, and change paramters on the left sidebar to see effect on generated data in the plots below. Blue scatter plot are the observed measured values after adding errors.
        """
        self.equations = r"""
        \frac{dG}{dt} = -G(t)X(t)-p_1[G(t)-G_b] + \frac{Ra(t)}{V} \\ 
        \texttt{\char32} \\
        G_{measured}(t) = N(G,\lambda) \\ 
        \texttt{\char32} \\
        \frac{dX}{dt} = -p_2[X(t)-S_GZ(t)] \\
        \texttt{\char32} \\
        Z(t) = \frac{G(t)-G_b}{1+exp[-\alpha(G(t)-G_b)]} + \beta \frac{Ra(t)}{V} \\ 
        \texttt{\char32} \\
        Ra(t) = A(1-R_H)f(t,T_1,W_1) + AR_Hf(t,T_2,W_2) \\ 
        \texttt{\char32} \\
        f(t,T,W) = \frac{1}{\sqrt{\pi W}} exp(\frac{-(log(t/T)-W/2)^2}{W})
        """
        self.explainer_text = """
        $G(t)$ - Glucose state variable [mg/dL]  
        $X(t)$ - Insulin action in a remote compartment [1/min]  
        $Ra(t)$ - Rate of appearance (input funtion) [mg/kg/min]  
        $p_1$ - glucose effectiveness [1/min]  
        $V$ - distribution volume of glucose relative to body weight [dL/kg]  
        $G_b$ - Basal glucose level [mg/dL]  
        $$\lambda$$ - measurement error std [mg/dL]   
        $p_2$ - decay dynamics of X [1/min]  
        $S_G$ - Insulin sensitivity proxy [1/min per mg/dL]  
        $$alpha$$ - shape parameter [dL/mg]  
        $$beta$$ - coupling paramaeter [min]  
        $A$ - total fixed AUC of intake, which is calculated from meal carbohydrate content [mg/kg]  
        $R_H$ - distributions of 2 peaks  
        $T_{1,2}$ - peak times [min]  
        $W_{1,2}$ - widths  

        Model reproduced from:
        [_A Glucose-Only Model to Extract Physiological Information from Postprandial Glucose Profiles in Subjects with Normal Glucose Tolerance_](https://journals.sagepub.com/doi/full/10.1177/19322968211026978)
        (Eichenlab et. al. 2021)
        """

        self.dynamic_vars = ['G', 'X']
        self.dynamic_vars_labels = ['Glucose values', 'X values']
        self.dynamic_vars_colors = [COLOR_G, COLOR_X]

        # model params_
        # (name, latex_str, units, default_val, step_size, is_calculated, explanations, max_val, min_val)
        self.params = {
            'p1': model_param('p1', 'p_1', '[1/min]', 25e-3, 5e-3, False),
            'p2': model_param('p2', 'p_2', '[1/min]', 20e-3, 5e-3, False),
            'SG': model_param('SG', 'S_G', '[mg/dL/min]', 10e-4, 5e-4, False),
            'alpha': model_param('alpha', '\alpha', '[dL/mg]', 0.1, 0.05, False),
            'beta': model_param('beta', '\beta', '[min]', 5, 1, False),
            'T1': model_param('T1', 'T_1', '[min]', 20, 10, False),
            'T2': model_param('T2', 'T_2', '[min]', 120, 10, False),
            'W1': model_param('W1', 'W_1', '', 0.6, 0.2, False),
            'W2': model_param('W2', 'W_2', '', 0.4, 0.2, False),
            'RH': model_param('W1', 'R_H', '', 0.7, 0.1, False),
            'V': model_param('V', 'V', '[dL/kg]', 1.45, 0.1, False),
            'lambda': model_param('lambda', '\lambda', '[mg\dL]', 5, 1, False),
            'f': model_param('f', 'f', '', 0.9, 0.1, False),
            # bodyweight
            # input_mg
            'D': model_param('D', 'D', '[mg/kg]', 75_000/75, 250, False),
            'A': model_param('A', 'A', '[mg/kg]', 0.9*(75_000/75), 0.9*(250), False),
            'Gb': model_param('Gb', 'G_b', '[mg\dL]', 90, 10, False),
            'G0': model_param('G0', 'G_0', '[mg\dL]', 91, 10, False),
            'X0': model_param('X0', 'X_0', '[1\min]', 0, 10, False),
        }

        # inital conditions
        self.y0 = [self.params['G0'].value, self.params['X0'].value]

        # ode function
        def ODE_func(t, ys, Ra, params):
            Gb = params['Gb'].value
            p1 = params['p1'].value
            alpha = params['alpha'].value
            beta = params['beta'].value
            p2 = params['p2'].value
            SG = params['SG'].value
            V = params['V'].value
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
            A = params['A'].value
            RH = params['RH'].value
            T1 = params['T1'].value
            W1 = params['W1'].value
            T2 = params['T2'].value
            W2 = params['W2'].value

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

        Gb = self.params['Gb'].value
        alpha = self.params['alpha'].value
        beta = self.params['beta'].value
        SG = self.params['SG'].value
        V = self.params['V'].value
        G = self.dynamics_df['G']
        Ra = self.Ra
        t = self.t_eval

        G_minus_Gb = G-Gb
        Zpos = 1 + np.exp(-alpha*(G-Gb))
        Z = (G-Gb)/Zpos + beta*Ra(t)/V
        Y_GOM = SG*Z

        aux_vars_df = pd.DataFrame(
            {'t': t, 'G_minus_Gb': G_minus_Gb, 'Zpos': Zpos, 'Y_GOM': Y_GOM})
        self.aux_vars_df = aux_vars_df
        self.aux_vars = aux_vars_df.drop('t', axis=1).columns

    def plot_aux_vars(self):
        self.calc_aux_vars()
        # Plot outputs
        n_vars = len(self.aux_vars)+1
        fig, axes = plt.subplots(
            n_vars, 1, figsize=(6, 0.6+n_vars), sharex=True)
        # Plot Ra input
        ax = axes[0]
        ax.plot(self.t_eval, self.dynamics_df['Ra']/self.params['V'].value,
                lw=1, color=COLOR_RA, alpha=0.5, label='Ra/V')
        ax.fill_between(
            self.t_eval, 0, self.dynamics_df['Ra']/self.params['V'].value, color=COLOR_RA, alpha=0.4)
        ax.set(ylabel='Ra(t)/V')
        ax.grid()
        # plot aux vars
        for i, var in enumerate(self.aux_vars):
            ax = axes[i+1]
            ax.plot(self.t_eval, self.aux_vars_df[var], ls='-', lw=2,
                    color='k', alpha=0.5, label=var)
            ax.set(ylabel=f'{var}(t)')
            ax.grid()
        ax.set(xlabel='t (minutes)', xticks=np.arange(
            self.t_start-1, self.t_end, 60))
        fig.tight_layout()
        return fig, axes


if __name__ == "__main__":
    model = GOM_ODE()
    model.simulate()
    fig, axes = model.plot_aux_vars()
    fig, axes = model.plot_dynamic_vars()
    plt.show()

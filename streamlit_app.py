import streamlit as st
from ode_models import GOM_ODE

st.set_page_config(layout="wide")

# Init model
model = GOM_ODE()

with st.sidebar:
    st.header(model.title)
    st.text('Change parameters and\nsee effect on generated data')
    # get param inputs from sidebar
    for param_name, param in model.params.items():
        new_param_val = st.number_input(
            label=f'{param.name} {param.units}', min_value=float(param.min_val), max_value=float(param.max_val),
            value=float(param.default_val), step=float(param.step_size), key=param_name, format="%f")
        # update
        model.params[param_name].value = new_param_val

# Simulate
model.simulate()

with st.expander('Model details', expanded=True):
    # Title & Equation
    st.header(model.title)
    st.markdown(model.main_text)
    st.latex(model.equations)
    # explainer
    st.markdown(model.explainer_text)

with st.expander('Generated dynamics', expanded=True):
    # Plot main vars
    fig, axes = model.plot_dynamic_vars()
    st.pyplot(fig)

with st.expander('Auxilary variables', expanded=False):
    # Plot aux vars
    fig, axes = model.plot_aux_vars()
    st.pyplot(fig)

    # Show data tables
    st.write(model.dynamics_df)
    st.write(model.aux_vars_df)

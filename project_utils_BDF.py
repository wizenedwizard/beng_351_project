import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import copy
import random

def run_model(stimulus_val, p_base, tspan, t_eval, dNetwork, param_to_vary, param_value, segments):
    """
    Solves ODE system for a set of given parameters 
    -------
    stimulus_val: scalar concentration of your system's stimulus 
    p_base: dictionary to access parameters
    tspan: tuple with (start, end) times for simulation run
    t_eval: list or array of time points to get a solution from the ODE system
    dNetwork: function with ODEs to model
    param_to_vary: dictionary key for parameter being varied
    param_value: scalar value of param_to_vary for this simulation run
    segments: list of dictionaries containing each dose and the time span it is applied over the simulation
              Should look like so, where n is the total number of doses applied:
              segments = [{stimkey: stimulus_1, "tspan": tspan_1}, {stimkey: stimulus_2, "tspan": tspan_2} ... {stimkey: stimulus_n, "tspan": tspan_n}]            
    -------         
    returns
    -------
    time_all: array for time of simulation
    species_all: species concentrations over the simulation
    """
    stimkey = list(p_base.keys())[0]
    # Create a fresh copy of parameters for this run
    if not segments:
        if not isinstance(stimulus_val*1.0, float):
            stimulus_val = stimulus_val[0]
        segments = [{stimkey: stimulus_val, "tspan": tspan}]
    time_all = []
    states_all = []
    p = copy.deepcopy(p_base)
    p[param_to_vary] = param_value
    current_y0 = [p[key] for key in list(p_base.keys()) if key[0].isupper()]
    for seg in segments:
        for key in list(seg.keys()):
            p[key] = seg[key]
        # Generate time points for this segment
        tspan = seg["tspan"]
        t_eval = np.linspace(tspan[0], tspan[1], 200)
        # Run the simulation
        sol = solve_ivp(
            fun=lambda t, y: dNetwork(t, y, p),
            t_span=tspan,
            y0=current_y0,
            t_eval=t_eval,
            method='BDF'
        )
        # Store values
        time_all.append(sol.t)
        states_all.append(sol.y)
        # Use the final state of this segment as the initial condition for the next segment.
        current_y0 = sol.y[:, -1]
        
    # Concatenate time and state arrays from all segments for continuous plotting.
    time_all = np.concatenate(time_all)
    states_all = np.concatenate(states_all, axis=1)
    
    return states_all, time_all

def graph_model(p_base, tspan, t_eval, dNetwork, segments=[]):
    """
    Checks parameters and returns graphs
    -------
    p_base: dictionary to access parameters
    tspan: tuple with (start, end) times for simulation run
    t_eval: list or array of time points to get a solution from the ODE syste
    dNetwork: function with ODEs to modelm
    segments: list of dictionaries containing each dose and the time span it is applied over the simulation
              Should look like so, where n is the total number of doses applied:
              segments = [{"s": s_1, "tspan": tspan_1}, {"s": s_2, "tspan": tspan_2} ... {"s": s_n, "tspan": tspan_n}]         
              If left empty, only one dose is applied for the whole tspan
    -------
    Plots
    -------
    If you enter a scalar stimulus value, the code will plot the time-course of all species for that m, changing the dose as specified in segments
    If you enter an array of stimuli, the code will plot the steady-state values of all species vs. stimulus with multiple curves if another dictionary key is varied
    """
    stimulus = list(p_base.keys())[0]
    
    # Raise errors if provided inputs will cause problems with graphing procedures
    if len(np.where([isinstance(val[1], np.ndarray) for val in list((p_base.items()))[1:]])[0]) > 1:
        raise ValueError(f"For graphing purposes, you can only set a range of values for one non-stimulus parameter at a time. \nParameters with more than one value:{[val for val in list((p_base.items()))[1:] if isinstance(val[1], np.ndarray)]}")
    # Raise an error if both m_values and segments have more than one value.
    if (not isinstance((p_base[stimulus]*1.0), float)) and segments:
        raise ValueError("Cannot perform a steady state response analysis while varying the dose over time")
    
    # Check to see if checking a range of doses or just one dose
    if isinstance((p_base[stimulus]*1.0), float):
        p_base[stimulus] = [p_base[stimulus]]

    var_names = [key for key in list(p_base.keys()) if key[0].isupper()]
    
    # Loop over m and parameter value(s) and generate a time-course for each simulation
    #check to see if any parameters are given ranges
    if len([val for val in list((p_base.items()))[1:] if isinstance(val[1], np.ndarray)]):
        param_to_vary, param_values = [val for val in list((p_base.items()))[1:] if isinstance(val[1], np.ndarray)][0]
    else:
        param_to_vary = None
        param_values = [1234567]
        
    psize = max(1, len(param_values))
    msize = max(1, len(p_base[stimulus]))
    FINAL_STATES = np.zeros((psize, msize,len(var_names)))
    for p in range(psize):
        for l in range(msize):
            solution, times = run_model(p_base[stimulus][l], p_base, tspan, t_eval, dNetwork, param_to_vary, param_values[p], segments)
            final_states = solution[:, -1]
            #print(f"m = {m:.2f}, Final state: {final_state}")
            FINAL_STATES[p,l,:] = final_states
    FINAL_STATES = np.array(FINAL_STATES)
        
    if msize == 1:
        # plot species concentrations over time if only graphing one ligand dose
        # Create a new figure for the time-dependent behavior for this m
        plt.figure(figsize=(10, 6))
        for i in range(len(var_names)):
            plt.plot(times, solution[i, :], label=var_names[i])
        plt.xlabel("Time [min]")
        plt.ylabel("Concentration [molecules]")
        plt.title(f"Time-dependent behavior for {stimulus} = {p_base[stimulus][0]:.2f} mM")
        if param_to_vary != None:
            plt.legend(title=f"{param_to_vary} = {param_values[0]}")
        else:
            plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if msize > 1:   
        # plot species concentrations vs m if graphing more than one ligand dose
        if len(var_names) % 3 == 0:
            fig, axes = plt.subplots(len(var_names)//3, 3, figsize=(12, 3*len(var_names)//3))
        elif len(var_names) % 2 == 0:
            fig, axes = plt.subplots(len(var_names)//2, 2, figsize=(15, 10))
        else: 
            nextup = len(var_names) + (len(var_names)%3)
            fig, axes = plt.subplots(nextup//3, 3, figsize=(15, 10))
        axes = axes.flatten()
        colors = [f"#{random.randrange(0x1000000):06x}" for i in range(max(1,len(param_values)))]

        # Plot a new line for each value of our varied parameter in the provided range
        for p in range(len(param_values)):
            for i in range(len(var_names)):          
                ax = axes[i]
                ax.semilogx(p_base[stimulus], FINAL_STATES[p, :, i], 'o-', markersize=5, color=colors[p], label=p)
                ax.set_xlabel("m [uM]")
                ax.set_ylabel(f"Steady-state {var_names[i]} [uM]")
                ax.set_title(f"{var_names[i]} vs. m")
                ax.grid(True)

        if param_to_vary != None: 
            axes[0].legend(title=param_to_vary)
        plt.tight_layout()
        plt.show()

import numpy as np
from scipy.stats import lognorm

class Fleet:
    
    def __init__(self, numofunits=1000, numcycles=50, requirement=10, 
                 life=25,stepsize=1,
                 theta2=0,theta3=0,stimulusrange=(0,100),stimulusnum=100,
                 state2start_init=(3.7,0.2), state2end_init=(2,0.1),
                 state3start_init=(4.5,0.3),bias2=0.4, bias3=0.4):
        
        self.numofunits = numofunits
        self.numcycles = numcycles
        self.requirement = requirement
        self.stepsize=stepsize
        self.theta2=theta2
        self.theta3=theta3
        self.stimulusrange=stimulusrange
        self.stimulusnum=stimulusnum
        self.state2start_init=state2start_init
        self.state2end_init=state2end_init
        self.state3start_init=state3start_init
        self.bias2=bias2
        self.bias3=bias3
        
        self.unit={}
        self.life=life
        
        #Instantiate with fleetfailure=False
        self.fleetfailure=False
        self.failedunits=[]
    
    def create_fleet(self):
        
        for unitnum in range(self.numofunits):
            
            #Generate stimulus table
            stimulus = np.random.uniform(low=self.stimulusrange[0], 
                                         high=self.stimulusrange[1], 
                                         size=(self.stimulusnum, self.numcycles))
            
            #Get the trajectories for state 2 for start params
            startmu2_trajectory,startsigma2_trajectory = self.directional_random_walk(
                self.numcycles -1 ,self.stepsize,self.theta2,self.bias2,
                self.state2start_init[0],self.state2start_init[1])
            
            #Get the trajectories for state 3 for start params
            startmu3_trajectory,startsigma3_trajectory = self.directional_random_walk(
                self.numcycles -1 ,self.stepsize,self.theta3,self.bias3,
                self.state3start_init[0],self.state3start_init[1])
            
            #Get the trajectories for the state 2 stop params
            stopmu2_trajectory, stopsigma2_trajectory = self.directional_random_walk(
                self.numcycles - 1,self.stepsize,0,0,
                self.state2end_init[0],self.state2end_init[1])
            
            #Generate the unit's data
            
            unitdatatbl,failed,failed_cycle = self.multinom_state_sim(stimulus, 
                                                  startmu2_trajectory, startsigma2_trajectory, 
                                                  stopmu2_trajectory, stopsigma2_trajectory,
                                                  startmu3_trajectory, startsigma3_trajectory)
            #Generate ages
            ages = np.arange(self.numcycles) +\
                np.random.uniform(-1, 1, size=self.numcycles)
            
            ages[ages<0]=0
                
            if not np.isnan(failed_cycle):
                failed_age = ages[failed_cycle]
                
                if failed_age <= self.life:
                    self.fleetfailure=True
            else:
                failed_age = np.nan
            
            self.unit[unitnum] = {'id':unitnum, 'age':ages,
                                  'data':unitdatatbl,
                                  'failed':failed,'failedage':failed_age}
            
            
    def directional_random_walk(self, n_steps, step_size, theta, 
                                bias_strength=0.4,x_init=0,y_init=1):
        """
        Generate a 2D random walk with a directional bias toward angle theta,
        and variable step size between 0 and step_size.

        Parameters:
            n_steps (int): Number of steps in the walk.
            step_size (float): Maximum length of each step.
            theta (float): Preferred direction in radians (0 = +x, pi/2 = +y).
            bias_strength (float): Between 0 and 1. Higher = stronger pull toward theta.

        Returns:
            x (np.ndarray): Array of x coordinates (length n_steps+1).
            y (np.ndarray): Array of y coordinates (length n_steps+1).
        """
        if y_init <= 0:
            print("Warning: negative values of y are not allowed. Reseting to 1.")
            y_init = 1
        
        # Initialize arrays
        x = np.zeros(n_steps + 1) + x_init
        y = np.zeros(n_steps + 1) + y_init

        for i in range(1, n_steps + 1):
            current_step = np.random.uniform(0, step_size)

            dx_bias = np.cos(theta)
            dy_bias = np.sin(theta)

            rand_angle = np.random.uniform(-np.pi, np.pi)
            dx_rand = np.cos(rand_angle)
            dy_rand = np.sin(rand_angle)

            dx = bias_strength * dx_bias + (1 - bias_strength) * dx_rand
            dy = bias_strength * dy_bias + (1 - bias_strength) * dy_rand

            norm = np.sqrt(dx**2 + dy**2)
            dx = (dx / norm) * current_step
            dy = (dy / norm) * current_step

            x[i] = x[i - 1] + dx
            #We need to make sure y doesn't go below 0, because that is nonsense
            y[i] = abs(y[i - 1] + dy)
            if y[i] == 0 :
                y[i]= 1e-6

        return x, y



    def multinom_state_sim(self,stimulus,
                           mustartstate2, sigmastartstate2,
                           muendstate2, sigmaendstate2,
                           mustartstate3, sigmastartstate3):
        """
        Classify stimulus matrix by sampling thresholds from lognormal distributions
        defined per Monte Carlo replicate.

        Parameters:
            stimulus (np.ndarray): (N, steps) array of stimuli per trial.
            mustartstate2, sigmastartstate2,
            muendstate2, sigmaendstate2,
            mustartstate3, sigmastartstate3: Arrays of length steps with per-mont parameters.

        Returns:
            tbl (list of dict): Each dict has 'stimulus' and 'result' for one Monte Carlo trial.
            """
        stimulus = np.asarray(stimulus)

        if stimulus.ndim != 2:
            raise ValueError("Stimulus must be a 2D array (num_trials x steps).")

        num_trials, steps = stimulus.shape

        # Validate parameter lengths
        def check_param_shape(arr, name):
            arr = np.asarray(arr)
            if arr.shape != (steps,):
                raise ValueError(f"{name} must be of shape ({steps},), got {arr.shape}")
            return arr

        mustartstate2 = check_param_shape(mustartstate2, "mustartstate2")
        sigmastartstate2 = check_param_shape(sigmastartstate2, "sigmastartstate2")
        muendstate2 = check_param_shape(muendstate2, "muendstate2")
        sigmaendstate2 = check_param_shape(sigmaendstate2, "sigmaendstate2")
        mustartstate3 = check_param_shape(mustartstate3, "mustartstate3")
        sigmastartstate3 = check_param_shape(sigmastartstate3, "sigmastartstate3")

        # Preallocate results
        tbl = []

        #Set unit as not failed
        failed = False
        failed_cycle = np.nan
        
        for step in range(steps):
            stim_col = stimulus[:, step]

            # Create distributions for this step
            state2start = lognorm(s=sigmastartstate2[step], 
                                  scale=np.exp(mustartstate2[step]))
            
            state2end = lognorm(s=sigmaendstate2[step], 
                                scale=np.exp(muendstate2[step]))
            
            state3start = lognorm(s=sigmastartstate3[step], 
                                  scale=np.exp(mustartstate3[step]))

            # Sample thresholds for this step
            s2start_thresh = state2start.rvs(size=num_trials)
            s2end_thresh = s2start_thresh + state2end.rvs(size=num_trials)
            s3start_thresh = state3start.rvs(size=num_trials)

            # Logical classification
            State11 = (stim_col < s2start_thresh) & (stim_col < s3start_thresh)
            State2  = (stim_col > s2start_thresh) & (stim_col < s2end_thresh) & (stim_col < s3start_thresh)
            State12 = (stim_col > s2end_thresh) & (stim_col < s3start_thresh)
            State3  = (stim_col > s3start_thresh)

            # Final states encoded as integers
            State1 = State11 | State12
            state_code = (State1.astype(int) * 1) + \
                         (State2.astype(int) * 2) + \
                             (State3.astype(int) * 3)

            results_tbl=np.column_stack((stim_col,state_code))
            #Sort the results by stimulus
            sorted_ind=results_tbl[:,0].argsort()
            
            results_tbl=results_tbl[sorted_ind]
            
            if failed == False:
                failed,failed_cycle = self.checkfailure(results_tbl,step)
            
            # Store in output list
            tbl.append((results_tbl))

        return tbl, failed,failed_cycle
    
    def checkfailure(self,results_tbl,step):
        
        failed = False
        failed_cycle = np.nan
        
        chktbl=results_tbl[results_tbl[:,0]<=self.requirement,:]

        if any(chktbl[:,1]==2) or any(chktbl[:,1]==3):
            failed=True
            failed_cycle=step

        return failed,failed_cycle


    def to_sql(self, database_path, table_name='trainingdata'):
        """Write the fleet's unit data to an SQLite database.

        Parameters
        ----------
        database_path : str
            Path to the SQLite database file to create/overwrite.
        table_name : str, optional
            Name of the table to store the data in, by default ``'trainingdata'``.

        Notes
        -----
        The resulting table contains four columns matching the
        expectations of :class:`ThreeStateSolverNetwork`:

        ``Result``  (int)
            The state code for a particular insult level.
        ``Age`` (float)
            Age corresponding to the timestep of the measurement.
        ``UnitID`` (int)
            Identifier of the unit within the fleet.
        ``Insult`` (float)
            The stimulus value associated with the result.

        The method flattens the internal ``unit`` dictionary into a
        long-format table and writes it to ``database_path`` using
        :func:`pandas.DataFrame.to_sql`.
        """

        # Local imports to keep base requirements light if SQL export is
        # never used.
        import pandas as pd
        from sqlalchemy import create_engine

        records = []

        for unit_id, unit_data in self.unit.items():
            ages = unit_data['age']
            timesteps = unit_data['data']

            for age, timestep_tbl in zip(ages, timesteps):
                timestep_tbl = np.asarray(timestep_tbl)
                if timestep_tbl.size == 0:
                    continue

                insults = timestep_tbl[:, 0]
                results = timestep_tbl[:, 1]

                for insult, result in zip(insults, results):
                    records.append({
                        'Result': int(result),
                        'Age': float(age),
                        'UnitID': int(unit_id),
                        'Insult': float(insult)
                    })

        if not records:
            # No data to write; create an empty database with the proper schema.
            df = pd.DataFrame(columns=['Result', 'Age', 'UnitID', 'Insult'])
        else:
            df = pd.DataFrame.from_records(records)

        engine = create_engine(f'sqlite:///{database_path}')
        with engine.begin() as connection:
            df.to_sql(table_name, connection, if_exists='replace', index=False)

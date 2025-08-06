# ---------- Imports ----------
import numpy as np
import os
from sqlalchemy import create_engine
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------- Class ----------
class ThreeStateSolverNetwork:
    """
    A class that will train on paired data with three states 
    (1, 2, 3) distributed across a range of input values.

    Inputs:
        - insult
        - age
        - unitid

    Output:
        - Fleet-level service life estimate, based on the condition:
          No events of type (2, 3) can be observed below a given
          'req' threshold for a fraction of 'numunits'.
    """

    def __init__(self, database_path, table_name='trainingdata', numunit=100, 
                 req=2.0, float64=False):
        
        assert np.isscalar(numunit), 'The number of units must be a scalar.'
        assert np.isscalar(req), 'The requirement must be a scalar.'
        assert os.path.exists(database_path), 'The database path does not exist.'
        assert isinstance(float64, bool), 'float64 must be a boolean (True or False).'

        # Store parameters
        self.numunits = numunit
        self.req = req
        self.table_name = table_name
        self.float64 = float64

        # Set dtypes dynamically
        self.np_dtype = np.float64 if float64 else np.float32
        self.torch_dtype = torch.float64 if float64 else torch.float32

        # Connect to the SQL database
        self.engine = create_engine(f'sqlite:///{database_path}')
        query = f"SELECT Result, Age, UnitID, Insult FROM {table_name};"
        self.trainingdataframe = pd.read_sql_query(query, self.engine)

    # ---------- Histogram Encoder Function ----------
    def encode_time_step(self, insults, 
                         results, age, bins=10, bin_range=(0, 10)):
        """
        Convert insults and result types into a fixed-length vector 
        using histogram binning.

        Output shape: [1 + 3 * bins]  (includes age)
        """
        insults = np.array(insults)
        results = np.array(results)

        hist_r1, _ = np.histogram(insults[results == 1], bins=bins,
                                  range=bin_range)
        hist_r2, _ = np.histogram(insults[results == 2], bins=bins,
                                  range=bin_range)
        hist_r3, _ = np.histogram(insults[results == 3], bins=bins,
                                  range=bin_range)

        features = np.concatenate([[age], hist_r1, hist_r2, hist_r3]).astype(self.np_dtype)
        return features

        #Build the PyTorch tensor
    def build_tensor_sequences(self, bins=10, bin_range=(0, 10)):
        """
        Convert the raw dataframe into a dictionary:
        { UnitID: torch.tensor([ [features_t1], [features_t2], ... ]) }

        Each feature_t includes histogram encoding + age.
        """
        grouped = self.trainingdataframe.groupby(['UnitID', 'Age'])

        # Temporary storage: {UnitID: [encoded_features_at_each_age]}
        sequences = {}

        for (unit_id, age), group in grouped:
            insults = group['Insult'].to_numpy()
            results = group['Result'].to_numpy()
            encoded = self.encode_time_step(insults, results, age,
                                            bins=bins, bin_range=bin_range)
            
            """
            Create a dictionary for the encodings mapped to the unitid.
            If there isn't already an entry, make one.
            """
            if unit_id not in sequences:
                sequences[unit_id] = []
                
            sequences[unit_id].append(encoded)

        # Sort each unit's sequence by age and convert to tensor
        tensor_sequences = {}
        for unit_id, seq in sequences.items():
            seq_sorted = sorted(seq, key=lambda x: x[0])  # sort by age
            tensor = torch.tensor(seq_sorted, dtype=self.torch_dtype)
            tensor_sequences[unit_id] = tensor

        return tensor_sequences

# ---------- Training Method ----------
    def trainmodel(self, bins=10, bin_range=(0, 10), hidden_size=64,
              epochs=10, lr=0.001, verbose=True):
        """
        Instantiate and train an LSTM model using tensor_sequences.
        """
        tensor_sequences = self.build_tensor_sequences(bins=bins, bin_range=bin_range)
        input_size = 1 + 3 * bins

        self.model = FailurePredictor(input_size=input_size, hidden_size=hidden_size)
        self.model = self.model.to(dtype=self.torch_dtype)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        self.model.train()
        sequences = list(tensor_sequences.values())
        targets = [torch.zeros(seq.size(0), 1, dtype=self.torch_dtype) for seq in sequences]  # dummy targets

        for epoch in range(epochs):
            total_loss = 0.0
            for seq, target in zip(sequences, targets):
                seq = seq.unsqueeze(0)       # [1, T, F]
                target = target.unsqueeze(0) # [1, T, 1]

                optimizer.zero_grad()
                output, _ = self.model[0](seq)
                prediction = self.model[1](output)
                loss = loss_fn(prediction, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(sequences):.4f}")


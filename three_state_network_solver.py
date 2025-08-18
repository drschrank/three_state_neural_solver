# ---------- Imports ----------
import numpy as np
import os
from sqlalchemy import create_engine
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import re
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
        
        #For check that the database path is valid
        db_path = Path(database_path).expanduser().resolve()
        assert db_path.exists() and db_path.is_file(), f"DB not found: {db_path}"

        #For checking that the table name is valid


        valid_ident = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
        assert isinstance(table_name, str), "table_name must be a string."
        assert valid_ident.match(table_name), (
            f"Invalid table_name '{table_name}'. "
            "Must start with a letter/underscore and contain only letters, digits, or underscores."
                )


        # Store parameters
        self.numunits = numunit
        self.req = req
        self.table_name = table_name
        self.float64 = float64

        # Set dtypes dynamically
        self.np_dtype = np.float64 if float64 else np.float32
        self.torch_dtype = torch.float64 if float64 else torch.float32

        # Connect to the SQL database
        self.engine = create_engine(f"sqlite:///{db_path.as_posix()}")
        query = f"SELECT Result, Age, UnitID, Insult FROM {table_name};"
        self.trainingdataframe = pd.read_sql_query(query, self.engine)

    # ---------- Histogram Encoder Function ----------
    def encode_time_step(self, timestepdata, 
                         age, bins=10, bin_range=(0, 10)):
        """
        Convert insults and result types into a fixed-length vector 
        using histogram binning.

        Output shape: [1 + 3 * bins]  (includes age)
        """
        timestepdata=timestepdata.sort_values(['Insult'])

        insults = timestepdata['Insult'].to_numpy()
        results = timestepdata['Result'].to_numpy()

        hist_r1, _ = np.histogram(insults[results == 1], bins=bins,
                                  range=bin_range)
        hist_r2, _ = np.histogram(insults[results == 2], bins=bins,
                                  range=bin_range)
        hist_r3, _ = np.histogram(insults[results == 3], bins=bins,
                                  range=bin_range)

        features = (np.concatenate([[age], hist_r1, hist_r2, hist_r3])
                    .astype(self.np_dtype))
        
        # Check for failure: Result in (2,3) at or below Insult < req
        failure_results=np.concatenate((insults[results == 2],
                                        insults[results == 3]))
        
        failure_condition = any(failure_results <= self.req)
        
        target = 1.0 if failure_condition == True else 0.0

        return features, target

        #Build the PyTorch tensor
    def build_tensor_sequences(self, bins=10, bin_range=(0, 10)):
        """
        Returns:
          tensor_sequences: { UnitID: torch.Tensor [T, F] }
          tensor_targets:   { UnitID: torch.Tensor [T, 1] }
        """
        unitdata = self.trainingdataframe.sort_values(['UnitID', 'Age'])
        grouped_by_unit = unitdata.groupby('UnitID')
    
        tensor_sequences = {}
        tensor_targets = {}
    
        for unit_id, group in grouped_by_unit:
            # Ensure strictly increasing ages within unit
            group = group.sort_values('Age')
    
            features_list = []
            target_list = []
    
            for age, timestepdata in group.groupby('Age'):
                features, target = self.encode_time_step(
                    timestepdata, age, bins=bins, bin_range=bin_range
                )
                # Expect: features -> np.ndarray [F], target -> scalar/np.ndarray []
                features_list.append(np.asarray(features))
                target_list.append(np.asarray(target))
    
            # Optional: sanity-check consistent feature shape across timesteps
            feat_shapes = {f.shape for f in features_list}
            if len(feat_shapes) != 1:
                raise ValueError(
                    f"Inconsistent feature shapes for UnitID={unit_id}: {feat_shapes}. "
                    "Pad/truncate to a common length before stacking."
                )
    
            # Stack to a single ndarray, then convert (no copy) to torch
            feats_np = np.stack(features_list, axis=0).astype(self.np_dtype, copy=False)  # [T, F]
            targs_np = np.asarray(target_list, dtype=self.np_dtype).reshape(-1, 1)        # [T, 1]
    
            tensor_sequences[unit_id] = torch.from_numpy(feats_np).to(dtype=self.torch_dtype)
            tensor_targets[unit_id]   = torch.from_numpy(targs_np).to(dtype=self.torch_dtype)
    
        return tensor_sequences, tensor_targets


# ---------- Training Method ----------

    def trainmodel(self, bins=10, bin_range=(0, 10), hidden_size=64,
               epochs=10, lr=0.001, batch_size=5, verbose=True):
        """
        Train the LSTM model using PyTorch Dataset and DataLoader.
        """
        # Step 1: Build training sequences
        tensor_sequences, tensor_targets = self.build_tensor_sequences(
            bins=bins, bin_range=bin_range)
        input_size = 1 + 3 * bins

        # Step 2: Set up model
        self.model = FailurePredictor(input_size=input_size,
                                  hidden_size=hidden_size)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device=device, dtype=self.torch_dtype)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.BCELoss(reduction='none')
        self.model.train()

        # Step 3: DataLoader setup
        dataset = FailureDataset(tensor_sequences, tensor_targets)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, collate_fn=self.collate_fn)

        # Step 4: Training loop
        for epoch in range(epochs):
            total_loss = 0.0

            for batch_seqs, batch_tgts, mask in dataloader:
                # Move to the correct device and dtype
                batch_seqs = batch_seqs.to(device=device, dtype=self.torch_dtype)
                batch_tgts = batch_tgts.to(device=device, dtype=self.torch_dtype)
                mask = mask.to(device=device)
                
                optimizer.zero_grad()

                outputs = self.model(batch_seqs)  # [B, T, 1]

                elementwise_loss = loss_fn(outputs, batch_tgts).squeeze(-1)  # [B, T]
                masked_loss = elementwise_loss[mask]
                loss = masked_loss.mean()

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(dataloader):.4f}")
                
    def predict(self, test_sequences, batch_size=5):
        """
        Predict failure probabilities for a dictionary of input sequences.

        Args:
            test_sequences (dict): { UnitID: tensor([[feature_t1], [feature_t2], ...]) }
            batch_size (int): Number of sequences to process in a batch.

        Returns:
            predictions_dict (dict): { UnitID: tensor([[prob_t1], [prob_t2], ...]) }
            """
        self.model.eval()
        device = next(self.model.parameters()).device  # Automatically detect device

        dataset = FailureDataset(test_sequences, test_sequences)  # Dummy targets
        dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=self.collate_fn)

        unit_ids = list(test_sequences.keys())
        predictions_dict = {}
        start_idx = 0

        with torch.no_grad():
            for batch_seqs, _, mask in dataloader:
                batch_seqs = batch_seqs.to(device=device, dtype=self.torch_dtype)

                outputs = self.model(batch_seqs).squeeze(-1)  # [B, T]
                masked_outputs = [out[:m.sum()] for out, m in zip(outputs, mask)]

                for i, probs in enumerate(masked_outputs):
                    predictions_dict[unit_ids[start_idx + i]] = probs.cpu()

            start_idx += len(masked_outputs)

        return predictions_dict

           
    @staticmethod       
    def collate_fn(batch):
        """
        Pads variable-length sequences and returns a mask.
        Returns:
            - padded_sequences: [B, T_max, F]
            - padded_targets:   [B, T_max, 1]
            - mask:             [B, T_max] where valid=1, padded=0
            """
        sequences, targets = zip(*batch)  # each: list of tensors
        lengths = [seq.size(0) for seq in sequences]

        padded_seqs = pad_sequence(sequences, batch_first=True)
        padded_tgts = pad_sequence(targets, batch_first=True)

        mask = torch.zeros(len(sequences), padded_seqs.size(1), dtype=torch.bool)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1

        return padded_seqs, padded_tgts, mask


class FailurePredictor(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size=1):
        super(FailurePredictor, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)          # [batch, seq_len, hidden_size]
        out = self.fc(lstm_out)             # [batch, seq_len, 1]
        out = self.sigmoid(out)             # [batch, seq_len, 1] with values in (0,1)
        return out

class FailureDataset(Dataset):
    def __init__(self, tensor_sequences, tensor_targets):
        self.sequences = list(tensor_sequences.values())
        self.targets = list(tensor_targets.values())

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

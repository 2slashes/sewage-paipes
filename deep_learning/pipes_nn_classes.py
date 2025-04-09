import torch
from torch import nn
from torch.utils.data import Dataset
import os
import pandas as pd


class PipesDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        curr_dir = os.path.dirname(__file__)
        data_path = os.path.join(curr_dir, path)
        self.df = pd.read_csv(data_path, dtype=str)  # dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        all = self.df.iloc[idx]

        state = all.iloc[0]
        action = all.iloc[1]

        # Create a list, where each entry in the list is an int
        # the list as a whole represents the state of the board
        state_int_list = [int(x) for x in state]
        state_tensor = torch.tensor(state_int_list)

        action_int_list = [int(x) for x in action]
        action_tensor = torch.tensor(action_int_list)

        return (state_tensor, action_tensor)


class PipesPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)


class PipesLoss(nn.Module):
    def __init__(self, penalty=1.0):
        super().__init__()
        self.penalty = penalty

    def forward(self, pred, actions):
        """
        predictions: Tensor of shape (batch_size, num_classes) - raw logits
        labels: Tensor of shape (batch_size, num_classes) - multi-hot encoded labels (0 or 1)
        """
        # convert the output to probabilities
        probabilities = nn.functional.softmax(pred, dim=1)

        # normalize labels
        normalized_labels = actions / actions.sum(dim=1, keepdim=True).clamp(min=1e-7)

        # compute the sum for the labels
        sums = torch.sum(actions, dim=1)

        # cross entropy loss for each pipe
        loss = -torch.sum(normalized_labels * torch.log(probabilities + 1e-7), dim=1) / sums
        # take the average of the loss values
        return loss.mean()

    def forward_bad(self, pred, actions):
        """
        predictions: Tensor of shape (batch_size, num_classes) - raw logits
        labels: Tensor of shape (batch_size, num_classes) - multi-hot encoded labels (0 or 1)
        """
        # convert the output to probabilities
        probabilities = nn.functional.softmax(pred, dim=1)

        # get the most likely pipe as predicted by the neural network
        predicted_pipe = torch.argmax(pred, dim=1)

        # check if the pipe is in the labels
        # if so, set label to 1 where the predicted pipe is and set everything else to 0
        # if not, normalize the label and keep it as is
        valid_mask = actions.gather(1, predicted_pipe.unsqueeze(1)).squeeze(1) != 0  # Shape: (batch_size

        # Only set the predicted pipe index to 1 where the label contains the predicted pipe
        edit = torch.zeros_like(actions, dtype=torch.float)

        edit.scatter_(1, predicted_pipe.unsqueeze(1), valid_mask.float().unsqueeze(1))

        # Compute cross-entropy loss
        loss_if_valid = -torch.sum(edit * torch.log(probabilities + 1e-7), dim=1)

        # Normalize the labels when the predicted pipe is not in the original labels
        normalized_labels = actions / actions.sum(dim=1, keepdim=True).clamp(min=1e-7)
        loss_if_invalid = -torch.sum(normalized_labels * torch.log(probabilities + 1e-7), dim=1)

        # Use `torch.where` to apply different losses based on `valid_mask`
        loss = torch.where(valid_mask, loss_if_valid, loss_if_invalid)

        # Return the mean loss across the batch
        return loss.mean()

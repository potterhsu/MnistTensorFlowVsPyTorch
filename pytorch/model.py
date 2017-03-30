import os
import glob
import torch
import torch.nn as nn


class Model(nn.Module):
    CHECKPOINT_FILENAME_PATTERN = 'model-{}.tar'

    def __init__(self):
        super(Model, self).__init__()
        self._features = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self._classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), 64 * 4 * 4)
        x = self._classifier(x)
        return x

    def save(self, path_to_dir, step, maximum=5):
        path_to_models = glob.glob(os.path.join(path_to_dir, Model.CHECKPOINT_FILENAME_PATTERN.format('*')))
        if len(path_to_models) == maximum:
            min_step = min([int(path_to_model.split('/')[-1][6:-4]) for path_to_model in path_to_models])
            path_to_min_step_model = os.path.join(path_to_dir, Model.CHECKPOINT_FILENAME_PATTERN.format(min_step))
            os.remove(path_to_min_step_model)

        path_to_checkpoint_file = os.path.join(path_to_dir, Model.CHECKPOINT_FILENAME_PATTERN.format(step))
        torch.save(self.state_dict(), path_to_checkpoint_file)
        return path_to_checkpoint_file

    def load(self, path_to_checkpoint_file):
        self.load_state_dict(torch.load(path_to_checkpoint_file))
        step = int(path_to_checkpoint_file.split('/')[-1][6:-4])
        return step

    def load_max_step(self, path_to_checkpoint_dir):
        path_to_models = glob.glob(os.path.join(path_to_checkpoint_dir, Model.CHECKPOINT_FILENAME_PATTERN.format('*')))
        max_step = max([int(path_to_model.split('/')[-1][6:-4]) for path_to_model in path_to_models])
        path_to_checkpoint_file = os.path.join(path_to_checkpoint_dir, Model.CHECKPOINT_FILENAME_PATTERN.format(max_step))
        self.load(path_to_checkpoint_file)
        return max_step

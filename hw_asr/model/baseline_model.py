from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class BaselineModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.net = Sequential(
            # people say it can aproximate any function...
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        return {"logits": self.net(spectrogram)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here


class MyLSTM(BaseModel):
    def __init__(self, n_feats, n_class, n_layers=3, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.encoder = nn.LSTM(n_feats, fc_hidden, batch_first=True, num_layers=n_layers)
        self.head = nn.Linear(fc_hidden, n_class)

    def forward(self, spectrogram, *args, **kwargs):
        x, _ = self.encoder(spectrogram)
        x = self.head(x)
        return x

    def transform_input_lengths(self, input_lengths):
        return input_lengths

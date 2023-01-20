from torch import nn
import copy
import torch


class DQN(nn.Module):
    def __init__(self, input_shape, n_outputs):
        super().__init__()
        self._input_shape = input_shape
        self._n_outputs = n_outputs

        # determine if device is cuda or cpu
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # verify input shape
        try:
            self._input_shape = tuple(self._input_shape)
        except TypeError:
            self._input_shape = (self._input_shape, )
        if len(self._input_shape) != 3 or self._input_shape[1:] != (84, 84):
            raise ValueError('Invalid input shape: {}.'.format(self._input_shape))
        ch, h, w = self._input_shape

        # create online learning network
        self._layers = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self._n_outputs),
        )

    def deserialize(self, serialization):
        if serialization.get('type') != self.__class__.__name__:
            raise ValueError('Incompatible serialization.')
        if '_input_shape' in serialization:
            self._input_shape = serialization['_input_shape']
        if '_n_outputs' in serialization:
            self._n_outputs = serialization['_n_outputs']
        if '_state_dict' in serialization:
            self.load_state_dict(serialization['_state_dict'])

    def serialize(self):
        return {
            'type':             self.__class__.__name__,
            '_input_shape':     self._input_shape,
            '_n_outputs':       self._n_outputs,
            '_state_dict':      self.state_dict(),
        }

    def forward(self, x):
        return self._layers(x)


class DoubleDQN(DQN):
    def __init__(self, input_shape, n_outputs):
        super().__init__(input_shape, n_outputs)

        # duplicate for offline network and freeze it
        self._offline_layers = copy.deepcopy(self._layers)
        for param in self._offline_layers.parameters():
            param.requires_grad = False

    def forward(self, x, eval_type='online'):
        if eval_type.lower() == 'online':
            return self._layers(x)
        elif eval_type.lower() == 'offline':
            return self._offline_layers(x)
        raise ValueError('Invalid type of evaluation.')

    def sync(self):
        self._offline_layers.load_state_dict(self._layers.state_dict())
        for param in self._offline_layers.parameters():
            param.requires_grad = False

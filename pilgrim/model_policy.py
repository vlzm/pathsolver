import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1, activation_function="relu", use_batch_norm=True):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else None
        self.activation = self._get_activation_function(activation_function)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else None
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        out += residual
        out = self.activation(out)
        return out

    @staticmethod
    def _get_activation_function(name):
        if name == "relu":
            return nn.ReLU()
        elif name == "mish":
            return nn.Mish()
        else:
            raise ValueError(f"Unknown activation function: {name}")


class PilgrimPolicy(nn.Module):
    """
    Policy-сеть, которая вместо оценки 'value' возвращает
    логиты для всех возможных действий.
    """
    def __init__(
        self,
        state_size,
        n_actions,             # <-- Количество действий
        hd1=5000,
        hd2=1000,
        nrd=2,
        dropout_rate=0.1,
        activation_function="relu",
        use_batch_norm=True
    ):
        super(PilgrimPolicy, self).__init__()
        self.hd1 = hd1
        self.hd2 = hd2
        self.nrd = nrd
        self.use_batch_norm = use_batch_norm

        self.input_layer = nn.Linear(state_size, hd1)
        self.bn1 = nn.BatchNorm1d(hd1) if use_batch_norm else None
        self.activation = self._get_activation_function(activation_function)
        self.dropout = nn.Dropout(dropout_rate)

        if hd2 > 0:
            self.hidden_layer = nn.Linear(hd1, hd2)
            self.bn2 = nn.BatchNorm1d(hd2) if use_batch_norm else None
            hidden_dim_for_output = hd2
        else:
            self.hidden_layer = None
            self.bn2 = None
            hidden_dim_for_output = hd1

        if nrd > 0 and hd2 > 0:
            self.residual_blocks = nn.ModuleList([
                ResidualBlock(hd2, dropout_rate, activation_function, use_batch_norm)
                for _ in range(nrd)
            ])
        else:
            self.residual_blocks = None

        # Вместо 1 выхода (value) делаем выход в n_actions.
        self.output_layer = nn.Linear(hidden_dim_for_output, n_actions)

    def forward(self, z):
        x = z.float()
        x = self.input_layer(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        if self.hidden_layer:
            x = self.hidden_layer(x)
            if self.bn2:
                x = self.bn2(x)
            x = self.activation(x)
            x = self.dropout(x)

        if self.residual_blocks:
            for block in self.residual_blocks:
                x = block(x)

        # На выходе теперь logits размером [batch_size, n_actions].
        # В PPO обычно возвращают именно logits, а softmax делается снаружи,
        # когда мы вычисляем распределение действий.
        logits = self.output_layer(x)

        return logits

    @staticmethod
    def _get_activation_function(name):
        if name == "relu":
            return nn.ReLU()
        elif name == "mish":
            return nn.Mish()
        else:
            raise ValueError(f"Unknown activation function: {name}")

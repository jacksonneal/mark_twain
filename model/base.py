from torch import nn


class Base(nn.Module):

    def __init__(self, params):
        super().__init__()
        dimensions = params.dimensions
        layers = []
        if params.initial_bn:
            layers.append(nn.BatchNorm1d(dimensions[0]))
        for i in range(len(dimensions) - 1):
            layers += [
                nn.Linear(dimensions[i], dimensions[i + 1], bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(dimensions[i + 1])
            ]
            if params.dropout > 0:
                layers.append(nn.Dropout(p=params.dropout))
        layers.append(nn.Linear(dimensions[-1], 1 + len(params.aux_target_cols)))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

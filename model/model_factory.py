from torch import nn


def build_model(params):
    if params.model_name == "BASE":
        layers = []

        if params.feature_set is not None:
            num_features = [params.num_features, 20, 10]
        else:
            num_features = [1050, 200, 100]

        if params.initial_bn:
            layers.append(nn.BatchNorm1d(num_features[0]))

        for i in range(len(num_features) - 1):
            layers += [
                nn.Linear(num_features[i], num_features[i + 1], bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(num_features[i + 1])
            ]

            if params.dropout > 0:
                layers.append(nn.Dropout(p=params.dropout))

        layers.append(nn.Linear(num_features[-1], 1 + len(params.aux_target_cols)))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)
    else:
        raise Exception(f"unsupported model name {params.model_name}")

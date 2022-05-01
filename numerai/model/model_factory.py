from numerai.model.aemlp import AEMLP
from numerai.model.base import Base
from numerai.model.tmlp import TransformerMLP
from numerai.model.cae import CAE


def build_model(params):
    if params.model_name == "BASE":
        return Base(params)
    elif params.model_name == "AEMLP" or params.model_name == "AE-MLP":
        return AEMLP(params)
    elif params.model_name == "TMLP":
        return TransformerMLP(params)
    elif params.model_name == "CAE":
        return CAE(params)
    else:
        raise Exception(f"unsupported model name {params.model_name}")

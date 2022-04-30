from numerai.model.aemlp import AEMLP
from numerai.model.base import Base
from numerai.model.tmlp import TransformerMLP


def build_model(params):
    if params.model_name == "BASE":
        return Base(params)
    elif params.model_name == "AEMLP":
        return AEMLP(params)
    elif params.model_name == "TMLP":
        return TransformerMLP(params)
    else:
        raise Exception(f"unsupported model name {params.model_name}")

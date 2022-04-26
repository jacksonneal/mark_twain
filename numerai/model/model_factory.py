from numerai.model.aemlp import AEMLP
from numerai.model.base import Base


def build_model(params):
    if params.model_name == "BASE":
        return Base(params)
    elif params.model_name == "AE-MLP":
        return AEMLP(params)
    else:
        raise Exception(f"unsupported model name {params.model_name}")

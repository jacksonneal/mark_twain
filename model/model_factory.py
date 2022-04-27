# from model.ae import AE
from model.base import Base
from model.autoencoder import AutoEncoder
from model.ae_test import AE
from model.AE_CONV import AEConv


def build_model(params):
    if params.model_name == "BASE":
        return Base(params)
    elif params.model_name == "AE":
        return AE(params)
    elif params.model_name == "AUTO":
        return AutoEncoder(params)
    elif params.model_name == "AECONV":
        return AEConv(params)
    else:
        raise Exception(f"unsupported model name {params.model_name}")

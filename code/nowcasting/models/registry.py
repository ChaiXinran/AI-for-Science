from nowcasting.models import nowcastnet
from nowcasting.models.nowcastnet_pwv import PWVCoupledNet
from nowcasting.models.nowcastnet_pwv_v2 import PWVCoupledNetV2
from nowcasting.models.nowcastnet_pwv_v3 import PWVCoupledNetV3
from nowcasting.models.nowcastnet_pwv_v3_object import PWVCoupledNetV3Object
from nowcasting.models.nowcastnet_pwv_v4 import PWVCoupledNetV4


MODEL_REGISTRY = {
    "NowcastNet": nowcastnet.Net,
    "nowcasting": nowcastnet.Net,
    "PWVCoupledNowcastNet": PWVCoupledNet,
    "PWVCoupledNowcastNetV2": PWVCoupledNetV2,
    "PWVCoupledNowcastNetV3": PWVCoupledNetV3,
    "PWVCoupledNowcastNetV3Object": PWVCoupledNetV3Object,
    "PWVCoupledNowcastNetV4": PWVCoupledNetV4,
}


def available_models():
    return sorted(MODEL_REGISTRY)


def get_model_class(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            "Name of network unknown {}. Available models: {}".format(
                model_name,
                ", ".join(available_models()),
            )
        )
    return MODEL_REGISTRY[model_name]


def build_model(configs):
    return get_model_class(configs.model_name)(configs)

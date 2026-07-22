from nowcasting.models import nowcastnet
from nowcasting.models.nowcastnet_pwv import (
    PWVBirthGrowthNet,
    PWVContrastiveTriggerNet,
    PWVCoupledNet,
)
from nowcasting.models.nowcastnet_pwv_object import PWVCoupledNetObject


MODEL_REGISTRY = {
    "NowcastNet": nowcastnet.Net,
    "nowcasting": nowcastnet.Net,
    "PWVCoupledNowcastNet": PWVCoupledNet,
    "PWVBirthGrowthNowcastNet": PWVBirthGrowthNet,
    "PWVContrastiveTriggerNowcastNet": PWVContrastiveTriggerNet,
    "PWVCoupledNowcastNetObject": PWVCoupledNetObject,
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

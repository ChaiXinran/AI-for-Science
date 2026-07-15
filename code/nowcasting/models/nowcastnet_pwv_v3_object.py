from nowcasting.models.nowcastnet_pwv_v3 import PWVCoupledNetV3
from nowcasting.models.object_head import ObjectPredictionHead


class PWVCoupledNetV3Object(PWVCoupledNetV3):
    """V3 backbone with an auxiliary convective-object prediction head."""

    def __init__(self, configs):
        super(PWVCoupledNetV3Object, self).__init__(configs)
        self.object_head = ObjectPredictionHead(
            self.pred_length,
            base_channels=getattr(configs, "object_head_base_channels", 24),
            intensity_scale=getattr(configs, "intensity_scale", 128.0),
        )

    def forward(self, all_frames, pwv_frames=None, return_aux=False):
        aux = super(PWVCoupledNetV3Object, self).forward(
            all_frames,
            pwv_frames=pwv_frames,
            return_aux=True,
        )
        aux["object"] = self.object_head(aux)
        if return_aux:
            return aux
        return aux["prediction"]

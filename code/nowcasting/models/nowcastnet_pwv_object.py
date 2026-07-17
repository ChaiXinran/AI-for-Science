from nowcasting.models.nowcastnet_pwv import PWVCoupledNet
from nowcasting.models.object_head import ObjectPredictionHead


class PWVCoupledNetObject(PWVCoupledNet):
    """PWVCoupledNet with an auxiliary convective-object prediction head."""

    def __init__(self, configs):
        super(PWVCoupledNetObject, self).__init__(configs)
        self.object_head = ObjectPredictionHead(
            self.pred_length,
            base_channels=getattr(configs, "object_head_base_channels", 24),
            intensity_scale=getattr(configs, "intensity_scale", 128.0),
        )

    def forward(self, all_frames, pwv_frames=None, return_aux=False):
        aux = super(PWVCoupledNetObject, self).forward(
            all_frames,
            pwv_frames=pwv_frames,
            return_aux=True,
        )
        aux["object"] = self.object_head(aux)
        if return_aux:
            return aux
        return aux["prediction"]

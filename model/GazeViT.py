
import torch.nn as nn
from .Deit import deit_small_distilled_patch16_224

class GazeViT(nn.Module):
    
    def __init__(self,  args, base_encoder=None):
        
        super(GazeViT, self).__init__()
        self.dim = args.dim

        self.size_sat = [256, 256]
        self.size_sat_default = [256, 256]
        self.size_grd = [112, 224]

        if args.sat_res != 0:
            self.size_sat = [args.sat_res, args.sat_res]

        encoder_model = deit_small_distilled_patch16_224
        self.street_net = encoder_model(crop=False, img_size=self.size_grd, num_classes=args.dim)
        self.aerial_net = encoder_model(crop=args.crop, img_size=self.size_sat, num_classes=args.dim)

    def forward(self, im_s, im_a, delta=None, atten=None, indexes=None, heatmap = None):
        if atten is not None:
            return self.street_net(im_s), self.aerial_net(x=im_a, atten=atten, heatmap=heatmap)
        else:
            return self.street_net(im_s), self.aerial_net(x=im_a, indexes=indexes)    

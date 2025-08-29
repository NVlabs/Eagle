import torch, os
import torch.nn as nn
from timm import create_model
from transformers import CLIPImageProcessor
from .convnext import convnext_xxlarge
from torch.utils.checkpoint import checkpoint
import torch
from torchvision import transforms as T
from PIL import Image



cfg={
    "crop_size": 256,
    "do_center_crop": True,
    "do_normalize": True,
    "do_resize": True,
    "feature_extractor_type": "CLIPFeatureExtractor",
    "image_mean": [
        0.48145466,
        0.4578275,
        0.40821073
    ],
    "image_std": [
        0.26862954,
        0.26130258,
        0.27577711
    ],
    "resample": 3,
    "size": 256
}



MEAN_SLIP = [0.5, 0.5, 0.5]
STD_SLIP = [0.5, 0.5, 0.5]

MEAN_CLIP = [0.48145466, 0.4578275, 0.40821073]
STD_CLIP = [0.26862954, 0.26130258, 0.27577711]


a = [s_slip / s_clip for s_slip, s_clip in zip(STD_SLIP, STD_CLIP)]
b = [(m_slip - m_clip) / s_clip for m_slip, m_clip, s_clip in zip(MEAN_SLIP, MEAN_CLIP, STD_CLIP)]


class SlipToClipTransform:
    def __init__(self, a, b):
        self.a = torch.tensor(a).view(-1, 1, 1)
        self.b = torch.tensor(b).view(-1, 1, 1)
    
    def __call__(self, x_slip):
        return x_slip * self.a.to(x_slip.device) + self.b.to(x_slip.device)
slip_to_clip = SlipToClipTransform(a, b)

class ConvNextVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, normalize_type=None):
        super().__init__()

        self.is_loaded = False
        self.freeze_vision=args.freeze_vision
        self.input_image_size=args.input_image_size
        self.vision_tower_name = vision_tower
        self.name = 'convnext'
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.pre_norm = normalize_type

        print('pre_norm: ', self.pre_norm)
        self.delay_load = delay_load
        self.load_model()

    def load_model(self):
        if 'xxlarge' in self.vision_tower_name:
            if self.delay_load:
                self.vision_tower = convnext_xxlarge(pretrained=False)
            else:
                self.vision_tower = convnext_xxlarge(self.vision_tower_name)
            setattr(self.vision_tower, 'hidden_size', 3072)
        elif os.path.exists(self.vision_tower_name):
            self.vision_tower = torch.load(self.vision_tower_name)
        else:
            assert False, 'Not implemented'


        self.vision_tower = self.vision_tower.to(torch.bfloat16)

        if self.freeze_vision:
            self.vision_tower.requires_grad_(False)

        # if self.vision_tower.grad_checkpointing:
        for s in self.vision_tower.stages:
            s.grad_checkpointing = True

        self.is_loaded = True

    def feature_select(self, image_forward_outs):

        if self.select_layer>100:
            image_features = image_forward_outs[-4:]
        else:
            image_features = image_forward_outs[-1]
        return image_features

    def forward_features(self, x):
        x = self.vision_tower.stem(x)
        image_forward_out=[]
        for blk in self.vision_tower.stages:
            x = blk(x)
            b,c,h,w=x.shape
            image_forward_out.append(x.view(b,c,-1).transpose(1,2))
        return image_forward_out

    def forward(self, images):
        if self.freeze_vision:
            with torch.no_grad():
                image_features = self._forward_images(images)
        else:
            image_features = self._forward_images(images)

        return image_features

    def _forward_images(self, images):
        
        if type(images) is list:
            image_features = []
            for image in images:
                if self.pre_norm == 'siglip':
                    dtype = image.dtype
                    image = slip_to_clip(image.to(torch.float32)).to(dtype)
                image_forward_out = self.forward_features(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out)
                image_features.append(image_feature)
        else:
            if self.pre_norm == 'siglip':
                dtype = images.dtype
                images = slip_to_clip(images.to(torch.float32)).to(dtype)
            image_forward_outs = self.forward_features(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        assert  NotImplementedError
        pass

    @property
    def num_attention_heads(self):
        # as constant
        return 16
    @property
    def num_layers(self):
        # as constant
        return 4
    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size

    @property
    def num_patches(self):
        return (self.input_image_size // self.patch_embed.patch_size[0]) ** 2


class ConvNextFPNVisionTower(nn.Module):
    def __init__(self, 
                 vision_tower, 
                 args, 
                 fpn_target_level=1,
                 fpn_layer_idx=[1,2,3],
                 fpn_input_dim=[768,1536,3072],
                 delay_load=False):
        
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower.replace('-fpn', 'fpn')
        self.freeze_vision = getattr(args, "frozen_backbone", True)
        # self.input_image_size = getattr(args, "vision_tower_input_size", 1024) 
        self.input_image_size = 1024  # hardcode
        self.select_layer = args.mm_vision_select_layer # no effect
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.need_fpn = True
        self.fpn_layer_idx = fpn_layer_idx # [1, 2, 3] # x8, x16, x32
        self.fpn_input_dim = [768, 1536, 3072]
        self.delay_load = delay_load
        self.load_model()

    def load_model(self):
        if self.is_loaded:
            return
        
        self.image_processor = CLIPImageProcessor(**cfg)
        if 'xxlarge' in self.vision_tower_name:
            self.vision_tower = convnext_xxlarge(self.vision_tower_name)
            setattr(self.vision_tower, 'hidden_size', self.fpn_input_dim)
            # setattr(self.vision_tower, 'hidden_size', 3072)
        else:
            self.vision_tower = convnext_large_mlp(self.vision_tower_name)
            setattr(self.vision_tower, 'hidden_size', 1536)
        if self.freeze_vision:
            self.vision_tower.requires_grad_(False)

        # if self.vision_tower.grad_checkpointing:
        for s in self.vision_tower.stages:
            s.grad_checkpointing = True

        if self.input_image_size is not None:
            self.image_processor.size=self.input_image_size
            self.image_processor.crop_size={
                'height':self.input_image_size,
                'width': self.input_image_size
            }

        self.is_loaded = True

    @torch.no_grad()
    def forward_features(self, x):
        x = self.vision_tower.stem(x)
        image_forward_out=[]
        for blk in self.vision_tower.stages:
            x = blk(x)
            image_forward_out.append(x)
        return image_forward_out

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.forward_features(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_features.append(image_feature)
        else:
            image_features = self.forward_features(images.to(device=self.device, dtype=self.dtype))
            image_features = [image_features[idx] for idx in self.fpn_layer_idx]

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        assert  NotImplementedError
        pass

    @property
    def num_attention_heads(self):
        # as constant
        return 16
    @property
    def num_layers(self):
        # as constant
        return 4
    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size

    @property
    def num_patches(self):
        return (cfg['image_size'] // self.patch_embed.patch_size[0]) ** 2

if __name__ == '__main__':
    COMBINED_STD = [s_slip / s_clip for s_slip, s_clip in zip(STD_SigLIP, STD_CLIP)]
    COMBINED_MEAN = [(m_slip - m_clip) / s_clip for m_slip, m_clip, s_clip in zip(MEAN_SigLIP, MEAN_CLIP, STD_CLIP)]

    # 定义合并的归一化变换
    combined_normalize = T.Normalize(mean=COMBINED_MEAN, std=COMBINED_STD)
    x = torch.randn(1, 3, 256, 256).cuda()
    a = normalize_clip(x).to(torch.bfloat16)
    b = normalize_siglip(x).to(torch.bfloat16)
    c = denormalize_siglip(b.to(torch.float32))
    c2 = normalize_clip(c).to(torch.bfloat16)
    c3 = combined_normalize(b)
    print((c-x).abs().max())
    print((c2-a).abs().max())
    print((c3-a).abs().max())
    from IPython import embed
    embed()
    exit()
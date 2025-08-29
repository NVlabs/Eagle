import torch, os
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .siglip_vision_tower import SiglipVisionTower
from internvl.model.c_radio import RADIOModel, RADIOConfig
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from copy import deepcopy
import random
import math

class MultiBackboneChannelConcatenationVisionTower(nn.Module):
    def __init__(self,
                 vision_tower,
                 args,
                 grid_size=32,
                 convnext_img_size=512,
                 radio_img_size=512,
                 siglip_img_size=448,
                 normalize_type=None,
                 raw_config=None):
        
        super().__init__()

        self.is_loaded = False
        self.grid_size = grid_size
        self.num_tokens = self.grid_size ** 2
        self.normalize_type = args.normalize_type
        self.moe_version_type = args.moe_version_type
        self.raw_config = raw_config
        print("moe_version_type: ", self.moe_version_type)
        assert self.moe_version_type in [None, 'all_tiling', 'seq_concat', 'feat_concat', 'convnext_512_siglip_448', 'radio_448_siglip_448'], f"Unknown self.moe_version_type: {self.moe_version_type}"
        
        vision_tower_name_list = vision_tower.split(";")
        # self.input_image_size = 1024
        self.siglip_img_size = siglip_img_size
        self.convnext_img_size = convnext_img_size
        self.radio_img_size = radio_img_size
        self.load_vision_towers(vision_tower_name_list, args)

      
    def load_vision_towers(self, vision_tower_name_list, args):
        self.vision_towers = nn.ModuleList()

        freeze_backbone_list = args.freeze_backbones # note this is a str
        if freeze_backbone_list is not None and len(freeze_backbone_list) > 0:
            print("The frozen backbones: ", freeze_backbone_list)
        else:
            # make it a blank str
            freeze_backbone_list = ""
        for name in vision_tower_name_list:
            
            ## ConvNeXt
            if name == 'convnext-1024':
                convnext_args = deepcopy(args)

                convnext_args.freeze_vision = False
                if 'convnext-1024' in freeze_backbone_list:
                    convnext_args.freeze_vision = True

                from .convnext_encoder import ConvNextVisionTower
                convnext_args.input_image_size = self.convnext_img_size
                convnext_vision_tower = args.vision_tower_convnext_path
                convnext_vision_tower = ConvNextVisionTower(convnext_vision_tower, 
                                                                convnext_args, delay_load=args.delay_load, normalize_type=self.normalize_type)
                convnext_vision_tower.load_model()      
                self.vision_towers.append(convnext_vision_tower)

            ## PaliSigLIP
            elif name == 'palisiglip':
                palisiglip_args = deepcopy(args)
                palisiglip_args.input_image_size = self.siglip_img_size

                palisiglip_args.freeze_vision = False
                if 'palisiglip' in freeze_backbone_list:
                    palisiglip_args.freeze_vision = True

                palisiglip_vision_tower = SiglipVisionTower(args.vision_tower_siglip_path, palisiglip_args, delay_load=args.delay_load, raw_config=self.raw_config)     
   
                palisiglip_vision_tower.load_model()
                self.vision_towers.append(palisiglip_vision_tower)
                
            elif 'radio' in name:
                if args.delay_load:
                    radio_vision_config = args.radio_vision_config
                    if isinstance(radio_vision_config, dict):
                        radio_vision_config = RADIOConfig(**radio_vision_config)
                    radio_vision_model = RADIOModel(radio_vision_config)
                else:
                    radio_vision_model = RADIOModel.from_pretrained(
                        args.vision_tower_radio_path, torch_dtype=torch.bfloat16, config=args.radio_vision_config )
                radio_vision_model.input_image_size = self.radio_img_size
                self.vision_towers.append(radio_vision_model)

        # Set the image processor
        self.image_processor = None
        self.is_loaded = True

    def load_model(self):
        assert self.is_loaded, "All the vision encoders should be loaded during initialization!"

    def forward(self, x):
        # x is a Tensor if moe_version_type is None or 'all_tiling'
        # else is a tuple(Tensor, Tensor)
        if isinstance(x, dict):
            x = x['pixel_values']
        if self.moe_version_type in [None, 'all_tiling']:
           assert False, "This is not implemented"
        elif 'convnext' in self.moe_version_type or 'radio' in self.moe_version_type:
            features = {}
            image_input_size = x.shape[2]
            assert x.shape[2] == x.shape[3], f"Image should be a square but size ({x.shape[2]} x {x.shape[3]})"
            
            for vision_tower in self.vision_towers:
                if vision_tower.input_image_size != image_input_size:
                    resized_x = F.interpolate(x.float(), 
                                            size=(vision_tower.input_image_size, vision_tower.input_image_size), 
                                            mode='bilinear', 
                                            align_corners=True).to(dtype=x.dtype)
                else:
                    resized_x = x
                
                feature = vision_tower(resized_x)
                features[vision_tower.name] = feature
        else:
            assert False, "This is not implemented"
        return features
        
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.clip_vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.clip_vision_tower.parameters()).device

    @property
    def config(self):
        assert NotImplementedError
        pass

    @property
    def hidden_size(self):
        if self.moe_version_type == 'convnext_512_siglip_448' or self.moe_version_type == 'radio_448_siglip_448':
            res = {}
            for vision_tower in self.vision_towers:
                res[vision_tower.name] = vision_tower.hidden_size
            return res
        else:
            return sum([_.hidden_size for _ in self.vision_towers])

    @property
    def num_patches(self):
        return self.num_tokens




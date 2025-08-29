import torch
from torchvision import transforms as T
from PIL import Image

# 定义 SLIP 和 CLIP 的均值和标准差
MEAN_SLIP = [0.5, 0.5, 0.5]
STD_SLIP = [0.5, 0.5, 0.5]

MEAN_CLIP = [0.48145466, 0.4578275, 0.40821073]
STD_CLIP = [0.26862954, 0.26130258, 0.27577711]

# 计算从 x_slip 到 x_clip 的转换参数
a = [s_slip / s_clip for s_slip, s_clip in zip(STD_SLIP, STD_CLIP)]
b = [(m_slip - m_clip) / s_clip for m_slip, m_clip, s_clip in zip(MEAN_SLIP, MEAN_CLIP, STD_CLIP)]
print(a, b)
# 定义自定义的线性变换
class SlipToClipTransform:
    def __init__(self, a, b):
        self.a = torch.tensor(a).view(-1, 1, 1)
        self.b = torch.tensor(b).view(-1, 1, 1)
    
    def __call__(self, x_slip):
        return x_slip * self.a + self.b

# 定义图像预处理变换
preprocess = T.Compose([
    T.Resize((1024, 1024)),
    T.ToTensor(),
    T.Normalize(mean=MEAN_SLIP, std=STD_SLIP),  # 应用 SLIP 归一化
])

# 加载图像
img = Image.open('/home/zhiqil/workspace/tensor_after_pixel_shuffle.png').convert('RGB')

# 应用预处理和 SLIP 归一化
tensor_slip = preprocess(img)

# 应用从 SLIP 到 CLIP 的转换
slip_to_clip = SlipToClipTransform(a, b)
tensor_clip_transformed = slip_to_clip(tensor_slip.to(torch.float32))

# 直接对原始图像应用 CLIP 归一化
preprocess_clip = T.Compose([
    T.Resize((1024, 1024)),
    T.ToTensor(),
    T.Normalize(mean=MEAN_CLIP, std=STD_CLIP)
])
tensor_clip_direct = preprocess_clip(img)

# 比较两个张量的最大差异
difference = torch.abs(tensor_clip_transformed - tensor_clip_direct).max()
print(f'最大差异：{difference.item()}')

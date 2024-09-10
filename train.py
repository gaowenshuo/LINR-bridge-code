import torch
from torch import nn, optim
from siren_pytorch import SirenNet, SirenWrapper
from PIL import Image
import numpy as np
from tqdm import tqdm
from videosdsloss import SDSVideoLoss
import os


import argparse
parser = argparse.ArgumentParser(description="Control script with command line arguments")
parser.add_argument('--input_path', type=str, required=True, help='Path to the input image')
parser.add_argument('--caption', type=str, required=True, help='Caption for the process')
parser.add_argument('--model_name', type=str, required=True, help='Video diffusion model name')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the output image')
parser.add_argument('--k_frames', type=int, default=12, help='Number of frames')
parser.add_argument('--train_iter', type=int, default=10000, help='Iteration of training')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

# 使用解析的参数
input_path = args.input_path
# color initialize
import json
# 从文件中读取 JSON 数据
with open(input_path+".json", 'r') as f:
    colors = json.load(f)
color_channel_num = len(colors)

##

caption = args.caption
model_name = args.model_name
output_path = args.output_path
if not os.path.exists(output_path):
    os.makedirs(output_path)
import json
args_path = os.path.join(output_path, 'args.json')
with open(args_path, 'w') as f:
    json.dump(vars(args), f, indent=4)
# 定义 SIREN 网络
class ClipSigmoid(nn.Module):
    def __init__(self):
        super(ClipSigmoid, self).__init__()
    def forward(self, x):
        sigmoid_output = (torch.sigmoid(x) - 0.5) * 1.0000001 + 0.5
        return torch.clamp(sigmoid_output, min=0.0, max=1.0)
net = SirenNet(
    dim_in=2,                        # 输入维度，例如2D坐标
    dim_hidden=64,                  # 隐藏层维度
    dim_out=color_channel_num,
    num_layers=4,                    # 层数
    final_activation = ClipSigmoid(),
    w0_initial=30.                   # 超参数
).to("cuda")
# 包装器
wrapper = SirenWrapper(
    net,
    image_width=256,
    image_height=256,
    colors=colors
).to("cuda")
device=wrapper().device
print(f"Using device: {device}")
# 读取目标图像
img_path = input_path
target_img = Image.open(img_path)
target_img = target_img.resize((256, 256))  # 调整图像大小
target_img = np.array(target_img) / 255.0  # 归一化图像到 [0, 1]
# 将目标图像转换为 torch tensor
target_img_tensor = torch.tensor(target_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

layer_imgs = []
file_name = os.path.basename(input_path)  # 获取文件名
prefix = file_name.split('.')[0]  # 提取文件名中'.'之前的部分作为前缀
base_path = os.path.dirname(input_path)
for i in range(color_channel_num):
    layer_file_path = os.path.join(base_path, f"{prefix}_layer/layer_{i}.png")
    target_layer_img = Image.open(layer_file_path)
    target_layer_img = target_layer_img.resize((256, 256))  # 调整图像大小
    target_layer_img = np.array(target_layer_img) / 255.0  # 归一化图像到 [0, 1]
    layer_imgs.append(torch.tensor(target_layer_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device))



# 初始化损失函数和优化器
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(wrapper.parameters(), lr=1e-3)

# 训练模型
num_epochs = 10000
rangetqdm = tqdm(range(num_epochs), desc="Reconstruction")
for epoch in rangetqdm:
    optimizer.zero_grad()
    loss, loss_entropy = wrapper(target_img_tensor, return_entropy = True)
    if float(epoch) / float(num_epochs) > 0.5:
        loss += ((float(epoch) / float(num_epochs)) ** 2 * 2e0 * loss_entropy)
    for param in wrapper.parameters():
        loss += 0.000001 * torch.sum(param ** 2) #l2_reg_loss
    for i in range(color_channel_num):
        loss += (wrapper.show_layer(n=i, img=layer_imgs[i]))
    rangetqdm.set_postfix({"loss":"{0:1.5f}".format(loss.item())})
    loss.backward()
    max_norm = 2.0  # 设置最大范数
    torch.nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm)
    optimizer.step()
    

# 使用训练好的模型生成图像
with torch.no_grad():
    pred_img = wrapper()
    pred_img = pred_img.squeeze().permute(1, 2, 0).cpu().numpy()

# 将生成的图像保存为文件
pred_img = (pred_img * 255).astype(np.uint8)
output_img = Image.fromarray(pred_img)
output_img.save(os.path.join(output_path, 'reconstruction.png'))

# 看看每一层
for i in range(color_channel_num):
    # 使用训练好的模型生成图像
    with torch.no_grad():
        pred_img = wrapper.show_layer(n=i)
        pred_img = pred_img.squeeze().permute(1, 2, 0).cpu().numpy()

    # 将生成的图像保存为文件
    pred_img = (pred_img * 255).astype(np.uint8)
    output_img = Image.fromarray(pred_img)
    output_img.save(os.path.join(output_path, f'reconstruction_layer{i}.png'))

print("reconstruction complete")

### video diffusion

import torch
import numpy as np
import imageio
from PIL import Image

def generate_and_save_gif(wrappers, K_frames, n_iter):
    path = os.path.join(output_path, f"iter_{n_iter+1}")
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(K_frames):
        with torch.no_grad():
            pred_img = wrappers[i]()
            pred_img = pred_img.squeeze().permute(1, 2, 0).cpu().numpy()
        pred_img = (pred_img * 255).clip(0, 255).astype(np.uint8)
        output_img = Image.fromarray(pred_img)
        output_img.save(os.path.join(path, f'frame_{i}.png'))
    gif_path = os.path.join(path, "result.gif")
    # 读取所有PNG文件
    frames = []
    for i in range(K_frames):
        frame_path = os.path.join(path, f'frame_{i}.png')
        frames.append(Image.open(frame_path))
    # 保存为GIF
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,  # 每帧的持续时间（毫秒）
        loop=0  # 0表示无限循环
    )

video_sds_loss = SDSVideoLoss(caption, model_name, device=wrapper().device)

K_frames = args.k_frames
Train_iter = args.train_iter

nets = [SirenNet(dim_in=2, dim_hidden=64, dim_out=color_channel_num, num_layers=4, final_activation = ClipSigmoid(), w0_initial=30.).to(device) for i in range(K_frames)]

wrappers = [SirenWrapper(nets[i],image_width=256,image_height=256,colors=colors).to(device) for i in range(K_frames)]

while True:
    flag = True
    for i in range(K_frames):
        wrappers[i].load_state_dict(wrapper.state_dict())

    all_params = []
    for i in range(K_frames):
        all_params.extend(wrappers[i].parameters())

    # 创建优化器
    optimizer = optim.Adam(all_params, lr=3e-5)
    import torchvision
    blurrer = torchvision.transforms.GaussianBlur(kernel_size=(11,11), sigma=(10,10))
    def drop_nans(params):
        if not torch.isfinite(params).all():
            print("Disturbed: Params contain non-finite values. Replacing NaN values. Try another seed.")
            assert False
        return torch.nan_to_num(params.detach().float(), 0.0, 1.0, -1.0)
    # 保存模型状态的函数
    def save_model_state():
        return None # deprecated
        return {param: param.clone() for param in all_params}
    # 恢复模型状态的函数
    def load_model_state(state):
        return # deprecated
        for param in all_params:
            param.data.copy_(state[param].data)
    # 初始化保存的模型状态
    last_valid_state = save_model_state() # deprecated
    # 训练模型
    num_epochs = Train_iter
    for epoch in tqdm(range(num_epochs), desc="video diffusion"):
        optimizer.zero_grad()
        outputs = []
        outputs.append(target_img_tensor) # 首尾各放一张原图
        loss_entropys = []
        for i in range(K_frames):
            raster_image, loss_entropy = wrappers[i](return_entropy = True)
            outputs.append(raster_image)
            loss_entropys.append(2e2 * loss_entropy)
        #outputs.append(target_img_tensor) # 首尾各放一张原图
        output_video = torch.stack(outputs, dim=1)  # [1, K, 3, 256, 256]
        loss = None
        try:
            loss = video_sds_loss(output_video)
        except Exception as e:  # 捕获所有异常
            print("Retrying")
        loss += torch.tensor(loss_entropys).mean()
        for param in all_params:
            loss += 0.000005 * torch.sum(param ** 2) #l2_reg_loss
        #for i in range(1, K_frames):
            #last_frame = blurrer(wrappers[i-1]().clone().detach())
            #this_frame = blurrer(wrappers[i]())
            #mse = nn.MSELoss()
            #loss += 1e3 * (mse(last_frame, this_frame))
        loss.backward()
        max_norm = 2.0  # 设置最大范数
        torch.nn.utils.clip_grad_norm_(all_params, max_norm)
        optimizer.step()
        try:
            for param in all_params:
                param = drop_nans(param)
            last_valid_state=save_model_state()
        except Exception as e:  # 捕获所有异常
            load_model_state(last_valid_state)
            flag = False
            break
        if (epoch + 1) % 100 == 0:
            generate_and_save_gif(wrappers, K_frames, epoch)
    if flag:
        break

# 使用训练好的模型生成图像
for i in range(K_frames):
    with torch.no_grad():
        pred_img = wrappers[i]()
        pred_img = pred_img.squeeze().permute(1, 2, 0).cpu().numpy()
# 将生成的图像保存为文件
    pred_img = (pred_img * 255).astype(np.uint8)
    output_img = Image.fromarray(pred_img)
    output_img.save(os.path.join(output_path, f'frame_{i}.png'))

print("video diffusion complete")

import os
from tqdm import tqdm
from easydict import EasyDict as edict
import torch
import pydiffvg
from PIL import Image
import argparse
import numpy as np
parser = argparse.ArgumentParser(description="Control script with command line arguments")
parser.add_argument('--input_path', type=str, required=True, help='Path to the input image')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the output image')
args = parser.parse_args()
from torch import nn, optim
input_path = args.input_path
output_path = args.output_path
K_frames = 1

import json
import imageio
from PIL import Image

# use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())
device = pydiffvg.get_device()

h, w = 256, 256

render = pydiffvg.RenderFunction.apply


parameters = edict()
parameters.point = []

def init_shapes(svg_path):
    svg = f'{svg_path}'
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(svg)
    target_width = w
    target_height = h
    factor_width = target_width / canvas_width
    factor_height = target_height / canvas_height
    min_factor = min(factor_height, factor_width)
    colors = []
    for group in shape_groups_init:
        colors.append([group.fill_color[0].item(), group.fill_color[1].item(), group.fill_color[2].item()])
    for path in shapes_init:
        path.points[:,0] -= (canvas_width / 2)
        path.points[:,0] *= (min_factor * 3 / 4)
        path.points[:,0] += (target_width / 2)
        path.points[:,1] -= (canvas_height / 2)
        path.points[:,1] *= (min_factor * 3 / 4)
        path.points[:,1] += (target_height / 2)
        path.stroke_width *= ((factor_width + factor_height)/2.0 * 3 / 4)
        path.points = path.points.to(device)
    for path in shapes_init:
        path.points.requires_grad = True
        parameters.point.append(path.points)
    return shapes_init, shape_groups_init, colors


shapes_frames = []
shape_groups_frames = []
merge_scheme = []
for i in range(1):
    shapes, shape_groups, colors = init_shapes(svg_path=input_path)
    # 合并相邻颜色相同的元素
    merged_colors = [colors[0]]
    for i in range(1, len(colors)):
        if colors[i] == merged_colors[-1]:
            merged_colors[-1] = colors[i]
        else:
            merged_colors.append(colors[i])
            merge_scheme.append(i)
    K_frames = len(merged_colors)
    merge_scheme.append(len(shapes))
    # 将列表存储为JSON
    json_array = json.dumps(merged_colors)
    # 将JSON保存到文件
    with open(output_path+".json", 'w') as f:
        f.write(json_array)
    shapes_init = shapes
    shape_groups_init = shape_groups
    print(merge_scheme)


for i in range(1):
    with torch.no_grad():
        # render image
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes_init, shape_groups_init)
        img = render(w, h, 2, 2, 0, None, *scene_args)
        # compose image with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        pred_img = img.cpu().numpy()
    pred_img = (pred_img * 255).astype(np.uint8)
    output_img = Image.fromarray(pred_img)
    output_img.save(os.path.join(output_path))

file_name = os.path.basename(output_path)  # 获取文件名
prefix = file_name.split('.')[0]  # 提取文件名中'.'之前的部分作为前缀
base_path = os.path.dirname(output_path)

for i in range(K_frames-1, -1, -1):
    with torch.no_grad():
        for path in shapes_init[merge_scheme[i]:]: # make invisible
            path.points[:,0] -= 1000
            path.points[:,1] -= 1000
        # render image
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes_init, shape_groups_init)
        img = render(w, h, 2, 2, 0, None, *scene_args)
        # compose image with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        pred_img = img.cpu().numpy()
    pred_img = (pred_img * 255).astype(np.uint8)
    output_img = Image.fromarray(pred_img)
    layer_path = os.path.join(base_path, f"{prefix}_layer")
    if not os.path.exists(layer_path):  # 如果文件夹不存在
        os.makedirs(layer_path)  # 创建文件夹
    output_img.save(os.path.join(base_path, f"{prefix}_layer/layer_{i}.png"))
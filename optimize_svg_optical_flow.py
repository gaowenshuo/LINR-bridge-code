import cv2
import torch
import numpy as np
import pydiffvg
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
parser.add_argument('--target_path', type=str, required=True, help='Path to the target image')
parser.add_argument('--k_frames', type=int, default=16, help='Number of frames')
parser.add_argument('--train_iter', type=int, default=10000, help='Iteration trained')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
from torch import nn, optim
input_path = args.input_path
output_path = args.output_path
target_path = args.target_path
K_frames = args.k_frames
train_iter = args.train_iter
if not os.path.exists(output_path):
    os.makedirs(output_path)
import json
import imageio
from PIL import Image
args_path = os.path.join(output_path, 'args.json')
with open(args_path, 'w') as f:
    json.dump(vars(args), f, indent=4)
# use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())
device = pydiffvg.get_device()

def compute_flows(reconstruction_file, frame_files):
    """
    计算光流并保存到内存中。

    参数:
    - reconstruction_file: reconstruction.png的文件路径
    - frame_files: frame_0.png到frame_5.png的文件路径列表

    返回:
    - flow_dict: 一个字典，键是图像文件名，值是对应的光流数组
    """
    reconstruction = cv2.imread(reconstruction_file, cv2.IMREAD_GRAYSCALE)
    flow_dict = {}

    for frame_file in frame_files:
        frame = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
        flow = cv2.calcOpticalFlowFarneback(reconstruction, frame, None, 
                                            pyr_scale=0.5, levels=3, winsize=15, 
                                            iterations=3, poly_n=5, poly_sigma=1.2, 
                                            flags=0)
        flow_dict[frame_file] = flow

    return flow_dict

def find_max_flow(flow, x, y):
    """
    在给定的浮点坐标附近寻找光流模长最大的点，并返回该点的位移。

    参数:
    - flow: 光流数组，形状为 (H, W, 2)
    - x, y: 输入的浮点坐标

    返回:
    - (new_x, new_y): 新的坐标点
    """
    h, w = flow.shape[:2]
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)

    candidates = [
        (x0, y0), (x0, y1),
        (x1, y0), (x1, y1)
    ]

    max_mag = 0
    best_offset = (0, 0)
    for (cx, cy) in candidates:
        flow_vec = flow[cy, cx]
        mag = np.linalg.norm(flow_vec)
        if mag > max_mag:
            max_mag = mag
            best_offset = flow_vec

    new_x = x + best_offset[0]
    new_y = y + best_offset[1]
    return new_x, new_y

def apply_flow_to_shapes(shapes_init, flow):
    """
    根据光流变换shapes_init中的点坐标，并原地替换。

    参数:
    - shapes_init: 初始的shapes列表
    - flow: 光流数组，形状为 (H, W, 2)
    """
    for path in shapes_init:
        points = path.points.cpu().numpy()
        transformed_points = []

        for point in points:
            x, y = point
            if 0 <= x < flow.shape[1] and 0 <= y < flow.shape[0]:
                new_x, new_y = find_max_flow(flow, x, y)
                transformed_points.append([new_x, new_y])
            else:
                transformed_points.append([x, y])

        path.points = torch.tensor(transformed_points, device=path.points.device)

def transform_svg_points_with_flow(svg_file, flow_dict):
    """
    变换SVG中的点坐标，并将其应用光流后原地替换。

    参数:
    - svg_file: SVG文件路径
    - flow_dict: 一个字典，键是图像文件名，值是对应的光流数组
    """
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(svg_file)
    target_width = 256
    target_height = 256
    factor_width = target_width / canvas_width
    factor_height = target_height / canvas_height
    min_factor = min(factor_height, factor_width)

    for i, (frame_file, flow) in enumerate(flow_dict.items()):
        # 为每个frame单独复制shapes_init
        import copy
        shapes_copy = copy.deepcopy(shapes_init)
        
        # 对点坐标进行初步转换
        for path in shapes_copy:
            path.points[:, 0] -= (canvas_width / 2)
            path.points[:, 0] *= (min_factor * 3 / 4)
            path.points[:, 0] += (target_width / 2)
            path.points[:, 1] -= (canvas_height / 2)
            path.points[:, 1] *= (min_factor * 3 / 4)
            path.points[:, 1] += (target_height / 2)
            path.stroke_width *= ((factor_width + factor_height) / 2.0 * 3 / 4)

        # 应用光流到转换后的点坐标
        apply_flow_to_shapes(shapes_copy, flow)
        
        # 将shapes_copy保存或进一步处理
        # 例如保存到文件或用于后续操作
        # pydiffvg.save_svg(f"transformed_svg_frame_{i}.svg", target_width, target_height, shapes_copy, shape_groups_init)
        from save_svg import save_svg
        save_svg(os.path.join(output_path, f"frame_{i}.svg"), 256, 256, shapes_copy, shape_groups_init)

# 示例使用
svg_file = input_path
reconstruction_img = os.path.join(target_path,"reconstruction.png")
frame_files = [os.path.join(target_path,f"frame_{i}.png") for i in range(K_frames)]
flows = compute_flows(reconstruction_img, frame_files)
transform_svg_points_with_flow(svg_file, flows)

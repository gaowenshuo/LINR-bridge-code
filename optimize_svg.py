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
    return shapes_init, shape_groups_init

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.spatial import Delaunay
import numpy as np
from torch.nn import functional as nnf
class ConformalLoss:
    def __init__(self, device: torch.device):
        self.device = device
    def get_template_angles(self, points: torch.Tensor) -> torch.Tensor:
        angles_ = []
        for i in range(len(self.template_faces)):
            triangles = points[self.template_faces[i]]
            triangles_roll_a = points[self.template_faces_roll_a[i]]
            edges = triangles_roll_a - triangles
            length = edges.norm(dim=-1)
            edges = edges / (length + 1e-1)[:, :, None]
            edges_roll = torch.roll(edges, 1, 1)
            cosine = torch.einsum('ned,ned->ne', edges, edges_roll)
            angles = torch.arccos(cosine)
            angles_.append(angles)
        return angles_
    def update_template(self, template_points) -> torch.Tensor:
        with torch.no_grad():
            self.template_faces = self.triangulate_template(template_points)
            self.template_faces_roll_a = [torch.roll(self.template_faces[i], 1, 1) for i in range(len(self.template_faces))]   
    def triangulate_template(self, template_points) -> torch.Tensor:
        faces_ = []
        points_np = [template_points[i].clone().detach().cpu().numpy() for i in range(len(template_points))]
        holes = []
        poly = Polygon(points_np[0], holes=holes)
        poly = poly.buffer(0)
        points_np = np.concatenate(points_np)
        faces = Delaunay(points_np).simplices
        is_intersect = np.array([poly.contains(Point(points_np[face].mean(0))) for face in faces], dtype=np.bool)
        faces_.append(torch.from_numpy(faces[is_intersect]).to(self.device, dtype=torch.int64))
        return faces_
    def __call__(self, parameters1, parameters2) -> torch.Tensor:
        loss_angles = 0
        points1 = torch.cat(parameters1)
        angles1 = self.get_template_angles(points1)
        points2 = torch.cat(parameters2)
        angles2 = self.get_template_angles(points2)
        for i in range(len(angles1)):
            loss_angles += nnf.mse_loss(angles1[i], angles2[i].detach().to(self.device))
        return loss_angles

conformal_loss = ConformalLoss(device)

print('initializing shape')
shapes_frames = []
shape_groups_frames = []
for i in range(K_frames):
    shapes, shape_groups = init_shapes(svg_path=input_path)
    shapes_frames.append(shapes)
    shape_groups_frames.append(shape_groups)
from copy import deepcopy
shapes_init, shape_groups_init = deepcopy(shapes_frames[0]), deepcopy(shape_groups_frames[0])

def get_points(shapes, return_paths=False):
    points = []
    points_paths = []
    count = 0
    for shape in shapes:
        points.append(shape.points)
        points_paths.append([shape.points])
        count = count + 1
    if return_paths:
        return count, points_paths
    return points

num_iter = int(train_iter / 25)
pg = [{'params': parameters["point"], 'lr': 0.5}]
optim = torch.optim.Adam(pg, betas=(0.9, 0.9), eps=1e-6)


print("start training")
# training loop
t_range = tqdm(range(num_iter))
shapes_last_frames = deepcopy(shapes_frames)
for step in t_range:
    target_img_tensor_frames = []
    for i in range(K_frames):
        target_img = Image.open(os.path.join(target_path,f"iter_{int(step/4+1)*100}/frame_{i}.png"))
        target_img = target_img.resize((256, 256))  # 调整图像大小
        target_img = np.array(target_img) / 255.0  # 归一化图像到 [0, 1]
        target_img_tensor = torch.tensor(target_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        target_img_tensor_frames.append(target_img_tensor)
    criterion = nn.MSELoss().to(device)
    optim.zero_grad()
    loss = 0
    for i in range(K_frames):
        # render image
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes_frames[i], shape_groups_frames[i])
        img = render(w, h, 2, 2, step, None, *scene_args)
        # compose image with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        x = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
        x = x.repeat(1, 1, 1, 1)
        loss += 1e2 * criterion(x, target_img_tensor_frames[i])
        if False: # use conformal
            # global conformal
            conformal_loss.update_template(get_points(shapes_init))
            loss_conformal = 1e-4 * conformal_loss(get_points(shapes_frames[i]), get_points(shapes_init))
            # local conformal
            count, points_paths_init = get_points(shapes_init, return_paths=True)
            count, points_paths = get_points(shapes_frames[i], return_paths=True)
            for j in range(count):
                conformal_loss.update_template(points_paths_init[j])
                loss_conformal += (conformal_loss(points_paths[j], points_paths_init[j]) * 1e2 / float(count))
            # compare with last iter
            if step > 0:
                # global conformal
                conformal_loss.update_template(get_points(shapes_last_frames[i]))
                loss_conformal += 1e-4 * conformal_loss(get_points(shapes_frames[i]), get_points(shapes_last_frames[i]))
                # local conformal
                count, points_paths_init = get_points(shapes_last_frames[i], return_paths=True)
                count, points_paths = get_points(shapes_frames[i], return_paths=True)
                for j in range(count):
                    conformal_loss.update_template(points_paths_init[j])
                    loss_conformal += (conformal_loss(points_paths[j], points_paths_init[j]) * 1e2 / float(count))
                if i > 0:
                    # global conformal
                    conformal_loss.update_template(get_points(shapes_last_frames[i-1]))
                    loss_conformal += 1e-4 * conformal_loss(get_points(shapes_frames[i]), get_points(shapes_last_frames[i-1]))
                    # local conformal
                    count, points_paths_init = get_points(shapes_last_frames[i-1], return_paths=True)
                    count, points_paths = get_points(shapes_frames[i], return_paths=True)
                    for j in range(count):
                        conformal_loss.update_template(points_paths_init[j])
                        loss_conformal += (conformal_loss(points_paths[j], points_paths_init[j]) * 1e2 / float(count))
                if i < K_frames-1:
                    # global conformal
                    conformal_loss.update_template(get_points(shapes_last_frames[i+1]))
                    loss_conformal += 1e-4 * conformal_loss(get_points(shapes_frames[i]), get_points(shapes_last_frames[i+1]))
                    # local conformal
                    count, points_paths_init = get_points(shapes_last_frames[i+1], return_paths=True)
                    count, points_paths = get_points(shapes_frames[i], return_paths=True)
                    for j in range(count):
                        conformal_loss.update_template(points_paths_init[j])
                        loss_conformal += (conformal_loss(points_paths[j], points_paths_init[j]) * 1e2 / float(count))
            loss += 2e-7 * loss_conformal
    loss.backward()
    optim.step()
    shapes_last_frames = deepcopy(shapes_frames)
from save_svg import save_svg
for i in range(K_frames):
    output = os.path.join(output_path, f"frame_{i}.svg")
    save_svg(output, w, h, shapes_frames[i], shape_groups_frames[i])
# save png
for i in range(K_frames):
    with torch.no_grad():
        # render image
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes_frames[i], shape_groups_frames[i])
        img = render(w, h, 2, 2, 0, None, *scene_args)
        # compose image with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        pred_img = img.cpu().numpy()
    pred_img = (pred_img * 255).astype(np.uint8)
    output_img = Image.fromarray(pred_img)
    output_img.save(os.path.join(output_path, f'frame_{i}.png'))



gif_path = os.path.join(output_path, "result.gif")

# 读取所有PNG文件
frames = []
for i in range(K_frames):
    frame_path = os.path.join(output_path, f'frame_{i}.png')
    frames.append(Image.open(frame_path))

# 保存为GIF
frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=200,  # 每帧的持续时间（毫秒）
    loop=0  # 0表示无限循环
)
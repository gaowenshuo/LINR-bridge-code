import pydiffvg
import numpy as np
import os
from PIL import Image
import torch
from save_svg import save_svg
import argparse
output_path = "omg"
w, h = 256, 256
render = pydiffvg.RenderFunction.apply

# 提取SVG中的点坐标
def extract_points(svg_path):
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
    points = []
    for shape in shapes:
        points.append(shape.points.cpu().numpy())
    return points, shapes, shape_groups

# 插值函数
def interpolate_points(points1, points2, t):
    # 确保两个路径的点数量相同
    if len(points1) != len(points2):
        raise ValueError("Points arrays must be of the same length")
    
    interpolated_points = []
    for p1, p2 in zip(points1, points2):
        interpolated_points.append(p1 * (1 - t) + p2 * t)
    return interpolated_points

# 生成插值帧
def generate_interpolated_frames(svg_files, num_interpolations):
    frames = []
    shapes_frames = []
    shape_groups_frames = []

    all_points = [extract_points(f) for f in svg_files]

    for i in range(len(all_points) - 1):
        for t in np.linspace(0, 1, num_interpolations):
            interpolated_points = interpolate_points(all_points[i][0], all_points[i + 1][0], t)
            
            # 复制原始 shapes 和 shape_groups
            import copy
            shapes = copy.deepcopy(all_points[i][1])
            shape_groups = copy.deepcopy(all_points[i][2])
            
            # 将插值点更新到 shapes 中
            for shape, points in zip(shapes, interpolated_points):
                shape.points = torch.tensor(points, device='cpu')
            
            shapes_frames.append(shapes)
            shape_groups_frames.append(shape_groups)
            frames.append((interpolated_points, shapes, shape_groups))

    return frames, shapes_frames, shape_groups_frames

# 将SVG文件转换为PNG
def save_png(output_path, i, shapes, shape_groups):
    with torch.no_grad():
        # 渲染图像
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
        img = render(3*w, 3*h, 2, 2, 0, None, *scene_args)
        # 使用白色背景合成图像
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=img.device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        pred_img = img.cpu().numpy()
    pred_img = (pred_img * 255).astype(np.uint8)
    output_img = Image.fromarray(pred_img)
    output_img.save(os.path.join(output_path, f'frame_{i}.png'))

# 生成GIF
def generate_gif(svg_files, output_gif, num_interpolations):
    frames, shapes_frames, shape_groups_frames = generate_interpolated_frames(svg_files, num_interpolations)
    
    # 保存每一帧为PNG文件
    for i, (points, shapes, shape_groups) in enumerate(frames):
        save_png(output_path, i, shapes, shape_groups)

    # 将PNG文件转换为GIF
    images = []
    for i in range(len(frames)):
        png_filename = os.path.join(output_path, f"frame_{i}.png")
        img = Image.open(png_filename)
        images.append(img)
    print(len(frames))
    images[0].save(output_gif, save_all=True, append_images=images[1:], duration=10, loop=0)

# 主函数
def main(args):
    svg_folder = args.input_path
    global output_path
    output_path = args.output_path
    k_frames = args.k_frames
    os.makedirs(output_path, exist_ok=True)
    svg_files = [os.path.join(svg_folder, f"frame_{i}.svg") for i in range(k_frames)]
    num_interpolations = args.num_interpolations  # 插入帧数
    output_gif = os.path.join(output_path, "interpolated_animation.gif")

    generate_gif(svg_files, output_gif, num_interpolations)
    print(f"GIF saved as {output_gif}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate interpolated GIF from SVG frames.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the folder containing SVG files.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the folder where output PNGs and GIF will be saved.')
    parser.add_argument('--k_frames', type=int, required=True, help='Number of SVG frames to process.')
    parser.add_argument('--num_interpolations', type=int, default=5, help='Number of interpolated frames between each pair of SVG frames.')

    args = parser.parse_args()
    main(args)

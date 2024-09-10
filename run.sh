#!/bin/bash
# conda activate design

export HF_ENDPOINT=https://hf-mirror.com

# Define an array of tasks, each with its own parameters
declare -A task1=(
    ["input_svg"]="./input_images/flag.svg"
    ["caption"]="anime style, vector 2D art, a red flag waves smoothly in the wind."
    ["k_frames"]=12
    ["model_name"]="damo-vilab/text-to-video-ms-1.7b"
    ["cuda_device"]=0
    ["train_iter"]=8000
)

declare -A task2=(
    ["input_svg"]="./input_images/new_dolphin.svg"
    ["caption"]="anime style, vector 2D art, a cartoon blue dolphin swims and flexes its body smoothly."
    ["k_frames"]=12
    ["model_name"]="damo-vilab/text-to-video-ms-1.7b"
    ["cuda_device"]=1
    ["train_iter"]=8000
)

declare -A task3=(
    ["input_svg"]="./input_images/tree.svg"
    ["caption"]="anime style, vector 2D art, a green tree is swaying left and right in the wind."
    ["k_frames"]=12
    ["model_name"]="damo-vilab/text-to-video-ms-1.7b"
    ["cuda_device"]=2
    ["train_iter"]=8000
)

declare -A task4=(
    ["input_svg"]="./input_images/new_dance.svg"
    ["caption"]="anime style, vector 2D art, a woman in a yellow dress is dancing on her feet and hands."
    ["k_frames"]=12
    ["model_name"]="damo-vilab/text-to-video-ms-1.7b"
    ["cuda_device"]=3
    ["train_iter"]=8000
)

declare -A task5=(
    ["input_svg"]="./input_images/new_wavehands.svg"
    ["caption"]="anime style, vector 2D art, a man is waving his hands smoothly."
    ["k_frames"]=12
    ["model_name"]="damo-vilab/text-to-video-ms-1.7b"
    ["cuda_device"]=4
    ["train_iter"]=8000
)

declare -A task6=(
    ["input_svg"]="./input_images/new_dog.svg"
    ["caption"]="anime style, vector 2D art, a cartoon yellow dog is walking."
    ["k_frames"]=12
    ["model_name"]="damo-vilab/text-to-video-ms-1.7b"
    ["cuda_device"]=5
    ["train_iter"]=8000
)

declare -A task7=(
    ["input_svg"]="./input_images/new_butterfly.svg"
    ["caption"]="anime style, vector 2D art, yellow spotted butterfly flies and flaps its wings."
    ["k_frames"]=12
    ["model_name"]="damo-vilab/text-to-video-ms-1.7b"
    ["cuda_device"]=6
    ["train_iter"]=8000
)

declare -A task8=(
    ["input_svg"]="./input_images/new_clock.svg"
    ["caption"]="the blue round clock with white board and two blue pointers rotates clockwise smoothly."
    ["k_frames"]=12
    ["model_name"]="damo-vilab/text-to-video-ms-1.7b"
    ["cuda_device"]=7
    ["train_iter"]=20000
)

# Array of all tasks
tasks=(task1 task2 task3 task4 task5 task6 task7 task8)

# Base output paths
path_inr_output_base="./output_results/test/"
path_svg_output_base="./output_results/test/"

# Iterate over each task
for task_name in "${tasks[@]}"; do
    # Access the task associative array
    declare -n task="$task_name"
    
    input_svg="${task[input_svg]}"
    caption="${task[caption]}"
    k_frames="${task[k_frames]}"
    model_name="${task[model_name]}"
    cuda_device="${task[cuda_device]}"
    train_iter="${task[train_iter]}"

    filename=$(basename -- "$input_svg")
    filename="${filename%.*}"
    path_image="./input_images/${filename}.png"
    path_inr_output="${path_inr_output_base}${filename}_inr"
    path_svg_output="${path_svg_output_base}${filename}_svg_optical_flow"
    path_gif_output="${path_svg_output}/interpolation/"

    echo "Processing task: $task_name"
    echo "Input SVG: $input_svg"
    echo "Caption: $caption"
    echo "K_frames: $k_frames"
    echo "Model Name: $model_name"
    echo "CUDA Device: $cuda_device"
    echo "Train Iter: $train_iter"
    echo "-----------------------------"

    CUDA_VISIBLE_DEVICES="$cuda_device" python preprocess.py --input_path="$input_svg" --output_path="$path_image"
    echo "Preprocessed $input_svg to $path_image"

    CUDA_VISIBLE_DEVICES="$cuda_device" python train.py --input_path="$path_image" --caption="$caption" --model_name="$model_name" --output_path="$path_inr_output" --k_frames="$k_frames" --train_iter="$train_iter"
    echo "Trained model for $path_image with caption '$caption'"

    CUDA_VISIBLE_DEVICES="$cuda_device" python optimize_svg_optical_flow.py --input_path="$input_svg" --output_path="$path_svg_output" --target_path="$path_inr_output" --k_frames="$k_frames" --train_iter="$train_iter"
    echo "Optimized SVG for $input_svg to $path_svg_output"

    CUDA_VISIBLE_DEVICES="$cuda_device" python interpolation.py --input_path="$path_svg_output" --output_path="$path_gif_output" --k_frames="$k_frames"
    echo "Generated GIF for $path_svg_output to $path_gif_output"

    echo "Completed task: $task_name"
    echo "============================="
done

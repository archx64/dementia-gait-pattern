#!/bin/bash

demo_path="/home/aicenter/Dev/mmpose/demo"
download_link="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e"

echo "demo folder path: ${demo_path}"

echo "${demo_path}/body3d_pose_lifter_demo.py"

python -W ignore::FutureWarning "/${demo_path}/body3d_pose_lifter_demo.py"  \
    "${demo_path}/rtmdet_m_640-8xb32_coco-person.py \
    https://

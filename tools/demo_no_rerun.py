# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import glob
import itertools
import numpy as np
import torch
import torchvision
import sys
import uuid
import time
from datetime import datetime
from collections import defaultdict
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    print("✅ Open3D可用，将使用体素下采样功能")
except ImportError:
    OPEN3D_AVAILABLE = False
    print("⚠️ Open3D不可用，将跳过体素下采样")
    o3d = None

from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation

from cubifyanything.batching import Sensors
from cubifyanything.boxes import GeneralInstance3DBoxes
from cubifyanything.capture_stream import CaptureDataset
from cubifyanything.color import random_color
from cubifyanything.cubify_transformer import make_cubify_transformer
from cubifyanything.dataset import CubifyAnythingDataset
from cubifyanything.instances import Instances3D
from cubifyanything.preprocessor import Augmentor, Preprocessor

class TimingStats:
    """耗时统计类"""
    def __init__(self):
        self.timing_data = defaultdict(list)
        self.step_times = {}
        self.frame_times = []
        self.start_time = None
    
    def start_timer(self, step_name):
        """开始计时"""
        self.step_times[step_name] = time.time()
    
    def end_timer(self, step_name):
        """结束计时并记录"""
        if step_name in self.step_times:
            elapsed = time.time() - self.step_times[step_name]
            self.timing_data[step_name].append(elapsed)
            del self.step_times[step_name]
            return elapsed
        return 0
    
    def start_total_timer(self):
        """开始总计时"""
        self.start_time = time.time()
    
    def end_total_timer(self):
        """结束总计时"""
        if self.start_time is not None:
            total_time = time.time() - self.start_time
            self.timing_data['total_processing'].append(total_time)
            return total_time
        return 0
    
    def add_frame_time(self, frame_time):
        """添加单帧处理时间"""
        self.frame_times.append(frame_time)
    
    def get_stats(self, step_name):
        """获取某个步骤的统计信息"""
        times = self.timing_data[step_name]
        if not times:
            return {'count': 0, 'total': 0, 'avg': 0, 'min': 0, 'max': 0}
        return {
            'count': len(times),
            'total': sum(times),
            'avg': sum(times) / len(times),
            'min': min(times),
            'max': max(times)
        }
    
    def print_summary(self):
        """打印耗时统计汇总"""
        print("\n" + "="*80)
        print("⏱️  耗时统计汇总报告")
        print("="*80)
        
        # 计算总时间
        total_stats = self.get_stats('total_processing')
        if total_stats['count'] > 0:
            print(f"🕒 总处理时间: {total_stats['total']:.2f}秒")
        
        if self.frame_times:
            print(f"📊 处理帧数: {len(self.frame_times)}")
            print(f"⚡ 平均每帧时间: {sum(self.frame_times)/len(self.frame_times):.3f}秒")
            print(f"🚀 最快帧: {min(self.frame_times):.3f}秒")
            print(f"🐌 最慢帧: {max(self.frame_times):.3f}秒")
            print(f"⚡ 平均FPS: {len(self.frame_times)/sum(self.frame_times):.2f}")
        
        print("\n📋 各步骤耗时详情:")
        print("-" * 80)
        
        # 按总耗时排序显示各步骤
        step_totals = []
        for step_name in self.timing_data:
            if step_name != 'total_processing':
                stats = self.get_stats(step_name)
                if stats['count'] > 0:
                    step_totals.append((step_name, stats))
        
        step_totals.sort(key=lambda x: x[1]['total'], reverse=True)
        
        for step_name, stats in step_totals:
            total_time = stats['total']
            avg_time = stats['avg']
            count = stats['count']
            min_time = stats['min']
            max_time = stats['max']
            
            print(f"{step_name:30} | "
                  f"总计: {total_time:7.2f}s | "
                  f"次数: {count:4d} | "
                  f"平均: {avg_time:6.3f}s | "
                  f"最小: {min_time:6.3f}s | "
                  f"最大: {max_time:6.3f}s")
        
        print("="*80)

def move_device_like(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    try:
        return src.to(dst)
    except:
        return src.to(dst.device)

def move_to_current_device(x, t):
    if isinstance(x, (list, tuple)):
        return [move_device_like(x_, t) for x_ in x]
    return move_device_like(x, t)

def move_input_to_current_device(batched_input: Sensors, t: torch.Tensor):
    return { name: { name_: move_to_current_device(m, t) for name_, m in s.items() } for name, s in batched_input.items() }

def process_data_and_save_outputs(model, dataset, augmentor, preprocessor, score_thresh=0.0, viz_on_gt_points=False, output_dir="outputs"):
    """处理数据并保存NOCS图和点云"""
    print(f"📁 开始处理数据，输出目录: {output_dir}")
    
    # 初始化耗时统计
    timing_stats = TimingStats()
    timing_stats.start_total_timer()

    os.makedirs(output_dir, exist_ok=True)
    nocs_dir = os.path.join(output_dir, "nocs_images")
    accumulated_dir = os.path.join(output_dir, "accumulated")

    os.makedirs(nocs_dir, exist_ok=True)
    os.makedirs(accumulated_dir, exist_ok=True)

    is_depth_model = "wide/depth" in augmentor.measurement_keys
    frame_count = 0
    device = model.pixel_mean

    all_pred_boxes = []
    all_pointclouds = []
    instance_pointclouds = {}
    video_id = None
    
    for sample in dataset:
        frame_start_time = time.time()
        frame_count += 1
        video_id = sample["meta"]["video_id"]

        if frame_count == 1:
            print("📤 开始处理数据并收集结果...")
        elif frame_count % 10 == 0:
            print(f"📊 Processing frame {frame_count}...")

        # 数据加载和预处理
        timing_stats.start_timer('data_loading')
        image = np.moveaxis(sample["wide"]["image"][-1].numpy(), 0, -1)        
        timing_stats.end_timer('data_loading')

        # 数据打包和设备转移
        timing_stats.start_timer('data_packaging')
        packaged = augmentor.package(sample)
        packaged = move_input_to_current_device(packaged, device)
        packaged = preprocessor.preprocess([packaged])
        timing_stats.end_timer('data_packaging')

        # 模型推理
        timing_stats.start_timer('model_inference')
        with torch.no_grad():
            pred_instances = model(packaged)[0]
        timing_stats.end_timer('model_inference')

        # 预测结果过滤
        timing_stats.start_timer('prediction_filtering')
        pred_instances = pred_instances[pred_instances.scores >= score_thresh]
        
        if len(pred_instances) > 0:
            boxes_3d = pred_instances.get("pred_boxes_3d")
            all_pred_boxes.append({
                'centers': boxes_3d.gravity_center.cpu().numpy(),
                'sizes': boxes_3d.dims.cpu().numpy(),
                'rotations': boxes_3d.R.cpu().numpy(),
                'scores': pred_instances.scores.cpu().numpy(),
                'frame': frame_count
            })
        timing_stats.end_timer('prediction_filtering')
        
        if viz_on_gt_points and sample["sensor_info"].has("gt"):
            # 深度图反投影
            timing_stats.start_timer('depth_unprojection')
            depth_gt = sample["gt"]["depth"][-1]
            matched_image = torch.tensor(np.array(Image.fromarray(image).resize((depth_gt.shape[1], depth_gt.shape[0]))))

            RT_camera_to_world = sample["sensor_info"].gt.RT[-1]
            xyz, valid = unproject(depth_gt, sample["sensor_info"].gt.depth.K[-1], RT_camera_to_world)
            
            height, width = depth_gt.shape[-2:]
            print(f"📐 Depth图尺寸: {height}x{width}, Image图尺寸: {image.shape[0]}x{image.shape[1]}")
            v_coords, u_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
            pixel_coords = torch.stack([u_coords, v_coords], dim=-1)
            
            xyzrgb = torch.cat((xyz, matched_image / 255.0), dim=-1)[valid]
            pixel_coords_valid = pixel_coords[valid]
            timing_stats.end_timer('depth_unprojection')
            
            if len(xyzrgb) > 0:
                print(f"💡 处理当前帧点云: {len(xyzrgb):,} 个点")
                
                # 点云数据收集
                timing_stats.start_timer('pointcloud_collection')
                all_pointclouds.append({
                    'points': xyzrgb[..., :3].cpu().numpy(),
                    'colors': xyzrgb[..., 3:].cpu().numpy(),
                    'frame': frame_count
                })
                timing_stats.end_timer('pointcloud_collection')
                
                # GT实例处理
                timing_stats.start_timer('gt_instance_processing')
                if "world_instances_3d" in sample and sample["world_instances_3d"] is not None:
                    world_gt_instances = sample["world_instances_3d"]
                    print(f"Using world coordinate GT instances: {len(world_gt_instances)} instances")
                    
                elif "world" in sample and "instances" in sample["world"]:
                    world_gt_instances = sample["world"]["instances"]
                    print(f"Using world coordinate GT instances from sample['world']: {len(world_gt_instances)} instances")
                    
                else:
                    if "wide" in sample and "instances" in sample["wide"]:
                        gt_instances = sample["wide"]["instances"]
                        world_gt_instances = gt_instances.clone()
                        
                        if world_gt_instances.has("gt_boxes_3d"):
                            original_boxes = world_gt_instances.get("gt_boxes_3d")
                            
                            RT_np = RT_camera_to_world.numpy()
                            
                            centers = original_boxes.gravity_center.cpu().numpy()
                            centers_homogeneous = np.concatenate([centers, np.ones((centers.shape[0], 1))], axis=1)
                            transformed_centers = (RT_np @ centers_homogeneous.T).T[:, :3]
                            
                            transformed_R = RT_np[:3, :3] @ original_boxes.R.cpu().numpy()
                            transformed_boxes = GeneralInstance3DBoxes(
                                np.concatenate([transformed_centers, original_boxes.dims.cpu().numpy()], axis=1),
                                transformed_R
                            )
                            
                            world_gt_instances.set("gt_boxes_3d", transformed_boxes)
                            print(f"Transformed camera coordinate instances to world: {len(world_gt_instances)} instances")
                        else:
                            print("Warning: No gt_boxes_3d found in instances")
                    else:
                        print("Warning: No GT instances available for NOCS generation")
                timing_stats.end_timer('gt_instance_processing')

                # NOCS生成
                timing_stats.start_timer('nocs_generation')
                nocs_image = generate_instance_nocs_map(
                    xyzrgb[..., :3].cpu().numpy(),  # (N, 3) 格式
                    world_gt_instances,  # 使用变换到世界坐标系的instances
                    sample["sensor_info"].wide.image.K[-1].numpy(),  # 保留兼容性，但不使用
                    RT_camera_to_world.numpy(),  # 保留兼容性，但不使用
                    depth_gt.shape[-2:],  # 使用depth图尺寸
                    pixel_coords=pixel_coords_valid.cpu().numpy()  # 传入像素坐标对应关系 (N, 2)
                )
                timing_stats.end_timer('nocs_generation')
                
                # NOCS图像保存
                if nocs_image is not None:
                    timing_stats.start_timer('nocs_image_saving')
                    nocs_image_pil = Image.fromarray(nocs_image)
                    nocs_file = os.path.join(nocs_dir, f"frame_{frame_count:04d}_nocs.png")
                    nocs_image_pil.save(nocs_file)
                    print(f"💾 保存NOCS图: {nocs_file}")
                    timing_stats.end_timer('nocs_image_saving')
                
                # 实例点云收集
                timing_stats.start_timer('instance_pointcloud_collection')
                collect_instance_pointclouds(
                    xyzrgb[..., :3].cpu().numpy(),
                    xyzrgb[..., 3:].cpu().numpy(),
                    world_gt_instances,
                    frame_count,
                    instance_pointclouds
                )
                timing_stats.end_timer('instance_pointcloud_collection')

        # 记录单帧处理时间
        frame_time = time.time() - frame_start_time
        timing_stats.add_frame_time(frame_time)
        
        # 每10帧输出一次耗时统计
        if frame_count % 10 == 0:
            avg_frame_time = sum(timing_stats.frame_times[-10:]) / min(10, len(timing_stats.frame_times))
            print(f"⏱️ 最近10帧平均处理时间: {avg_frame_time:.3f}秒")

    # 累积结果保存
    print(f"📊 完成数据处理，正在保存累积结果...")
    
    # 预测框信息保存
    if all_pred_boxes:
        timing_stats.start_timer('prediction_saving')
        accumulated_predictions_file = os.path.join(accumulated_dir, "all_predictions.txt")
        with open(accumulated_predictions_file, 'w') as f:
            f.write("Frame\tNumBoxes\tCenters\tSizes\tScores\n")
            for i, box_data in enumerate(all_pred_boxes):
                f.write(f"{box_data['frame']}\t{len(box_data['centers'])}\t")
                f.write(f"{box_data['centers'].tolist()}\t{box_data['sizes'].tolist()}\t{box_data['scores'].tolist()}\n")
        print(f"💾 保存累积预测框信息: {accumulated_predictions_file}")
        
        # 合并所有预测框并保存为点云
        all_centers = np.concatenate([box['centers'] for box in all_pred_boxes], axis=0)
        all_sizes = np.concatenate([box['sizes'] for box in all_pred_boxes], axis=0)
        all_scores = np.concatenate([box['scores'] for box in all_pred_boxes], axis=0)
        
        # 根据分数生成颜色（分数越高越红，分数越低越蓝）
        colors = []
        for score in all_scores:
            red = min(255, int(score * 255))
            blue = max(0, int((1 - score) * 255))
            colors.append([red, 0, blue])
        colors = np.array(colors) / 255.0
        
        accumulated_predictions_ply = os.path.join(accumulated_dir, "all_predictions.ply")
        save_pointcloud_ply(all_centers, colors, accumulated_predictions_ply)
        print(f"💾 保存累积预测框为点云: {accumulated_predictions_ply}")
        print(f"💡 累积预测: {len(all_centers)} 个检测结果")
        timing_stats.end_timer('prediction_saving')
    
    print("🚀 跳过全场景累积点云处理（专注于物体级处理）")
    
    # 归一化实例点云保存
    if instance_pointclouds:
        timing_stats.start_timer('normalized_pointcloud_saving')
        save_normalized_instance_pointclouds(instance_pointclouds, output_dir)
        timing_stats.end_timer('normalized_pointcloud_saving')
    
    # 结束总计时
    total_time = timing_stats.end_total_timer()
    
    # 数据处理完成后的提示
    print(f"\n✅ 数据处理完成！总共处理了 {frame_count} 帧，耗时 {total_time:.2f}秒")
    print(f"📁 输出目录: {output_dir}")
    print(f"  📸 NOCS图: {nocs_dir}")
    print(f"  📊 累积结果: {accumulated_dir}")
    print(f"  🎯 归一化Instance点云: {os.path.join(output_dir, 'normalized_instances')}")
    if all_pred_boxes:
        total_predictions = sum(len(box['centers']) for box in all_pred_boxes)
        print(f"🔴 累积预测: {total_predictions} 个检测结果")
    if instance_pointclouds:
        print(f"🎯 Instance累积: {len(instance_pointclouds)} 个不同的object")
    
    # 打印详细耗时统计
    timing_stats.print_summary()

def process_data_visualization_only(dataset, output_dir="outputs_viz"):
    """仅可视化数据，保存NOCS图和点云"""
    print(f"📁 开始数据可视化，输出目录: {output_dir}")
    
    # 初始化耗时统计
    timing_stats = TimingStats()
    timing_stats.start_total_timer()

    os.makedirs(output_dir, exist_ok=True)
    nocs_dir = os.path.join(output_dir, "nocs_images")
    accumulated_dir = os.path.join(output_dir, "accumulated")

    os.makedirs(nocs_dir, exist_ok=True)
    os.makedirs(accumulated_dir, exist_ok=True)

    frame_count = 0
    all_pointclouds = []
    instance_pointclouds = {}
    
    for sample in dataset:
        frame_start_time = time.time()
        frame_count += 1

        if frame_count == 1:
            print("📤 开始处理数据并收集结果...")
        elif frame_count % 10 == 0:
            print(f"📊 Processing frame {frame_count}...")

        # 检查数据结构并正确访问数据
        if "wide" not in sample:
            print("Warning: 'wide' key not found in sample, skipping frame")
            continue
            
        if "image" not in sample["wide"]:
            print("Warning: 'image' key not found in sample['wide'], skipping frame")
            continue

        # 数据加载
        timing_stats.start_timer('data_loading')
        image = np.moveaxis(sample["wide"]["image"][-1].numpy(), 0, -1)        
        timing_stats.end_timer('data_loading')

        # 处理当前帧的点云数据并生成NOCS
        if "gt" in sample and sample["gt"] is not None and "depth" in sample["gt"]:
            # 深度图反投影
            timing_stats.start_timer('depth_unprojection')
            depth_gt = sample["gt"]["depth"][-1]
            matched_image = torch.tensor(np.array(Image.fromarray(image).resize((depth_gt.shape[1], depth_gt.shape[0]))))

            # 使用正确的RT变换将点云从相机坐标系变换到世界坐标系
            RT_camera_to_world = sample["sensor_info"].gt.RT[-1]  # 相机到世界的变换
            xyz, valid = unproject(depth_gt, sample["sensor_info"].gt.depth.K[-1], RT_camera_to_world, max_depth=10.0)
            xyzrgb = torch.cat((xyz, matched_image / 255.0), dim=-1)[valid]
            
            # 保留像素坐标对应关系
            height, width = depth_gt.shape[-2:]
            print(f"📐 [VIZ] Depth图尺寸: {height}x{width}, Image图尺寸: {image.shape[0]}x{image.shape[1]}")
            v_coords, u_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
            pixel_coords = torch.stack([u_coords, v_coords], dim=-1)  # (H, W, 2)
            
            # 应用valid mask，同时保留像素坐标和点云数据
            xyzrgb = torch.cat((xyz, matched_image / 255.0), dim=-1)[valid]
            pixel_coords_valid = pixel_coords[valid]  # 对应的像素坐标 (N, 2)
            timing_stats.end_timer('depth_unprojection')
            
            # 保存当前帧的点云
            if len(xyzrgb) > 0:
                print(f"💡 保存当前帧点云: {len(xyzrgb):,} 个点")
                
                # 点云数据收集
                timing_stats.start_timer('pointcloud_collection')
                all_pointclouds.append({
                    'points': xyzrgb[..., :3].cpu().numpy(),
                    'colors': xyzrgb[..., 3:].cpu().numpy(),
                    'frame': frame_count
                })
                timing_stats.end_timer('pointcloud_collection')
                
                # GT实例处理
                timing_stats.start_timer('gt_instance_processing')
                # 🌍 优先检查是否有世界坐标系GT instances
                if "world_instances_3d" in sample and sample["world_instances_3d"] is not None:
                    world_gt_instances = sample["world_instances_3d"]
                    print(f"🌍 [VIZ] 使用世界坐标系GT instances: {len(world_gt_instances)} 个实例")
                    
                elif "world" in sample and "instances" in sample["world"]:
                    world_gt_instances = sample["world"]["instances"]
                    print(f"🌍 [VIZ] 使用世界坐标系GT instances: {len(world_gt_instances)} 个实例")
                    
                else:
                    # 回退到相机坐标系instances，但需要变换
                    if "wide" in sample and "instances" in sample["wide"]:
                        gt_instances = sample["wide"]["instances"]
                        # 将相机坐标系中的instances变换到世界坐标系
                        world_gt_instances = gt_instances.clone()
                        
                        if world_gt_instances.has("gt_boxes_3d"):
                            original_boxes = world_gt_instances.get("gt_boxes_3d")
                            
                            # 将boxes从相机坐标系变换到世界坐标系
                            RT_np = RT_camera_to_world.numpy()
                            
                            # 变换boxes中心点
                            centers = original_boxes.gravity_center.cpu().numpy()
                            centers_homogeneous = np.concatenate([centers, np.ones((centers.shape[0], 1))], axis=1)
                            transformed_centers = (RT_np @ centers_homogeneous.T).T[:, :3]
                            
                            # 变换boxes的旋转矩阵
                            transformed_R = RT_np[:3, :3] @ original_boxes.R.cpu().numpy()
                            
                            # 创建新的boxes在世界坐标系中
                            transformed_boxes = GeneralInstance3DBoxes(
                                np.concatenate([transformed_centers, original_boxes.dims.cpu().numpy()], axis=1),
                                transformed_R
                            )
                            
                            # 更新instances中的boxes
                            world_gt_instances.set("gt_boxes_3d", transformed_boxes)
                            
                            print(f"📐 [VIZ] 已将相机坐标系instances变换到世界坐标系: {len(world_gt_instances)} 个实例")
                        else:
                            print("⚠️ [VIZ] 警告: instances中没有gt_boxes_3d字段")
                            world_gt_instances = None
                    else:
                        print("⚠️ [VIZ] 警告: 未找到任何GT instances数据")
                        world_gt_instances = None
                timing_stats.end_timer('gt_instance_processing')

                # 继续处理world_gt_instances（如果存在）
                if world_gt_instances is not None and len(world_gt_instances) > 0:
                    # NOCS生成
                    timing_stats.start_timer('nocs_generation')
                    nocs_image = generate_instance_nocs_map(
                        xyzrgb[..., :3].cpu().numpy(),  # (N, 3) 格式
                        world_gt_instances,  # 使用变换到世界坐标系的instances
                        sample["sensor_info"].wide.image.K[-1].numpy(),  # 保留兼容性，但不使用
                        RT_camera_to_world.numpy(),  # 保留兼容性，但不使用
                        depth_gt.shape[-2:],  # 使用depth图尺寸而不是image尺寸
                        pixel_coords=pixel_coords_valid.cpu().numpy()  # 传入像素坐标对应关系 (N, 2)
                    )
                    timing_stats.end_timer('nocs_generation')
                    
                    # NOCS图像保存
                    if nocs_image is not None:
                        timing_stats.start_timer('nocs_image_saving')
                        nocs_image_pil = Image.fromarray(nocs_image)
                        nocs_file = os.path.join(nocs_dir, f"frame_{frame_count:04d}_nocs.png")
                        nocs_image_pil.save(nocs_file)
                        print(f"💾 保存NOCS图: {nocs_file}")
                        timing_stats.end_timer('nocs_image_saving')

                    # 实例点云收集
                    timing_stats.start_timer('instance_pointcloud_collection')
                    collect_instance_pointclouds(
                        xyzrgb[..., :3].cpu().numpy(),
                        xyzrgb[..., 3:].cpu().numpy(),
                        world_gt_instances,
                        frame_count,
                        instance_pointclouds
                    )
                    timing_stats.end_timer('instance_pointcloud_collection')
        
        # 记录单帧处理时间
        frame_time = time.time() - frame_start_time
        timing_stats.add_frame_time(frame_time)
        
        # 每10帧输出一次耗时统计
        if frame_count % 10 == 0:
            avg_frame_time = sum(timing_stats.frame_times[-10:]) / min(10, len(timing_stats.frame_times))
            print(f"⏱️ 最近10帧平均处理时间: {avg_frame_time:.3f}秒")
    
    print("🚀 跳过全场景累积点云处理（专注于物体级处理）")

    # 归一化实例点云保存
    if instance_pointclouds:
        timing_stats.start_timer('normalized_pointcloud_saving')
        save_normalized_instance_pointclouds(instance_pointclouds, output_dir)
        timing_stats.end_timer('normalized_pointcloud_saving')

    # 结束总计时
    total_time = timing_stats.end_total_timer()

    # 数据处理完成后的提示
    print(f"\n✅ 数据处理完成！总共处理了 {frame_count} 帧，耗时 {total_time:.2f}秒")
    print(f"📁 输出目录: {output_dir}")
    print(f"  📸 NOCS图: {nocs_dir}")
    print(f"  📊 累积结果: {accumulated_dir}")
    print(f"  🎯 归一化Instance点云: {os.path.join(output_dir, 'normalized_instances')}")
    
    # 打印详细耗时统计
    timing_stats.print_summary()

def save_pointcloud_ply(points, colors, filename):
    """将点云保存为PLY文件"""
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)
    
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(points)):
            point = points[i]
            color = colors[i]
            f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")

def save_segmented_pointclouds(points_3d, colors, gt_instances, frame_num, output_dir):
    """保存分割后的点云"""

    if not gt_instances.has("gt_boxes_3d"):
        print("Warning: No gt_boxes_3d field found in instances for segmentation")
        return

    boxes_3d = gt_instances.get("gt_boxes_3d")
    corners = boxes_3d.corners.cpu().numpy()

    frame_dir = os.path.join(output_dir, f"frame_{frame_num:04d}_segments")
    os.makedirs(frame_dir, exist_ok=True)
    for i in range(len(boxes_3d)):
        box_corners = corners[i]
        in_box_mask = points_in_box(points_3d, box_corners)

        point_count = np.sum(in_box_mask)
        print(f"Instance {i}: {point_count} points in box")

        if point_count > 0:
            instance_points = points_3d[in_box_mask]
            instance_colors = colors[in_box_mask]
            filename = os.path.join(frame_dir, f"instance_{i:02d}_segmented.ply")
            save_pointcloud_ply(instance_points, instance_colors, filename)

            print(f"Saved segmented point cloud for frame {frame_num}, instance {i} to {filename}")

def collect_instance_pointclouds(points_3d, colors, gt_instances, frame_num, instance_pointclouds):
    """按instance ID收集点云用于累积"""

    if not gt_instances.has("gt_boxes_3d") or not gt_instances.has("gt_ids"):
        print("Warning: No gt_boxes_3d or gt_ids field found in instances")
        return

    boxes_3d = gt_instances.get("gt_boxes_3d")
    instance_ids = gt_instances.get("gt_ids")
    corners = boxes_3d.corners.cpu().numpy()
    centers = boxes_3d.gravity_center.cpu().numpy()
    rotations = boxes_3d.R.cpu().numpy()
    dims = boxes_3d.dims.cpu().numpy()
    for i in range(len(boxes_3d)):
        instance_id = instance_ids[i]
        box_corners = corners[i]
        box_center = centers[i]
        box_rotation = rotations[i]
        box_dims = dims[i]

        in_box_mask = points_in_box(points_3d, box_corners)

        point_count = np.sum(in_box_mask)

        if point_count > 0:
            instance_points = points_3d[in_box_mask]
            instance_colors = colors[in_box_mask]

            if instance_id not in instance_pointclouds:
                instance_pointclouds[instance_id] = {
                    'pointcloud_frames': [],
                    'reference_bbox': None
                }

            if instance_pointclouds[instance_id]['reference_bbox'] is None:
                instance_pointclouds[instance_id]['reference_bbox'] = {
                    'center': box_center,
                    'rotation': box_rotation,
                    'dims': box_dims,
                    'reference_frame': frame_num
                }
                print(f"🎯 设置instance {instance_id}的参考bbox (来自帧{frame_num})")

            instance_pointclouds[instance_id]['pointcloud_frames'].append({
                'points': instance_points,
                'colors': instance_colors,
                'frame': frame_num,
                'current_bbox': {
                    'center': box_center,
                    'rotation': box_rotation,
                    'dims': box_dims
                }
            })

            print(f"Collected {point_count} points for instance {instance_id} in frame {frame_num}")
            print(f"  Instance {instance_id} now has {len(instance_pointclouds[instance_id]['pointcloud_frames'])} frames")

def downsample_pointcloud_with_open3d(points, colors, voxel_size=[0.004]):
    """使用Open3D进行体素下采样"""
    if not OPEN3D_AVAILABLE:
        print(f"⚠️ Open3D不可用，跳过下采样，保留原始 {len(points):,} 个点")
        return points, colors
        
    try:
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 体素下采样
        original_count = len(pcd.points)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        downsampled_count = len(downsampled_pcd.points)
        
        # 转换回numpy数组
        downsampled_points = np.asarray(downsampled_pcd.points)
        downsampled_colors = np.asarray(downsampled_pcd.colors)
        
        reduction_ratio = (original_count - downsampled_count) / original_count * 100
        print(f"📦 体素下采样: {original_count:,} -> {downsampled_count:,} 点 "
              f"(减少 {reduction_ratio:.1f}%, 体素大小: {voxel_size}m)")
        
        return downsampled_points, downsampled_colors
        
    except Exception as e:
        print(f"❌ Open3D下采样失败: {e}")
        print(f"🔄 回退到原始点云: {len(points):,} 个点")
        return points, colors

def save_normalized_instance_pointclouds(instance_pointclouds, output_dir, voxel_sizes=[0.004]):
    """使用统一参考bbox归一化并保存instance点云，支持多级下采样"""
    normalized_dir = os.path.join(output_dir, "normalized_instances")
    os.makedirs(normalized_dir, exist_ok=True)
    
    print(f"📊 使用统一参考bbox归一化并保存instance点云，带体素下采样...")
    
    for instance_id, instance_data in instance_pointclouds.items():
        if not instance_data or 'pointcloud_frames' not in instance_data:
            continue
            
        pointcloud_frames = instance_data['pointcloud_frames']
        reference_bbox = instance_data['reference_bbox']
        
        if not pointcloud_frames or not reference_bbox:
            print(f"⚠️ Instance {instance_id}: 缺少点云数据或参考bbox")
            continue
            
        print(f"\n🎯 处理Instance {instance_id}:")
        print(f"  参考bbox来自帧 {reference_bbox['reference_frame']}")
        print(f"  总共 {len(pointcloud_frames)} 帧点云数据")
        
        ref_center = reference_bbox['center']
        ref_rotation = reference_bbox['rotation'] 
        ref_dims = reference_bbox['dims']
        
        all_normalized_points = []
        all_colors = []
        frame_info = []
        
        # 归一化处理
        for frame_data in pointcloud_frames:
            frame_points = frame_data['points']
            frame_colors = frame_data['colors']
            frame_num = frame_data['frame']
            
            normalized_points, normalized_colors = compute_nocs_with_bbox_orientation(
                frame_points, ref_center, ref_rotation, ref_dims, f"{instance_id}_frame_{frame_num}")
            
            all_normalized_points.append(normalized_points)
            all_colors.append(frame_colors)
            
            frame_info.append({
                'frame': frame_num,
                'points_count': len(frame_points),
                'normalized_range': {
                    'x': [normalized_points[:, 0].min(), normalized_points[:, 0].max()],
                    'y': [normalized_points[:, 1].min(), normalized_points[:, 1].max()],
                    'z': [normalized_points[:, 2].min(), normalized_points[:, 2].max()]
                }
            })
            
            print(f"  帧 {frame_num}: {len(frame_points):,} 点 -> 归一化范围 "
                  f"X[{normalized_points[:, 0].min():.3f}, {normalized_points[:, 0].max():.3f}]")
        
        if all_normalized_points:
            # 合并所有归一化点云
            final_normalized_points = np.concatenate(all_normalized_points, axis=0)
            final_colors = np.concatenate(all_colors, axis=0)
            
            print(f"📦 合并后总点数: {len(final_normalized_points):,} 个点")
            
            # 对合并后的点云进行多级下采样（如果启用）
            if voxel_sizes:
                for voxel_size in voxel_sizes:
                    # 下采样
                    downsampled_points, downsampled_colors = downsample_pointcloud_with_open3d(
                        final_normalized_points, final_colors, voxel_size)
                    
                    # 保存下采样后的点云
                    if voxel_size == voxel_sizes[0]:  # 主文件使用最细的体素
                        normalized_file = os.path.join(normalized_dir, f"instance_{instance_id}_normalized.ply")
                    else:
                        normalized_file = os.path.join(normalized_dir, f"instance_{instance_id}_normalized_voxel_{voxel_size}.ply")
                    
                    save_pointcloud_ply(downsampled_points, downsampled_colors, normalized_file)
                    print(f"💾 保存instance {instance_id}归一化点云 (体素{voxel_size}): {normalized_file}")
            else:
                # 不进行下采样，直接保存原始点云
                normalized_file = os.path.join(normalized_dir, f"instance_{instance_id}_normalized.ply")
                save_pointcloud_ply(final_normalized_points, final_colors, normalized_file)
                print(f"💾 保存instance {instance_id}归一化点云 (原始): {normalized_file}")
            '''
            # 保存统计信息
            stats_file = os.path.join(normalized_dir, f"instance_{instance_id}_normalized_stats.txt")
            with open(stats_file, 'w') as f:
                f.write(f"Instance ID: {instance_id}\n")
                f.write(f"Total frames: {len(pointcloud_frames)}\n")
                f.write(f"Original merged points: {len(final_normalized_points):,}\n")
                
                if voxel_sizes:
                    f.write(f"\nDownsampling results:\n")
                    for voxel_size in voxel_sizes:
                        ds_points, ds_colors = downsample_pointcloud_with_open3d(
                            final_normalized_points, final_colors, voxel_size)
                        f.write(f"  Voxel {voxel_size}m: {len(ds_points):,} points\n")
                else:
                    f.write(f"\nNo downsampling applied.\n")
                
                f.write(f"\nFinal normalized points range (original):\n")
                f.write(f"  X: [{final_normalized_points[:, 0].min():.6f}, {final_normalized_points[:, 0].max():.6f}]\n")
                f.write(f"  Y: [{final_normalized_points[:, 1].min():.6f}, {final_normalized_points[:, 1].max():.6f}]\n")
                f.write(f"  Z: [{final_normalized_points[:, 2].min():.6f}, {final_normalized_points[:, 2].max():.6f}]\n")
                
                f.write(f"\nReference BBox Information (from frame {reference_bbox['reference_frame']}):\n")
                f.write(f"  Center: {ref_center}\n")
                f.write(f"  Dimensions: {ref_dims}\n")
                f.write(f"  Rotation matrix:\n{ref_rotation}\n")
                
                f.write(f"\nPer-frame normalized ranges:\n")
                for info in frame_info:
                    f.write(f"  Frame {info['frame']}: {info['points_count']:,} points\n")
                    f.write(f"    X: [{info['normalized_range']['x'][0]:.6f}, {info['normalized_range']['x'][1]:.6f}]\n")
                    f.write(f"    Y: [{info['normalized_range']['y'][0]:.6f}, {info['normalized_range']['y'][1]:.6f}]\n") 
                    f.write(f"    Z: [{info['normalized_range']['z'][0]:.6f}, {info['normalized_range']['z'][1]:.6f}]\n")
            
            print(f"📊 保存统计信息: {stats_file}")
            '''

    if voxel_sizes:
        print(f"✅ 总共保存了 {len(instance_pointclouds)} 个instance的归一化点云（含多级下采样）")
        print(f"📦 体素下采样级别: {voxel_sizes}")
    else:
        print(f"✅ 总共保存了 {len(instance_pointclouds)} 个instance的归一化点云（无下采样）")
    print(f"📁 归一化点云目录: {normalized_dir}")
    print(f"🎯 所有归一化都使用各自instance的第一帧bbox作为统一参考")

def generate_instance_nocs_map(points_3d, gt_instances, camera_K, camera_RT, image_shape, pixel_coords=None):
    """为每帧中检测到的所有物体生成NOCS图"""
    if gt_instances is None or len(gt_instances) == 0:
        return None
    
    height, width = image_shape
    
    if points_3d.ndim == 3 and points_3d.shape[:2] == (height, width):
        print(f"📍 输入点云格式: {points_3d.shape} - 保持像素对应关系")
        points_3d_reshaped = points_3d.reshape(-1, 3)
        
        v_coords, u_coords = np.mgrid[0:height, 0:width]
        pixel_u = u_coords.flatten()
        pixel_v = v_coords.flatten()
        
        valid_mask = np.isfinite(points_3d_reshaped).all(axis=1) & (np.linalg.norm(points_3d_reshaped, axis=1) > 0.01)
        
    elif points_3d.ndim == 2:
        print(f"📍 输入点云格式: {points_3d.shape} - 需要像素坐标对应")
        if pixel_coords is None:
            print("❌ 错误: (N, 3) 格式的点云需要提供pixel_coords参数")
            return None
        
        points_3d_reshaped = points_3d
        pixel_u = pixel_coords[:, 0]
        pixel_v = pixel_coords[:, 1]
        
        valid_mask = (pixel_u >= 0) & (pixel_u < width) & (pixel_v >= 0) & (pixel_v < height) & \
                    np.isfinite(points_3d_reshaped).all(axis=1) & (np.linalg.norm(points_3d_reshaped, axis=1) > 0.01)
    else:
        print(f"❌ 错误: 不支持的点云格式 {points_3d.shape}")
        return None
    
    points_3d_valid = points_3d_reshaped[valid_mask]
    pixel_u_valid = pixel_u[valid_mask]
    pixel_v_valid = pixel_v[valid_mask]
    
    print(f"📊 有效点数: {len(points_3d_valid):,} / {len(points_3d_reshaped):,} ({len(points_3d_valid)/len(points_3d_reshaped)*100:.1f}%)")
    
    if len(points_3d_valid) == 0:
        print("⚠️ 没有有效的3D点")
        return None
    
    print(f"📐 创建NOCS图像尺寸: {height}x{width}")
    nocs_image = np.zeros((height, width, 3), dtype=np.float32)
    
    if gt_instances.has("gt_boxes_3d"):
        boxes_3d = gt_instances.get("gt_boxes_3d")
    else:
        print(f"❌ 错误: 在instances中未找到gt_boxes_3d字段. 可用字段: {list(gt_instances.get_fields().keys())}")
        return None
        
    corners = boxes_3d.corners.cpu().numpy()
    rotations = boxes_3d.R.cpu().numpy()
    centers = boxes_3d.gravity_center.cpu().numpy()
    dims = boxes_3d.dims.cpu().numpy()
    
    all_normalized_points = []
    all_normalized_colors = []
    for i in range(len(boxes_3d)):
        box_corners = corners[i]
        box_rotation = rotations[i]
        box_center = centers[i]
        box_dims = dims[i]
        
        in_box_mask = points_in_box(points_3d_valid, box_corners)
        
        point_count = np.sum(in_box_mask)
        print(f"🎯 Instance {i}: {point_count} 个点在边界框内")
        
        if point_count > 0:
            instance_points = points_3d_valid[in_box_mask]
            instance_u = pixel_u_valid[in_box_mask]
            instance_v = pixel_v_valid[in_box_mask]
            
            nocs_points, nocs_colors = compute_nocs_with_bbox_orientation(
                instance_points, box_center, box_rotation, box_dims, i)
            
            nocs_image[instance_v, instance_u] = nocs_colors
            
            all_normalized_points.append(instance_points)
            all_normalized_colors.append((nocs_colors * 255).astype(np.uint8))
    
    if np.any(nocs_image > 0):
        nocs_image_display = (np.clip(nocs_image, 0, 1) * 255).astype(np.uint8)
        return nocs_image_display
    else:
        return None

def compute_nocs_with_bbox_orientation(points, box_center, box_rotation, box_dims, instance_id):
    """使用bbox的完整信息计算NOCS变换"""
    print(f"Computing NOCS with bbox orientation for instance {instance_id}")
    print(f"  Input points: {points.shape}")
    print(f"  Box center: {box_center}")
    print(f"  Box rotation matrix:\n{box_rotation}")
    print(f"  Box dimensions: {box_dims}")
    
    centered_points = points - box_center
    print(f"  Points centered to box center, range: [{np.min(centered_points, axis=0)}, {np.max(centered_points, axis=0)}]")
    
    local_points = centered_points @ box_rotation
    print(f"  Points in local coordinate system, range: [{np.min(local_points, axis=0)}, {np.max(local_points, axis=0)}]")
    
    det_R = np.linalg.det(box_rotation)
    print(f"  Rotation matrix determinant: {det_R:.6f} (should be close to ±1)")
    
    orthogonal_check = np.max(np.abs(box_rotation @ box_rotation.T - np.eye(3)))
    print(f"  Orthogonality check (max error): {orthogonal_check:.6f} (should be close to 0)")
    
    expected_range = box_dims / 2.0
    actual_range = np.max(np.abs(local_points), axis=0)
    range_ratio = actual_range / expected_range
    print(f"  Expected bbox half-size: {expected_range}")
    print(f"  Actual point cloud range: {actual_range}")
    print(f"  Range ratio (should be ≤ 1.0): {range_ratio}")
    
    max_dim = np.max(box_dims)
    print(f"  Max dimension (longest edge): {max_dim}")
    print(f"  Box dimensions: {box_dims}")
    
    nocs_points = local_points / max_dim
    
    normalized_half_dims = (box_dims / max_dim) / 2.0
    print(f"  Normalized half dimensions: {normalized_half_dims}")
    
    nocs_points = np.clip(nocs_points, -0.5, 0.5)
    print(f"  NOCS points range: [{np.min(nocs_points, axis=0)}, {np.max(nocs_points, axis=0)}]")
    
    original_nocs = local_points / max_dim
    out_of_bounds = np.any((np.abs(original_nocs) > normalized_half_dims), axis=1)
    out_of_bounds_count = np.sum(out_of_bounds)
    if out_of_bounds_count > 0:
        print(f"  ⚠️  Warning: {out_of_bounds_count}/{len(points)} ({out_of_bounds_count/len(points)*100:.1f}%) points outside bbox bounds")
    
    nocs_colors = np.clip(nocs_points + 0.5, 0, 1)
    
    print(f"  Normalized dimensions (relative to max): {box_dims / max_dim}")
    print(f"  Aspect ratio preserved: longest edge = 1.0, others scaled proportionally")
    print(f"  Bbox center moved to coordinate system origin, max range: [-0.5, 0.5]")
    
    return nocs_points, nocs_colors

def points_in_box(points, corners):
    """检查点是否在边界框内"""
    min_bound = np.min(corners, axis=0)
    max_bound = np.max(corners, axis=0)
    in_box = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    
    return in_box

def get_camera_coords(depth):
    height, width = depth.shape
    device = depth.device

    camera_coords = torch.stack(
        torch.meshgrid(
            torch.arange(0, width, device=device),
            torch.arange(0, height, device=device), indexing="xy"),
        dim=-1)

    return camera_coords

def unproject(depth, K, RT, max_depth=10.0):
    camera_coords = get_camera_coords(depth) * depth[..., None]

    intrinsics_4x4 = torch.eye(4, device=depth.device)
    intrinsics_4x4[:3, :3] = K

    valid = depth > 0
    if max_depth is not None:
        valid &= (depth < max_depth)

    depth = depth[..., None]
    uvd = torch.cat((camera_coords, depth, torch.ones_like(depth)), dim=-1)

    camera_xyz =  torch.linalg.inv(intrinsics_4x4) @ uvd.view(-1, 4).T
    world_xyz = RT @ camera_xyz

    return world_xyz.T[..., :-1].reshape(uvd.shape[0], uvd.shape[1], 3), valid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_path", help="Path to the directory containing the .tar files, the full path to a single tar file (recommended), or a path to a txt file containing HTTP links. Using the value \"stream\" will attempt to stream from your device using the NeRFCapture app")
    parser.add_argument("--model-path", help="Path to the model to load")
    parser.add_argument("--no-depth", default=False, action="store_true", help="Skip loading depth.")
    parser.add_argument("--score-thresh", default=0.25, help="Threshold for detections")
    parser.add_argument("--every-nth-frame", default=None, type=int, help="Load every `n` frames")
    parser.add_argument("--viz-only", default=False, action="store_true", help="Skip loading a model and only visualize data.")
    parser.add_argument("--viz-on-gt-points", default=False, action="store_true", help="Backproject the GT depth to form a point cloud in order to visualize the predictions")
    parser.add_argument("--device", default="cpu", help="Which device to push the model to (cpu, mps, cuda)")
    parser.add_argument("--video-ids", nargs="+", help="Subset of videos to execute on. By default, all. Ignored if a tar file is explicitly given or in stream mode.")
    parser.add_argument("--output-dir", default=None, help="Output directory path (default: auto-generated)")
    parser.add_argument("--voxel-sizes", nargs="+", type=float, default=[0.004], help="体素下采样尺寸列表 (默认: 0.004)")
    parser.add_argument("--disable-downsampling", default=False, action="store_true", help="禁用体素下采样")

    args = parser.parse_args()
    print("Command Line Args:", args)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.viz_only:
            args.output_dir = f"outputs_viz_{timestamp}"
        else:
            args.output_dir = f"outputs_model_{timestamp}"

    dataset_path = args.dataset_path
    use_cache = False
    
    if dataset_path == "stream":
        dataset = CaptureDataset()
    else:
        dataset_files = []

        if os.path.isfile(dataset_path):
            if dataset_path.endswith(".txt"):
                with open(dataset_path, "r") as dataset_file:
                    dataset_files = [l.strip() for l in dataset_file.readlines()]
                use_cache = True
            else:
                args.video_ids = None
                dataset_files = [dataset_path]
        else:
            dataset_files = glob.glob(os.path.join(dataset_path, "ca1m-*.tar"))
            if len(dataset_files) == 0:
                raise ValueError(f"Failed to find any .tar files matching ca1m- prefix at {dataset_path}")

        if args.video_ids is not None:
            dataset_files = [df for df in dataset_files if Path(df).with_suffix("").name.split("-")[-1] in args.video_ids]

        if len(dataset_files) == 0:
            raise ValueError("No data was found")
            
        dataset = CubifyAnythingDataset(
            [Path(df).as_uri() if not df.startswith("https://") else df for df in dataset_files],
            yield_world_instances=args.viz_only,
            load_arkit_depth=not args.no_depth,
            use_cache=use_cache)

    if args.viz_only:
        if args.every_nth_frame is not None:
            dataset = itertools.islice(dataset, 0, None, args.every_nth_frame)
        
        process_data_visualization_only(dataset, output_dir=args.output_dir)
        sys.exit(0)

    assert args.model_path is not None
    print("Loading model checkpoint...")
    checkpoint = torch.load(args.model_path, map_location=args.device or "cpu")["model"]
    print("Model checkpoint loaded successfully")

    backbone_embedding_dimension = checkpoint["backbone.0.patch_embed.proj.weight"].shape[0]
    is_depth_model = any(k.startswith("backbone.0.patch_embed_depth.") for k in checkpoint.keys())

    print("Creating model...")
    model = make_cubify_transformer(dimension=backbone_embedding_dimension, depth_model=is_depth_model).eval()
    print("Model created successfully")

    print("Loading model state dict...")
    model.load_state_dict(checkpoint)
    print("Model loaded successfully")

    if args.device is not None:
        print(f"Moving model to device: {args.device}")
        model = model.to(args.device)
        print("Model moved to device successfully")

    dataset.load_arkit_depth = is_depth_model
    if args.every_nth_frame is not None:
        dataset = itertools.islice(dataset, 0, None, args.every_nth_frame)

    augmentor = Augmentor(("wide/image", "wide/depth") if is_depth_model else ("wide/image",))
    preprocessor = Preprocessor()
        
    process_data_and_save_outputs(model, dataset, augmentor, preprocessor, score_thresh=args.score_thresh, viz_on_gt_points=args.viz_on_gt_points, output_dir=args.output_dir)
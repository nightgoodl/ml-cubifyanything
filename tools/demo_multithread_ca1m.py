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
import json
import threading
import queue
import multiprocessing
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    print("✅ Open3D可用，将使用体素下采样功能")
except ImportError:
    OPEN3D_AVAILABLE = False
    print("⚠️ Open3D不可用，将跳过体素下采样")
    o3d = None

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

class TaskQueue:
    """任务队列管理器"""
    def __init__(self, compute_workers=4, io_workers=2):
        self.compute_queue = queue.Queue()
        self.io_queue = queue.Queue()
        self.compute_workers = compute_workers
        self.io_workers = io_workers
        self.shutdown = False
        
    def add_compute_task(self, task):
        """添加计算任务"""
        self.compute_queue.put(task)
    
    def add_io_task(self, task):
        """添加I/O任务"""
        self.io_queue.put(task)
    
    def stop(self):
        """停止任务队列"""
        self.shutdown = True

class ThreadSafeWriter:
    """线程安全的文件写入器"""
    def __init__(self):
        self.lock = threading.Lock()
    
    def write_ply(self, points, colors, filename):
        """线程安全地写入PLY文件"""
        with self.lock:
            save_pointcloud_ply(points, colors, filename)
    
    def write_nocs_image(self, nocs_image, filename):
        """线程安全地写入NOCS图像"""
        with self.lock:
            nocs_image_pil = Image.fromarray(nocs_image)
            nocs_image_pil.save(filename)

def extract_bbox_info_from_instances(gt_instances):
    """从gt_instances中提取bbox信息"""
    if not gt_instances.has("gt_boxes_3d") or not gt_instances.has("gt_ids"):
        print("⚠️ 警告: instances中缺少gt_boxes_3d或gt_ids字段")
        return {}
    
    boxes_3d = gt_instances.get("gt_boxes_3d")
    instance_ids = gt_instances.get("gt_ids")
    
    print(f"📊 提取bbox信息: 发现 {len(instance_ids)} 个实例")
    if len(instance_ids) > 0:
        print(f"📋 Instance ID示例: {instance_ids[0]} (类型: {type(instance_ids[0])})")
    
    bbox_info = {}
    centers = boxes_3d.gravity_center.cpu().numpy()
    rotations = boxes_3d.R.cpu().numpy()
    dims = boxes_3d.dims.cpu().numpy()
    
    for i in range(len(boxes_3d)):
        # 保持原始instance_id格式，无论是int还是string
        instance_id = instance_ids[i]
        # 对于JSON序列化，确保使用字符串格式
        instance_key = str(instance_id)
        bbox_info[instance_key] = {
            'center': centers[i],
            'rotation': rotations[i],
            'dimensions': dims[i],
            'corners': boxes_3d.corners[i].cpu().numpy()
        }
    
    return bbox_info

def save_bbox_info(bbox_info, filename):
    """保存bbox信息到JSON文件"""
    # 转换numpy数组为列表以便JSON序列化
    serializable_bbox_info = {}
    for instance_id, info in bbox_info.items():
        serializable_bbox_info[instance_id] = {
            'center': info['center'].tolist(),
            'rotation': info['rotation'].tolist(),
            'dimensions': info['dimensions'].tolist(),
            'corners': info['corners'].tolist()
        }
    
    with open(filename, 'w') as f:
        json.dump(serializable_bbox_info, f, indent=2)

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
        instance_id = str(instance_ids[i])  # 确保为字符串格式
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

def downsample_pointcloud_with_open3d(points, colors, voxel_size=0.004):
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

def compute_normalized_pointcloud(instance_id, instance_data, voxel_size):
    """计算归一化点云（纯计算任务）"""
    if not instance_data or 'pointcloud_frames' not in instance_data:
        return None
        
    pointcloud_frames = instance_data['pointcloud_frames']
    reference_bbox = instance_data['reference_bbox']
    
    if not pointcloud_frames or not reference_bbox:
        print(f"⚠️ Instance {instance_id}: 缺少点云数据或参考bbox")
        return None
    
    print(f"🎯 [计算] 处理Instance {instance_id}: {len(pointcloud_frames)} 帧")
    
    ref_center = reference_bbox['center']
    ref_rotation = reference_bbox['rotation'] 
    ref_dims = reference_bbox['dims']
    
    all_normalized_points = []
    all_colors = []
    
    # 归一化处理
    for frame_data in pointcloud_frames:
        frame_points = frame_data['points']
        frame_colors = frame_data['colors']
        frame_num = frame_data['frame']
        
        normalized_points, normalized_colors = compute_nocs_with_bbox_orientation(
            frame_points, ref_center, ref_rotation, ref_dims, f"{instance_id}_frame_{frame_num}")
        
        all_normalized_points.append(normalized_points)
        all_colors.append(frame_colors)
    
    if all_normalized_points:
        # 合并所有归一化点云
        final_normalized_points = np.concatenate(all_normalized_points, axis=0)
        final_colors = np.concatenate(all_colors, axis=0)
        
        # 下采样（如果启用）
        if voxel_size > 0:
            downsampled_points, downsampled_colors = downsample_pointcloud_with_open3d(
                final_normalized_points, final_colors, voxel_size)
        else:
            downsampled_points, downsampled_colors = final_normalized_points, final_colors
        
        return instance_id, downsampled_points, downsampled_colors
    
    return None

def save_pointcloud_task(result, output_dir):
    """保存点云任务（纯I/O操作）"""
    if result is None:
        return
    
    instance_id, points, colors = result
    objects_dir = os.path.join(output_dir, "objects")
    os.makedirs(objects_dir, exist_ok=True)
    
    normalized_file = os.path.join(objects_dir, f"instance_{instance_id}.ply")
    save_pointcloud_ply(points, colors, normalized_file)
    print(f"💾 [I/O] 保存instance {instance_id}归一化点云: {normalized_file}")

def save_normalized_instance_pointcloud_threaded(instance_id, instance_data, output_dir, voxel_size, writer):
    """多线程保存归一化实例点云（兼容旧接口）"""
    result = compute_normalized_pointcloud(instance_id, instance_data, voxel_size)
    if result:
        save_pointcloud_task(result, output_dir)

def generate_nocs_image_threaded(frame_data, writer):
    """多线程生成NOCS图像"""
    xyzrgb, world_gt_instances, depth_shape, pixel_coords_valid, frame_count, nocs_dir = frame_data
    
    if world_gt_instances is None or len(world_gt_instances) == 0:
        return
    
    # 生成NOCS图
    nocs_image = generate_instance_nocs_map(
        xyzrgb[..., :3],
        world_gt_instances,
        None, None,  # 不使用相机参数
        depth_shape,
        pixel_coords=pixel_coords_valid
    )
    
    # 保存NOCS图像
    if nocs_image is not None:
        nocs_file = os.path.join(nocs_dir, f"frame_{frame_count:04d}_nocs.png")
        writer.write_nocs_image(nocs_image, nocs_file)
        print(f"💾 [线程] 保存NOCS图: {nocs_file}")

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
    
    if len(points_3d_valid) == 0:
        print("⚠️ 没有有效的3D点")
        return None
    
    nocs_image = np.zeros((height, width, 3), dtype=np.float32)
    
    if gt_instances.has("gt_boxes_3d"):
        boxes_3d = gt_instances.get("gt_boxes_3d")
    else:
        print(f"❌ 错误: 在instances中未找到gt_boxes_3d字段")
        return None
        
    corners = boxes_3d.corners.cpu().numpy()
    rotations = boxes_3d.R.cpu().numpy()
    centers = boxes_3d.gravity_center.cpu().numpy()
    dims = boxes_3d.dims.cpu().numpy()
    
    for i in range(len(boxes_3d)):
        box_corners = corners[i]
        box_rotation = rotations[i]
        box_center = centers[i]
        box_dims = dims[i]
        
        in_box_mask = points_in_box(points_3d_valid, box_corners)
        
        point_count = np.sum(in_box_mask)
        
        if point_count > 0:
            instance_points = points_3d_valid[in_box_mask]
            instance_u = pixel_u_valid[in_box_mask]
            instance_v = pixel_v_valid[in_box_mask]
            
            nocs_points, nocs_colors = compute_nocs_with_bbox_orientation(
                instance_points, box_center, box_rotation, box_dims, i)
            
            nocs_image[instance_v, instance_u] = nocs_colors
    
    if np.any(nocs_image > 0):
        nocs_image_display = (np.clip(nocs_image, 0, 1) * 255).astype(np.uint8)
        return nocs_image_display
    else:
        return None

def compute_nocs_with_bbox_orientation(points, box_center, box_rotation, box_dims, instance_id):
    """使用bbox的完整信息计算NOCS变换"""
    centered_points = points - box_center
    local_points = centered_points @ box_rotation
    max_dim = np.max(box_dims)
    nocs_points = local_points / max_dim
    nocs_points = np.clip(nocs_points, -0.5, 0.5)
    nocs_colors = np.clip(nocs_points + 0.5, 0, 1)
    
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

def process_single_scene(scene_path, output_base_dir, voxel_size=0.004, compute_workers=4, io_workers=2):
    """处理单个场景的数据，使用分离的计算和I/O线程池"""
    scene_name = Path(scene_path).stem
    print(f"\n🎬 开始处理场景: {scene_name}")
    
    # 创建场景输出目录
    scene_output_dir = os.path.join(output_base_dir, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)
    
    nocs_dir = os.path.join(scene_output_dir, "nocs_images")
    objects_dir = os.path.join(scene_output_dir, "objects")
    os.makedirs(nocs_dir, exist_ok=True)
    os.makedirs(objects_dir, exist_ok=True)
    
    # 初始化耗时统计
    timing_stats = TimingStats()
    timing_stats.start_total_timer()
    
    # 创建数据集
    dataset = CubifyAnythingDataset(
        [Path(scene_path).as_uri()],
        yield_world_instances=True,
        load_arkit_depth=True,
        use_cache=False
    )
    
    frame_count = 0
    instance_pointclouds = {}
    scene_bbox_info = {}
    writer = ThreadSafeWriter()
    
    # 收集NOCS生成任务
    nocs_tasks = []
    
    for sample in dataset:
        frame_start_time = time.time()
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"📊 处理场景 {scene_name} 第 {frame_count} 帧...")
        
        # 检查数据结构
        if "wide" not in sample or "image" not in sample["wide"]:
            continue
        
        # 数据加载
        timing_stats.start_timer('data_loading')
        image = np.moveaxis(sample["wide"]["image"][-1].numpy(), 0, -1)        
        timing_stats.end_timer('data_loading')
        
        # 处理深度数据
        if "gt" in sample and sample["gt"] is not None and "depth" in sample["gt"]:
            timing_stats.start_timer('depth_unprojection')
            depth_gt = sample["gt"]["depth"][-1]
            matched_image = torch.tensor(np.array(Image.fromarray(image).resize((depth_gt.shape[1], depth_gt.shape[0]))))
            
            RT_camera_to_world = sample["sensor_info"].gt.RT[-1]
            xyz, valid = unproject(depth_gt, sample["sensor_info"].gt.depth.K[-1], RT_camera_to_world, max_depth=10.0)
            xyzrgb = torch.cat((xyz, matched_image / 255.0), dim=-1)[valid]
            
            height, width = depth_gt.shape[-2:]
            v_coords, u_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
            pixel_coords = torch.stack([u_coords, v_coords], dim=-1)
            pixel_coords_valid = pixel_coords[valid]
            timing_stats.end_timer('depth_unprojection')
            
            if len(xyzrgb) > 0:
                # GT实例处理
                timing_stats.start_timer('gt_instance_processing')
                world_gt_instances = None
                
                if "world_instances_3d" in sample and sample["world_instances_3d"] is not None:
                    world_gt_instances = sample["world_instances_3d"]
                elif "world" in sample and "instances" in sample["world"]:
                    world_gt_instances = sample["world"]["instances"]
                elif "wide" in sample and "instances" in sample["wide"]:
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
                
                timing_stats.end_timer('gt_instance_processing')
                
                if world_gt_instances is not None and len(world_gt_instances) > 0:
                    # 收集bbox信息（仅第一帧）
                    if frame_count == 1:
                        scene_bbox_info = extract_bbox_info_from_instances(world_gt_instances)
                    
                    # 添加NOCS生成任务
                    nocs_tasks.append((
                        xyzrgb.cpu().numpy(),
                        world_gt_instances,
                        depth_gt.shape[-2:],
                        pixel_coords_valid.cpu().numpy(),
                        frame_count,
                        nocs_dir
                    ))
                    
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
    
    # 分离式多线程处理：计算与I/O分离
    print(f"🔄 开始分离式处理: {len(nocs_tasks)} 个NOCS任务, {len(instance_pointclouds)} 个点云任务")
    
    # 1. 多线程生成NOCS图像（主要是I/O）
    timing_stats.start_timer('nocs_generation')
    with ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="NOCS-IO") as nocs_executor:
        nocs_futures = [nocs_executor.submit(generate_nocs_image_threaded, task, writer) for task in nocs_tasks]
        for future in as_completed(nocs_futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ NOCS生成失败: {e}")
    timing_stats.end_timer('nocs_generation')
    
    # 2. 分离式处理归一化实例点云
    if instance_pointclouds:
        print(f"🧮 开始计算 {len(instance_pointclouds)} 个归一化点云...")
        
        # 计算阶段：使用计算线程池
        timing_stats.start_timer('pointcloud_computation')
        computed_results = []
        with ThreadPoolExecutor(max_workers=compute_workers, thread_name_prefix="Compute") as compute_executor:
            compute_futures = []
            for instance_id, instance_data in instance_pointclouds.items():
                future = compute_executor.submit(
                    compute_normalized_pointcloud,
                    instance_id, instance_data, voxel_size
                )
                compute_futures.append(future)
            
            for future in as_completed(compute_futures):
                try:
                    result = future.result()
                    if result:
                        computed_results.append(result)
                except Exception as e:
                    print(f"❌ 点云计算失败: {e}")
        timing_stats.end_timer('pointcloud_computation')
        
        # I/O阶段：使用I/O线程池
        print(f"💾 开始保存 {len(computed_results)} 个点云文件...")
        timing_stats.start_timer('pointcloud_io')
        with ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="IO") as io_executor:
            io_futures = []
            for result in computed_results:
                future = io_executor.submit(save_pointcloud_task, result, scene_output_dir)
                io_futures.append(future)
            
            for future in as_completed(io_futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ 点云保存失败: {e}")
        timing_stats.end_timer('pointcloud_io')
    
    # 保存场景bbox信息
    if scene_bbox_info:
        bbox_file = os.path.join(scene_output_dir, "scene_bbox_info.json")
        save_bbox_info(scene_bbox_info, bbox_file)
        print(f"📊 保存场景bbox信息: {bbox_file}")
    
    # 结束总计时
    total_time = timing_stats.end_total_timer()
    
    print(f"\n✅ 场景 {scene_name} 处理完成！")
    print(f"📊 总共处理了 {frame_count} 帧，耗时 {total_time:.2f}秒")
    print(f"📁 输出目录: {scene_output_dir}")
    if instance_pointclouds:
        print(f"🎯 保存了 {len(instance_pointclouds)} 个归一化实例点云")
    
    # 打印详细的耗时统计
    timing_stats.print_summary()
    
    return scene_name, total_time, frame_count, len(instance_pointclouds)

def get_scene_files(data_dir, split, worker_id=None, total_workers=None):
    """获取指定split的场景文件列表，支持分布式处理"""
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        raise ValueError(f"Split目录不存在: {split_dir}")
    
    scene_files = glob.glob(os.path.join(split_dir, "ca1m-*.tar"))
    if len(scene_files) == 0:
        raise ValueError(f"在 {split_dir} 中未找到任何ca1m-*.tar文件")
    
    scene_files = sorted(scene_files)
    
    # 分布式处理：将文件分配给不同的worker
    if worker_id is not None and total_workers is not None:
        if worker_id >= total_workers or worker_id < 0:
            raise ValueError(f"worker_id {worker_id} 必须在 0 到 {total_workers-1} 之间")
        
        # 计算当前worker需要处理的文件
        files_per_worker = len(scene_files) // total_workers
        remaining_files = len(scene_files) % total_workers
        
        start_idx = worker_id * files_per_worker + min(worker_id, remaining_files)
        if worker_id < remaining_files:
            end_idx = start_idx + files_per_worker + 1
        else:
            end_idx = start_idx + files_per_worker
        
        worker_scene_files = scene_files[start_idx:end_idx]
        
        print(f"📁 Worker {worker_id}/{total_workers}: 在 {split} 集中分配到 {len(worker_scene_files)} 个场景文件")
        print(f"📊 处理范围: {start_idx} - {end_idx-1} (总共 {len(scene_files)} 个文件)")
        
        return worker_scene_files
    else:
        print(f"📁 在 {split} 集中发现 {len(scene_files)} 个场景文件")
        return scene_files

def main():
    parser = argparse.ArgumentParser(description="CA-1M数据集分布式多线程处理工具")
    parser.add_argument("--data-dir", default="/baai-cwm-vepfs/cwm/chongjie.ye/data/CA-1M/data", 
                       help="CA-1M数据集根目录路径")
    parser.add_argument("--output-dir", default="/baai-cwm-vepfs/cwm/chongjie.ye/data/CA-1M_output",
                       help="输出目录路径")
    parser.add_argument("--split", choices=["train", "val"], required=True,
                       help="选择处理的数据集划分")
    parser.add_argument("--voxel-size", type=float, default=0.004,
                       help="体素下采样尺寸")
    parser.add_argument("--disable-downsampling", action="store_true",
                       help="禁用体素下采样")
    
    # 分布式处理参数
    parser.add_argument("--worker-id", type=int, default=None,
                       help="当前worker的ID (0开始)")
    parser.add_argument("--total-workers", type=int, default=None,
                       help="总worker数量")
    
    # 线程配置参数
    parser.add_argument("--compute-workers", type=int, default=4,
                       help="计算线程数")
    parser.add_argument("--io-workers", type=int, default=2,
                       help="I/O线程数")
    parser.add_argument("--scene-workers", type=int, default=2,
                       help="场景处理线程数")
    parser.add_argument("--max-scenes", type=int, default=None,
                       help="最大处理场景数（用于测试）")
    
    args = parser.parse_args()
    
    # 验证分布式参数
    if (args.worker_id is None) != (args.total_workers is None):
        parser.error("--worker-id 和 --total-workers 必须同时提供或同时省略")
    
    if args.worker_id is not None and args.total_workers is not None:
        if args.worker_id >= args.total_workers or args.worker_id < 0:
            parser.error(f"worker_id {args.worker_id} 必须在 0 到 {args.total_workers-1} 之间")
    
    print("="*80)
    print("🚀 CA-1M数据集分布式多线程处理工具")
    print("="*80)
    print(f"📁 数据目录: {args.data_dir}")
    print(f"📤 输出目录: {args.output_dir}")
    print(f"📊 处理划分: {args.split}")
    
    # 分布式信息
    if args.worker_id is not None:
        print(f"🤖 分布式模式: Worker {args.worker_id}/{args.total_workers}")
    else:
        print("🏠 单机模式: 处理全部数据")
    
    # 线程配置
    print(f"🧮 计算线程数: {args.compute_workers}")
    print(f"💾 I/O线程数: {args.io_workers}")
    print(f"🎬 场景处理线程数: {args.scene_workers}")
    
    if args.disable_downsampling:
        print("📦 体素下采样: 禁用")
    else:
        print(f"📦 体素下采样尺寸: {args.voxel_size}m")
    print("="*80)
    
    # 创建输出目录
    split_output_dir = os.path.join(args.output_dir, args.split)
    if args.worker_id is not None:
        # 为每个worker创建单独的输出目录
        split_output_dir = os.path.join(split_output_dir, f"worker_{args.worker_id}")
    os.makedirs(split_output_dir, exist_ok=True)
    
    # 获取场景文件列表（支持分布式）
    try:
        scene_files = get_scene_files(args.data_dir, args.split, args.worker_id, args.total_workers)
    except ValueError as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)
    
    # 限制场景数量（用于测试）
    if args.max_scenes is not None:
        scene_files = scene_files[:args.max_scenes]
        print(f"🔬 测试模式: 仅处理前 {len(scene_files)} 个场景")
    
    # 设置下采样参数
    voxel_size = 0 if args.disable_downsampling else args.voxel_size
    
    # 多线程处理场景
    total_start_time = time.time()
    successful_scenes = []
    failed_scenes = []
    
    print(f"\n🎬 开始多线程处理 {len(scene_files)} 个场景...")
    
    with ThreadPoolExecutor(max_workers=args.scene_workers, thread_name_prefix="Scene") as executor:
        # 提交所有场景处理任务
        future_to_scene = {}
        for scene_file in scene_files:
            future = executor.submit(
                process_single_scene,
                scene_file,
                split_output_dir,
                voxel_size,
                args.compute_workers,
                args.io_workers
            )
            future_to_scene[future] = scene_file
        
        # 收集结果
        for future in as_completed(future_to_scene):
            scene_file = future_to_scene[future]
            try:
                result = future.result()
                successful_scenes.append(result)
                print(f"✅ 场景 {result[0]} 处理成功")
            except Exception as e:
                failed_scenes.append((Path(scene_file).stem, str(e)))
                print(f"❌ 场景 {Path(scene_file).stem} 处理失败: {e}")
    
    # 总结报告
    total_time = time.time() - total_start_time
    print("\n" + "="*80)
    if args.worker_id is not None:
        print(f"📊 Worker {args.worker_id} 处理完成统计报告")
    else:
        print("📊 处理完成统计报告")
    print("="*80)
    print(f"🕒 总处理时间: {total_time:.2f}秒")
    print(f"✅ 成功处理场景: {len(successful_scenes)} 个")
    print(f"❌ 失败场景: {len(failed_scenes)} 个")
    
    if successful_scenes:
        total_frames = sum(result[2] for result in successful_scenes)
        total_instances = sum(result[3] for result in successful_scenes)
        print(f"📊 总处理帧数: {total_frames}")
        print(f"🎯 总实例数: {total_instances}")
        print(f"⚡ 平均每场景处理时间: {sum(result[1] for result in successful_scenes)/len(successful_scenes):.2f}秒")
        
        if len(scene_files) > 0:
            throughput = len(successful_scenes) / total_time * 3600  # 场景/小时
            print(f"🚀 处理吞吐量: {throughput:.1f} 场景/小时")
    
    if failed_scenes:
        print("\n❌ 失败场景列表:")
        for scene_name, error in failed_scenes:
            print(f"  - {scene_name}: {error}")
    
    print(f"\n📁 输出目录: {split_output_dir}")
    
    # 分布式处理提示
    if args.worker_id is not None:
        print(f"\n💡 分布式处理提示:")
        print(f"   - 当前Worker: {args.worker_id}/{args.total_workers}")
        print(f"   - 其他worker请使用不同的worker-id (0-{args.total_workers-1})")
        print(f"   - 所有worker完成后，可合并 worker_* 目录下的结果")
    
    print("="*80)

if __name__ == "__main__":
    main()

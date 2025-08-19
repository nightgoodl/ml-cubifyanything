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
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation

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

# 导入原始代码中的工具函数
from demo_no_rerun import (
    TimingStats, move_device_like, move_to_current_device, 
    move_input_to_current_device, save_pointcloud_ply,
    downsample_pointcloud_with_open3d, compute_nocs_with_bbox_orientation,
    points_in_box, get_camera_coords, unproject
)

class MultiThreadProcessor:
    """多线程处理器类"""
    def __init__(self, max_workers=4, max_nocs_workers=2, max_pointcloud_workers=2):
        self.max_workers = max_workers
        self.max_nocs_workers = max_nocs_workers
        self.max_pointcloud_workers = max_pointcloud_workers
        self.timing_stats = TimingStats()
        self.lock = threading.Lock()
        
    def process_multiple_scenes(self, data_root, output_root, model_path=None, 
                              score_thresh=0.25, viz_only=False, 
                              every_nth_frame=None, voxel_sizes=[0.004], 
                              splits=None):
        """处理多个场景的主函数"""
        print(f"🚀 开始多线程处理场景")
        print(f"📁 数据根目录: {data_root}")
        print(f"📁 输出根目录: {output_root}")
        print(f"🧵 场景处理线程数: {self.max_workers}")
        print(f"🧵 NOCS生成线程数: {self.max_nocs_workers}")
        print(f"🧵 点云保存线程数: {self.max_pointcloud_workers}")
        
        # 扫描所有场景文件
        scene_files = self._discover_scene_files(data_root, splits)
        print(f"📊 发现 {len(scene_files)} 个场景文件")
        
        # 加载模型（如果不是仅可视化模式）
        model, augmentor, preprocessor = None, None, None
        if not viz_only:
            model, augmentor, preprocessor = self._load_model(model_path)
        
        self.timing_stats.start_total_timer()
        
        # 使用线程池处理场景
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_scene = {
                executor.submit(
                    self._process_single_scene, 
                    scene_file, output_root, model, augmentor, preprocessor,
                    score_thresh, viz_only, every_nth_frame, voxel_sizes
                ): scene_file for scene_file in scene_files
            }
            
            completed_count = 0
            for future in as_completed(future_to_scene):
                scene_file = future_to_scene[future]
                try:
                    result = future.result()
                    completed_count += 1
                    print(f"✅ 完成场景 {completed_count}/{len(scene_files)}: {Path(scene_file).name}")
                    if result:
                        print(f"   处理了 {result['frames']} 帧，{result['instances']} 个实例")
                except Exception as exc:
                    print(f"❌ 场景处理失败 {Path(scene_file).name}: {exc}")
        
        total_time = self.timing_stats.end_total_timer()
        print(f"\n🎉 所有场景处理完成！总耗时: {total_time:.2f}秒")
        print(f"📁 输出目录: {output_root}")
        self.timing_stats.print_summary()
        
    def _discover_scene_files(self, data_root, splits=None):
        """发现所有场景文件"""
        scene_files = []
        
        # 默认处理所有split
        if splits is None:
            splits = ['train', 'val']
        elif isinstance(splits, str):
            splits = [splits]
        
        print(f"🎯 处理数据集划分: {splits}")
        
        # 检查指定的split目录
        for split in splits:
            split_path = os.path.join(data_root, split)
            if os.path.exists(split_path):
                tar_files = glob.glob(os.path.join(split_path, "ca1m-*.tar"))
                scene_files.extend(tar_files)
                print(f"📂 {split}: 发现 {len(tar_files)} 个场景文件")
            else:
                print(f"⚠️ {split} 目录不存在: {split_path}")
        
        # 如果没有找到文件且只有一个split，尝试在根目录查找
        if not scene_files and len(splits) == 1:
            tar_files = glob.glob(os.path.join(data_root, "ca1m-*.tar"))
            if tar_files:
                scene_files.extend(tar_files)
                print(f"📂 根目录: 发现 {len(tar_files)} 个场景文件")
            
        return sorted(scene_files)
    
    def _load_model(self, model_path):
        """加载模型"""
        print("📥 加载模型...")
        checkpoint = torch.load(model_path, map_location="cpu")["model"]
        
        backbone_embedding_dimension = checkpoint["backbone.0.patch_embed.proj.weight"].shape[0]
        is_depth_model = any(k.startswith("backbone.0.patch_embed_depth.") for k in checkpoint.keys())
        
        model = make_cubify_transformer(dimension=backbone_embedding_dimension, depth_model=is_depth_model).eval()
        model.load_state_dict(checkpoint)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        
        augmentor = Augmentor(("wide/image", "wide/depth") if is_depth_model else ("wide/image",))
        preprocessor = Preprocessor()
        
        print("✅ 模型加载完成")
        return model, augmentor, preprocessor
    
    def _process_single_scene(self, scene_file, output_root, model, augmentor, preprocessor,
                            score_thresh, viz_only, every_nth_frame, voxel_sizes):
        """处理单个场景"""
        scene_path = Path(scene_file)
        scene_name = scene_path.stem  # 去掉.tar后缀
        
        # 确定输出目录结构
        split = scene_path.parent.name if scene_path.parent.name in ['train', 'val'] else 'unknown'
        scene_output_dir = os.path.join(output_root, split, scene_name)
        
        print(f"🎬 开始处理场景: {scene_name} (来自 {split})")
        print(f"📁 场景输出目录: {scene_output_dir}")
        
        try:
            # 创建输出目录结构
            os.makedirs(scene_output_dir, exist_ok=True)
            nocs_dir = os.path.join(scene_output_dir, "nocs_images")
            objects_dir = os.path.join(scene_output_dir, "objects")
            os.makedirs(nocs_dir, exist_ok=True)
            os.makedirs(objects_dir, exist_ok=True)
            
            # 创建数据集
            dataset = CubifyAnythingDataset(
                [scene_path.as_uri()],
                yield_world_instances=True,  # 总是获取世界坐标系instances
                load_arkit_depth=True,
                use_cache=False
            )
            
            if every_nth_frame is not None:
                dataset = itertools.islice(dataset, 0, None, every_nth_frame)
            
            # 处理场景数据
            if viz_only:
                result = self._process_scene_visualization_only(
                    dataset, scene_output_dir, nocs_dir, objects_dir, voxel_sizes)
            else:
                result = self._process_scene_with_model(
                    dataset, model, augmentor, preprocessor, 
                    scene_output_dir, nocs_dir, objects_dir, 
                    score_thresh, voxel_sizes)
            
            print(f"✅ 场景处理完成: {scene_name}")
            return result
            
        except Exception as e:
            print(f"❌ 场景处理失败 {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_scene_with_model(self, dataset, model, augmentor, preprocessor,
                                scene_output_dir, nocs_dir, objects_dir, 
                                score_thresh, voxel_sizes):
        """使用模型处理场景"""
        frame_count = 0
        instance_pointclouds = {}
        all_pred_boxes = []
        scene_bbox_info = None
        device = model.pixel_mean
        
        # NOCS和点云处理队列
        nocs_queue = Queue()
        pointcloud_queue = Queue()
        
        # 启动NOCS处理线程
        nocs_threads = []
        for i in range(self.max_nocs_workers):
            thread = threading.Thread(
                target=self._nocs_worker, 
                args=(nocs_queue, nocs_dir),
                daemon=True
            )
            thread.start()
            nocs_threads.append(thread)
        
        for sample in dataset:
            frame_count += 1
            
            # 提取场景bbox信息（只在第一帧或world帧）
            if scene_bbox_info is None and "world" in sample:
                scene_bbox_info = self._extract_scene_bbox_info(sample)
            
            if frame_count % 10 == 0:
                print(f"📊 处理帧 {frame_count}...")
            
            # 数据预处理
            image = np.moveaxis(sample["wide"]["image"][-1].numpy(), 0, -1)
            packaged = augmentor.package(sample)
            packaged = move_input_to_current_device(packaged, device)
            packaged = preprocessor.preprocess([packaged])
            
            # 模型推理
            with torch.no_grad():
                pred_instances = model(packaged)[0]
            pred_instances = pred_instances[pred_instances.scores >= score_thresh]
            
            # 收集预测结果
            if len(pred_instances) > 0:
                boxes_3d = pred_instances.get("pred_boxes_3d")
                all_pred_boxes.append({
                    'centers': boxes_3d.gravity_center.cpu().numpy(),
                    'sizes': boxes_3d.dims.cpu().numpy(),
                    'rotations': boxes_3d.R.cpu().numpy(),
                    'scores': pred_instances.scores.cpu().numpy(),
                    'frame': frame_count
                })
            
            # 处理深度数据和NOCS生成
            if sample["sensor_info"].has("gt") and "depth" in sample["gt"]:
                self._process_frame_depth_data(
                    sample, image, frame_count, nocs_queue, 
                    instance_pointclouds, scene_bbox_info)
        
        # 停止NOCS处理线程
        for _ in range(self.max_nocs_workers):
            nocs_queue.put(None)
        for thread in nocs_threads:
            thread.join()
        
        # 保存预测结果
        if all_pred_boxes:
            self._save_prediction_results(all_pred_boxes, scene_output_dir)
        
        # 保存场景bbox信息
        if scene_bbox_info:
            self._save_scene_bbox_info(scene_bbox_info, scene_output_dir)
        
        # 多线程保存归一化点云
        if instance_pointclouds:
            self._save_normalized_pointclouds_multithread(
                instance_pointclouds, objects_dir, voxel_sizes)
        
        return {
            'frames': frame_count,
            'instances': len(instance_pointclouds),
            'predictions': len(all_pred_boxes)
        }
    
    def _process_scene_visualization_only(self, dataset, scene_output_dir, 
                                        nocs_dir, objects_dir, voxel_sizes):
        """仅可视化处理场景"""
        frame_count = 0
        instance_pointclouds = {}
        scene_bbox_info = None
        
        # NOCS处理队列
        nocs_queue = Queue()
        
        # 启动NOCS处理线程
        nocs_threads = []
        for i in range(self.max_nocs_workers):
            thread = threading.Thread(
                target=self._nocs_worker,
                args=(nocs_queue, nocs_dir),
                daemon=True
            )
            thread.start()
            nocs_threads.append(thread)
        
        for sample in dataset:
            frame_count += 1
            
            # 提取场景bbox信息
            if scene_bbox_info is None and "world" in sample:
                scene_bbox_info = self._extract_scene_bbox_info(sample)
            
            if frame_count % 10 == 0:
                print(f"📊 [VIZ] 处理帧 {frame_count}...")
            
            # 处理图像数据
            if "wide" not in sample or "image" not in sample["wide"]:
                continue
                
            image = np.moveaxis(sample["wide"]["image"][-1].numpy(), 0, -1)
            
            # 处理深度数据和NOCS生成
            if "gt" in sample and sample["gt"] is not None and "depth" in sample["gt"]:
                self._process_frame_depth_data(
                    sample, image, frame_count, nocs_queue, 
                    instance_pointclouds, scene_bbox_info)
        
        # 停止NOCS处理线程
        for _ in range(self.max_nocs_workers):
            nocs_queue.put(None)
        for thread in nocs_threads:
            thread.join()
        
        # 保存场景bbox信息
        if scene_bbox_info:
            self._save_scene_bbox_info(scene_bbox_info, scene_output_dir)
        
        # 多线程保存归一化点云
        if instance_pointclouds:
            self._save_normalized_pointclouds_multithread(
                instance_pointclouds, objects_dir, voxel_sizes)
        
        return {
            'frames': frame_count,
            'instances': len(instance_pointclouds)
        }
    
    def _process_frame_depth_data(self, sample, image, frame_count, nocs_queue, 
                                instance_pointclouds, scene_bbox_info):
        """处理单帧的深度数据"""
        depth_gt = sample["gt"]["depth"][-1]
        matched_image = torch.tensor(np.array(Image.fromarray(image).resize((depth_gt.shape[1], depth_gt.shape[0]))))
        
        RT_camera_to_world = sample["sensor_info"].gt.RT[-1]
        xyz, valid = unproject(depth_gt, sample["sensor_info"].gt.depth.K[-1], RT_camera_to_world, max_depth=10.0)
        
        height, width = depth_gt.shape[-2:]
        v_coords, u_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        pixel_coords = torch.stack([u_coords, v_coords], dim=-1)
        
        xyzrgb = torch.cat((xyz, matched_image / 255.0), dim=-1)[valid]
        pixel_coords_valid = pixel_coords[valid]
        
        if len(xyzrgb) == 0:
            return
        
        # 获取GT实例
        world_gt_instances = self._get_world_gt_instances(sample, RT_camera_to_world)
        
        if world_gt_instances is not None and len(world_gt_instances) > 0:
            # 添加NOCS生成任务到队列
            nocs_task = {
                'points_3d': xyzrgb[..., :3].cpu().numpy(),
                'gt_instances': world_gt_instances,
                'image_shape': depth_gt.shape[-2:],
                'pixel_coords': pixel_coords_valid.cpu().numpy(),
                'frame_count': frame_count,
                'camera_K': sample["sensor_info"].wide.image.K[-1].numpy(),
                'camera_RT': RT_camera_to_world.numpy()
            }
            nocs_queue.put(nocs_task)
            
            # 收集实例点云
            self._collect_instance_pointclouds_threadsafe(
                xyzrgb[..., :3].cpu().numpy(),
                xyzrgb[..., 3:].cpu().numpy(),
                world_gt_instances,
                frame_count,
                instance_pointclouds
            )
    
    def _get_world_gt_instances(self, sample, RT_camera_to_world):
        """获取世界坐标系的GT实例"""
        # 优先使用世界坐标系instances
        if "world_instances_3d" in sample and sample["world_instances_3d"] is not None:
            return sample["world_instances_3d"]
        elif "world" in sample and "instances" in sample["world"]:
            return sample["world"]["instances"]
        else:
            # 从相机坐标系转换
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
                    return world_gt_instances
        
        return None
    
    def _nocs_worker(self, nocs_queue, nocs_dir):
        """NOCS生成工作线程"""
        while True:
            task = nocs_queue.get()
            if task is None:
                break
            
            try:
                nocs_image = self._generate_instance_nocs_map(
                    task['points_3d'], task['gt_instances'], 
                    task['camera_K'], task['camera_RT'],
                    task['image_shape'], task['pixel_coords']
                )
                
                if nocs_image is not None:
                    nocs_image_pil = Image.fromarray(nocs_image)
                    nocs_file = os.path.join(nocs_dir, f"frame_{task['frame_count']:04d}_nocs.png")
                    nocs_image_pil.save(nocs_file)
                    
            except Exception as e:
                print(f"❌ NOCS生成失败 (帧 {task['frame_count']}): {e}")
            finally:
                nocs_queue.task_done()
    
    def _generate_instance_nocs_map(self, points_3d, gt_instances, camera_K, camera_RT, image_shape, pixel_coords=None):
        """生成实例NOCS图（多线程安全版本）"""
        if gt_instances is None or len(gt_instances) == 0:
            return None
        
        height, width = image_shape
        
        if points_3d.ndim == 2:
            if pixel_coords is None:
                return None
            
            points_3d_reshaped = points_3d
            pixel_u = pixel_coords[:, 0]
            pixel_v = pixel_coords[:, 1]
            
            valid_mask = (pixel_u >= 0) & (pixel_u < width) & (pixel_v >= 0) & (pixel_v < height) & \
                        np.isfinite(points_3d_reshaped).all(axis=1) & (np.linalg.norm(points_3d_reshaped, axis=1) > 0.01)
        else:
            return None
        
        points_3d_valid = points_3d_reshaped[valid_mask]
        pixel_u_valid = pixel_u[valid_mask]
        pixel_v_valid = pixel_v[valid_mask]
        
        if len(points_3d_valid) == 0:
            return None
        
        nocs_image = np.zeros((height, width, 3), dtype=np.float32)
        
        if gt_instances.has("gt_boxes_3d"):
            boxes_3d = gt_instances.get("gt_boxes_3d")
        else:
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
    
    def _collect_instance_pointclouds_threadsafe(self, points_3d, colors, gt_instances, 
                                               frame_num, instance_pointclouds):
        """线程安全的实例点云收集"""
        if not gt_instances.has("gt_boxes_3d") or not gt_instances.has("gt_ids"):
            return
        
        boxes_3d = gt_instances.get("gt_boxes_3d")
        instance_ids = gt_instances.get("gt_ids")
        corners = boxes_3d.corners.cpu().numpy()
        centers = boxes_3d.gravity_center.cpu().numpy()
        rotations = boxes_3d.R.cpu().numpy()
        dims = boxes_3d.dims.cpu().numpy()
        
        with self.lock:
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
    
    def _save_normalized_pointclouds_multithread(self, instance_pointclouds, objects_dir, voxel_sizes):
        """多线程保存归一化点云"""
        print(f"🧵 使用 {self.max_pointcloud_workers} 个线程保存归一化点云...")
        
        with ThreadPoolExecutor(max_workers=self.max_pointcloud_workers) as executor:
            future_to_instance = {}
            
            for instance_id, instance_data in instance_pointclouds.items():
                if not instance_data or 'pointcloud_frames' not in instance_data:
                    continue
                
                future = executor.submit(
                    self._save_single_instance_pointcloud,
                    instance_id, instance_data, objects_dir, voxel_sizes
                )
                future_to_instance[future] = instance_id
            
            completed_count = 0
            for future in as_completed(future_to_instance):
                instance_id = future_to_instance[future]
                try:
                    result = future.result()
                    completed_count += 1
                    if result:
                        print(f"💾 保存实例 {completed_count}/{len(future_to_instance)}: {instance_id}")
                except Exception as exc:
                    print(f"❌ 实例点云保存失败 {instance_id}: {exc}")
    
    def _save_single_instance_pointcloud(self, instance_id, instance_data, objects_dir, voxel_sizes):
        """保存单个实例的归一化点云"""
        pointcloud_frames = instance_data['pointcloud_frames']
        reference_bbox = instance_data['reference_bbox']
        
        if not pointcloud_frames or not reference_bbox:
            return False
        
        # 创建实例目录
        instance_dir = os.path.join(objects_dir, str(instance_id))
        os.makedirs(instance_dir, exist_ok=True)
        
        ref_center = reference_bbox['center']
        ref_rotation = reference_bbox['rotation']
        ref_dims = reference_bbox['dims']
        
        all_normalized_points = []
        all_colors = []
        
        # 归一化处理
        for frame_data in pointcloud_frames:
            frame_points = frame_data['points']
            frame_colors = frame_data['colors']
            
            normalized_points, normalized_colors = compute_nocs_with_bbox_orientation(
                frame_points, ref_center, ref_rotation, ref_dims, f"{instance_id}")
            
            all_normalized_points.append(normalized_points)
            all_colors.append(frame_colors)
        
        if all_normalized_points:
            # 合并所有归一化点云
            final_normalized_points = np.concatenate(all_normalized_points, axis=0)
            final_colors = np.concatenate(all_colors, axis=0)
            
            # 多级下采样保存
            if voxel_sizes:
                for voxel_size in voxel_sizes:
                    downsampled_points, downsampled_colors = downsample_pointcloud_with_open3d(
                        final_normalized_points, final_colors, voxel_size)
                    
                    if voxel_size == voxel_sizes[0]:
                        normalized_file = os.path.join(instance_dir, f"normalized.ply")
                    else:
                        normalized_file = os.path.join(instance_dir, f"normalized_voxel_{voxel_size}.ply")
                    
                    save_pointcloud_ply(downsampled_points, downsampled_colors, normalized_file)
            else:
                normalized_file = os.path.join(instance_dir, f"normalized.ply")
                save_pointcloud_ply(final_normalized_points, final_colors, normalized_file)
            
            # 保存实例信息
            info_file = os.path.join(instance_dir, "info.json")
            instance_info = {
                'instance_id': int(instance_id) if isinstance(instance_id, (int, np.integer)) else str(instance_id),
                'total_frames': len(pointcloud_frames),
                'total_points': len(final_normalized_points),
                'reference_bbox': {
                    'center': ref_center.tolist() if hasattr(ref_center, 'tolist') else ref_center,
                    'rotation': ref_rotation.tolist() if hasattr(ref_rotation, 'tolist') else ref_rotation,
                    'dims': ref_dims.tolist() if hasattr(ref_dims, 'tolist') else ref_dims,
                    'reference_frame': reference_bbox['reference_frame']
                },
                'normalized_range': {
                    'x': [float(final_normalized_points[:, 0].min()), float(final_normalized_points[:, 0].max())],
                    'y': [float(final_normalized_points[:, 1].min()), float(final_normalized_points[:, 1].max())],
                    'z': [float(final_normalized_points[:, 2].min()), float(final_normalized_points[:, 2].max())]
                }
            }
            
            with open(info_file, 'w') as f:
                json.dump(instance_info, f, indent=2)
            
            return True
        
        return False
    
    def _extract_scene_bbox_info(self, sample):
        """提取场景bbox信息"""
        if "world" in sample and "instances" in sample["world"]:
            world_instances = sample["world"]["instances"]
            
            if world_instances.has("gt_boxes_3d") and world_instances.has("gt_ids"):
                boxes_3d = world_instances.get("gt_boxes_3d")
                instance_ids = world_instances.get("gt_ids")
                
                scene_info = {
                    'total_instances': len(boxes_3d),
                    'instances': []
                }
                
                centers = boxes_3d.gravity_center.cpu().numpy()
                dims = boxes_3d.dims.cpu().numpy()
                rotations = boxes_3d.R.cpu().numpy()
                
                for i in range(len(boxes_3d)):
                    instance_info = {
                        'id': int(instance_ids[i]) if isinstance(instance_ids[i], (int, np.integer)) else str(instance_ids[i]),
                        'center': centers[i].tolist(),
                        'dimensions': dims[i].tolist(),
                        'rotation': rotations[i].tolist()
                    }
                    
                    if world_instances.has("gt_names"):
                        names = world_instances.get("gt_names")
                        if i < len(names):
                            instance_info['category'] = names[i]
                    
                    scene_info['instances'].append(instance_info)
                
                return scene_info
        
        return None
    
    def _save_scene_bbox_info(self, scene_bbox_info, scene_output_dir):
        """保存场景bbox信息"""
        if scene_bbox_info:
            bbox_file = os.path.join(scene_output_dir, "scene_bbox_info.json")
            with open(bbox_file, 'w') as f:
                json.dump(scene_bbox_info, f, indent=2)
            print(f"💾 保存场景bbox信息: {bbox_file}")
    
    def _save_prediction_results(self, all_pred_boxes, scene_output_dir):
        """保存预测结果"""
        if all_pred_boxes:
            predictions_file = os.path.join(scene_output_dir, "predictions.json")
            
            # 转换为可序列化的格式
            serializable_predictions = []
            for box_data in all_pred_boxes:
                pred_data = {
                    'frame': int(box_data['frame']),
                    'num_boxes': len(box_data['centers']),
                    'centers': box_data['centers'].tolist(),
                    'sizes': box_data['sizes'].tolist(),
                    'rotations': box_data['rotations'].tolist(),
                    'scores': box_data['scores'].tolist()
                }
                serializable_predictions.append(pred_data)
            
            with open(predictions_file, 'w') as f:
                json.dump({
                    'total_frames': len(all_pred_boxes),
                    'total_predictions': sum(len(box['centers']) for box in all_pred_boxes),
                    'predictions': serializable_predictions
                }, f, indent=2)
            
            print(f"💾 保存预测结果: {predictions_file}")


def main():
    parser = argparse.ArgumentParser(description="多线程处理CA-1M数据集")
    
    parser.add_argument("--data-root", default="/baai-cwm-vepfs/cwm/chongjie.ye/data/CA-1M/data", 
                       help="数据集根目录路径")
    parser.add_argument("--output-root", default="/baai-cwm-vepfs/cwm/chongjie.ye/data/CA-1M_output", 
                       help="输出根目录路径")
    parser.add_argument("--model-path", help="模型路径")
    parser.add_argument("--viz-only", default=False, action="store_true", 
                       help="仅可视化模式，跳过模型加载")
    parser.add_argument("--score-thresh", default=0.25, type=float, 
                       help="检测阈值")
    parser.add_argument("--every-nth-frame", default=None, type=int, 
                       help="每N帧处理一次")
    parser.add_argument("--max-workers", default=4, type=int, 
                       help="场景处理最大线程数")
    parser.add_argument("--max-nocs-workers", default=2, type=int, 
                       help="NOCS生成最大线程数")
    parser.add_argument("--max-pointcloud-workers", default=2, type=int, 
                       help="点云保存最大线程数")
    parser.add_argument("--voxel-sizes", nargs="+", type=float, default=[0.004], 
                       help="体素下采样尺寸列表")
    parser.add_argument("--disable-downsampling", default=False, action="store_true", 
                       help="禁用体素下采样")
    parser.add_argument("--splits", nargs="+", default=None,
                       help="指定要处理的数据集划分 (train, val 或 both)")
    
    args = parser.parse_args()
    
    print("🚀 多线程CA-1M数据集处理器")
    print("=" * 60)
    print(f"📁 数据根目录: {args.data_root}")
    print(f"📁 输出根目录: {args.output_root}")
    print(f"🧵 最大工作线程: 场景={args.max_workers}, NOCS={args.max_nocs_workers}, 点云={args.max_pointcloud_workers}")
    
    if not args.viz_only and args.model_path is None:
        print("❌ 错误: 非可视化模式需要提供模型路径")
        sys.exit(1)
    
    if args.disable_downsampling:
        args.voxel_sizes = []
    
    # 创建处理器
    processor = MultiThreadProcessor(
        max_workers=args.max_workers,
        max_nocs_workers=args.max_nocs_workers,
        max_pointcloud_workers=args.max_pointcloud_workers
    )
    
    # 开始处理
    processor.process_multiple_scenes(
        data_root=args.data_root,
        output_root=args.output_root,
        model_path=args.model_path,
        score_thresh=args.score_thresh,
        viz_only=args.viz_only,
        every_nth_frame=args.every_nth_frame,
        voxel_sizes=args.voxel_sizes,
        splits=args.splits
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集验证工具
用于检查多视角物体重建、NOCS估计、6D姿态估计等任务的数据集完整性

作者: AI Assistant
日期: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import cv2
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DatasetValidator:
    """数据集验证器"""
    
    def __init__(self, scene_data: Dict):
        """
        初始化验证器
        
        参数:
            scene_data: 场景数据字典
        """
        self.scene_data = scene_data
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        
    def validate_all(self) -> Dict[str, Any]:
        """执行所有验证检查"""
        print("🔍 开始数据集验证...")
        print("=" * 60)
        
        # 基础数据验证
        self.validate_scene_colors()
        self.validate_scene_points()
        self.validate_scene_depths()
        self.validate_camera_parameters()
        self.validate_scene_scales()
        self.validate_scene_poses()
        self.validate_masks()
        self.validate_nocs_maps()
        self.validate_scene_instance_ids()
        self.validate_model_list()
        self.validate_bbox()
        
        # 一致性验证
        self.validate_consistency()
        
        # 生成报告
        self.generate_validation_report()
        
        return self.validation_results
    
    def validate_scene_colors(self) -> bool:
        """验证RGB图像数据"""
        print("🎨 验证RGB图像数据...")
        
        try:
            colors = self.scene_data['scene_colors']
            
            # 检查维度数量
            if len(colors.shape) != 4:
                self.errors.append(f"scene_colors维度错误: 期望4维, 实际{len(colors.shape)}维")
                return False
            
            # 检查最后一维是否为3（RGB通道）
            if colors.shape[-1] != 3:
                self.errors.append(f"scene_colors最后一维应为3(RGB), 实际{colors.shape[-1]}")
                return False
            
            # 检查数据类型
            if colors.dtype != np.uint8:
                self.errors.append(f"scene_colors数据类型错误: 期望uint8, 实际{colors.dtype}")
                return False
            
            # 检查值域
            if colors.min() < 0 or colors.max() > 255:
                self.errors.append(f"scene_colors值域错误: 应在[0,255], 实际[{colors.min()},{colors.max()}]")
                return False
            
            # 检查是否全为0（示例数据）
            if np.all(colors == 0):
                self.warnings.append("scene_colors全为0，可能是示例数据")
            
            self.validation_results['scene_colors'] = {
                'valid': True,
                'shape': colors.shape,
                'dtype': str(colors.dtype),
                'value_range': (int(colors.min()), int(colors.max())),
                'is_example_data': np.all(colors == 0)
            }
            
            print("✅ RGB图像数据验证通过")
            return True
            
        except KeyError:
            self.errors.append("缺少scene_colors字段")
            return False
        except Exception as e:
            self.errors.append(f"scene_colors验证失败: {e}")
            return False
    
    def validate_scene_points(self) -> bool:
        """验证3D点云数据"""
        print("🌐 验证3D点云数据...")
        
        try:
            points = self.scene_data['scene_points']
            
            # 检查维度数量
            if len(points.shape) != 4:
                self.errors.append(f"scene_points维度错误: 期望4维, 实际{len(points.shape)}维")
                return False
            
            # 检查最后一维是否为3（XYZ坐标）
            if points.shape[-1] != 3:
                self.errors.append(f"scene_points最后一维应为3(XYZ), 实际{points.shape[-1]}")
                return False
            
            # 检查数据类型
            if points.dtype != np.float32:
                self.errors.append(f"scene_points数据类型错误: 期望float32, 实际{points.dtype}")
                return False
            
            # 检查是否全为0（示例数据）
            if np.all(points == 0):
                self.warnings.append("scene_points全为0，可能是示例数据")
            
            self.validation_results['scene_points'] = {
                'valid': True,
                'shape': points.shape,
                'dtype': str(points.dtype),
                'value_range': (float(points.min()), float(points.max())),
                'is_example_data': np.all(points == 0)
            }
            
            print("✅ 3D点云数据验证通过")
            return True
            
        except KeyError:
            self.errors.append("缺少scene_points字段")
            return False
        except Exception as e:
            self.errors.append(f"scene_points验证失败: {e}")
            return False
    
    def validate_scene_depths(self) -> bool:
        """验证深度图数据"""
        print("📏 验证深度图数据...")
        
        try:
            depths = self.scene_data['scene_depths']
            
            # 检查维度数量
            if len(depths.shape) != 3:
                self.errors.append(f"scene_depths维度错误: 期望3维, 实际{len(depths.shape)}维")
                return False
            
            # 检查数据类型
            if depths.dtype != np.float32:
                self.errors.append(f"scene_depths数据类型错误: 期望float32, 实际{depths.dtype}")
                return False
            
            # 检查是否全为0（示例数据）
            if np.all(depths == 0):
                self.warnings.append("scene_depths全为0，可能是示例数据")
            
            self.validation_results['scene_depths'] = {
                'valid': True,
                'shape': depths.shape,
                'dtype': str(depths.dtype),
                'value_range': (float(depths.min()), float(depths.max())),
                'is_example_data': np.all(depths == 0)
            }
            
            print("✅ 深度图数据验证通过")
            return True
            
        except KeyError:
            self.errors.append("缺少scene_depths字段")
            return False
        except Exception as e:
            self.errors.append(f"scene_depths验证失败: {e}")
            return False
    
    def validate_camera_parameters(self) -> bool:
        """验证相机参数"""
        print("📷 验证相机参数...")
        
        try:
            # 验证外参
            extrinsics = self.scene_data['extrinsics']
            if len(extrinsics.shape) != 3 or extrinsics.shape[1:] != (4, 4):
                self.errors.append(f"extrinsics形状错误: 期望(?,4,4), 实际{extrinsics.shape}")
                return False
            
            # 验证内参
            intrinsics = self.scene_data['intrinsics']
            if len(intrinsics.shape) != 3 or intrinsics.shape[1:] != (3, 3):
                self.errors.append(f"intrinsics形状错误: 期望(?,3,3), 实际{intrinsics.shape}")
                return False
            
            # 检查外参和内参数量是否一致
            if extrinsics.shape[0] != intrinsics.shape[0]:
                self.errors.append(f"外参和内参数量不一致: 外参{extrinsics.shape[0]}, 内参{intrinsics.shape[0]}")
                return False
            
            # 检查内参矩阵的有效性
            for i, K in enumerate(intrinsics):
                if K[2, 2] != 1.0:
                    self.warnings.append(f"内参矩阵{i}的[2,2]元素应为1.0，实际为{K[2,2]}")
                if K[0, 1] != 0.0 or K[1, 0] != 0.0:
                    self.warnings.append(f"内参矩阵{i}的非对角元素应为0")
            
            self.validation_results['camera_parameters'] = {
                'valid': True,
                'extrinsics_shape': extrinsics.shape,
                'intrinsics_shape': intrinsics.shape,
                'is_example_data': np.all(extrinsics == 0) and np.all(intrinsics == 0)
            }
            
            print("✅ 相机参数验证通过")
            return True
            
        except KeyError as e:
            self.errors.append(f"缺少相机参数字段: {e}")
            return False
        except Exception as e:
            self.errors.append(f"相机参数验证失败: {e}")
            return False
    
    def validate_scene_scales(self) -> bool:
        """验证场景尺度"""
        print("📐 验证场景尺度...")
        
        try:
            scales = self.scene_data['scene_scales']
            
            # 检查形状
            if scales.shape[1] != 3:
                self.errors.append(f"scene_scales第二维应为3, 实际{scales.shape[1]}")
                return False
            
            # 检查是否全为0（示例数据）
            if np.all(scales == 0):
                self.warnings.append("scene_scales全为0，可能是示例数据")
            
            # 检查尺度一致性（每个类别的三个维度应该相等）
            for i, scale in enumerate(scales):
                if not np.allclose(scale, scale[0]):
                    self.warnings.append(f"类别{i}的尺度不一致: {scale}")
            
            self.validation_results['scene_scales'] = {
                'valid': True,
                'shape': scales.shape,
                'scales': scales.tolist(),
                'is_example_data': np.all(scales == 0)
            }
            
            print("✅ 场景尺度验证通过")
            return True
            
        except KeyError:
            self.errors.append("缺少scene_scales字段")
            return False
        except Exception as e:
            self.errors.append(f"scene_scales验证失败: {e}")
            return False
    
    def validate_scene_poses(self) -> bool:
        """验证场景姿态"""
        print("🔄 验证场景姿态...")
        
        try:
            poses = self.scene_data['scene_poses'][()]
            
            # 检查是否为字典类型
            if not isinstance(poses, dict):
                self.errors.append(f"scene_poses应为字典类型, 实际{type(poses)}")
                return False
            
            # 检查每个姿态矩阵
            for key, pose_array in poses.items():
                if not isinstance(pose_array, np.ndarray):
                    self.errors.append(f"姿态{key}应为numpy数组, 实际{type(pose_array)}")
                    continue
                    
                if len(pose_array.shape) != 3 or pose_array.shape[1:] != (4, 4):
                    self.errors.append(f"姿态{key}形状错误: 期望(?,4,4), 实际{pose_array.shape}")
                    continue
                
                # 检查齐次变换矩阵的有效性
                for i, pose in enumerate(pose_array):
                    if not np.allclose(pose[3, :], [0, 0, 0, 1]):
                        self.warnings.append(f"姿态{key}[{i}]的最后一行应为[0,0,0,1]")
                    
                    # 检查旋转矩阵的正交性
                    R = pose[:3, :3]
                    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
                        self.warnings.append(f"姿态{key}[{i}]的旋转矩阵不正交")
            
            self.validation_results['scene_poses'] = {
                'valid': True,
                'num_views': len(poses),
                'keys': list(poses.keys()),
                'is_example_data': all(np.all(pose_array == 0) for pose_array in poses.values())
            }
            
            print("✅ 场景姿态验证通过")
            return True
            
        except KeyError:
            self.errors.append("缺少scene_poses字段")
            return False
        except Exception as e:
            self.errors.append(f"scene_poses验证失败: {e}")
            return False
    
    def validate_masks(self) -> bool:
        """验证掩码数据"""
        print("🎭 验证掩码数据...")
        
        try:
            visible_masks = self.scene_data['visible_masks']
            masks = self.scene_data['masks'][()]
            
            # 验证visible_masks
            if len(visible_masks.shape) != 3:
                self.errors.append(f"visible_masks维度错误: 期望3维, 实际{len(visible_masks.shape)}维")
                return False
            
            # 验证masks字典
            if not isinstance(masks, dict):
                self.errors.append(f"masks应为字典类型, 实际{type(masks)}")
                return False
            
            # 检查每个掩码
            for key, mask_array in masks.items():
                if not isinstance(mask_array, np.ndarray):
                    self.errors.append(f"掩码{key}应为numpy数组, 实际{type(mask_array)}")
                    continue
                    
                if len(mask_array.shape) != 3:
                    self.errors.append(f"掩码{key}维度错误: 期望3维, 实际{len(mask_array.shape)}维")
                    continue
                
                # 检查掩码值
                unique_values = np.unique(mask_array)
                if not all(val in [0, 1, 2, 3, 4, 255] for val in unique_values):
                    self.warnings.append(f"掩码{key}包含意外的值: {unique_values}")
            
            self.validation_results['masks'] = {
                'valid': True,
                'visible_masks_shape': visible_masks.shape,
                'masks_keys': list(masks.keys()),
                'is_example_data': np.all(visible_masks == 0) and all(np.all(mask_array == 0) for mask_array in masks.values())
            }
            
            print("✅ 掩码数据验证通过")
            return True
            
        except KeyError as e:
            self.errors.append(f"缺少掩码字段: {e}")
            return False
        except Exception as e:
            self.errors.append(f"掩码数据验证失败: {e}")
            return False
    
    def validate_nocs_maps(self) -> bool:
        """验证NOCS地图"""
        print("🗺️ 验证NOCS地图...")
        
        try:
            nocs_maps = self.scene_data['nocs_maps']
            
            # 检查维度数量
            if len(nocs_maps.shape) != 4:
                self.errors.append(f"nocs_maps维度错误: 期望4维, 实际{len(nocs_maps.shape)}维")
                return False
            
            # 检查最后一维是否为3（RGB通道）
            if nocs_maps.shape[-1] != 3:
                self.errors.append(f"nocs_maps最后一维应为3(RGB), 实际{nocs_maps.shape[-1]}")
                return False
            
            # 检查数据类型
            if nocs_maps.dtype != np.uint8:
                self.errors.append(f"nocs_maps数据类型错误: 期望uint8, 实际{nocs_maps.dtype}")
                return False
            
            # 检查值域
            if nocs_maps.min() < 0 or nocs_maps.max() > 255:
                self.errors.append(f"nocs_maps值域错误: 应在[0,255], 实际[{nocs_maps.min()},{nocs_maps.max()}]")
                return False
            
            # 检查是否全为0（示例数据）
            if np.all(nocs_maps == 0):
                self.warnings.append("nocs_maps全为0，可能是示例数据")
            
            self.validation_results['nocs_maps'] = {
                'valid': True,
                'shape': nocs_maps.shape,
                'dtype': str(nocs_maps.dtype),
                'value_range': (int(nocs_maps.min()), int(nocs_maps.max())),
                'is_example_data': np.all(nocs_maps == 0)
            }
            
            print("✅ NOCS地图验证通过")
            return True
            
        except KeyError:
            self.errors.append("缺少nocs_maps字段")
            return False
        except Exception as e:
            self.errors.append(f"nocs_maps验证失败: {e}")
            return False
    
    def validate_scene_instance_ids(self) -> bool:
        """验证场景实例ID"""
        print("🆔 验证场景实例ID...")
        
        try:
            instance_ids = self.scene_data['scene_instance_ids'][()]
            
            # 检查是否为字典类型
            if not isinstance(instance_ids, dict):
                self.errors.append(f"scene_instance_ids应为字典类型, 实际{type(instance_ids)}")
                return False
            
            # 检查每个实例ID数组
            for key, ids in instance_ids.items():
                if not isinstance(ids, np.ndarray):
                    self.errors.append(f"实例ID{key}不是numpy数组")
                    continue
                
                if ids.dtype != np.uint8:
                    self.warnings.append(f"实例ID{key}数据类型应为uint8, 实际{ids.dtype}")
                
                # 检查ID值是否合理
                if len(ids) > 0 and (ids.min() < 1 or ids.max() > 4):
                    self.warnings.append(f"实例ID{key}值超出范围[1,4]: {ids}")
            
            self.validation_results['scene_instance_ids'] = {
                'valid': True,
                'keys': list(instance_ids.keys()),
                'is_example_data': all(len(ids) == 0 or np.all(ids == 0) for ids in instance_ids.values())
            }
            
            print("✅ 场景实例ID验证通过")
            return True
            
        except KeyError:
            self.errors.append("缺少scene_instance_ids字段")
            return False
        except Exception as e:
            self.errors.append(f"scene_instance_ids验证失败: {e}")
            return False
    
    def validate_model_list(self) -> bool:
        """验证模型列表"""
        print("📋 验证模型列表...")
        
        try:
            model_list = self.scene_data['model_list']
            
            # 检查是否为numpy数组
            if not isinstance(model_list, np.ndarray):
                self.errors.append("model_list不是numpy数组")
                return False
            
            # 检查数据类型
            if model_list.dtype != 'object':
                self.warnings.append(f"model_list数据类型应为object, 实际{model_list.dtype}")
            
            # 检查长度
            if len(model_list) != 4:
                self.warnings.append(f"model_list长度应为4, 实际{len(model_list)}")
            
            # 检查是否为空字符串
            if any(len(name) == 0 for name in model_list):
                self.warnings.append("model_list包含空字符串")
            
            self.validation_results['model_list'] = {
                'valid': True,
                'length': len(model_list),
                'models': model_list.tolist(),
                'is_example_data': all(name == '' for name in model_list)
            }
            
            print("✅ 模型列表验证通过")
            return True
            
        except KeyError:
            self.errors.append("缺少model_list字段")
            return False
        except Exception as e:
            self.errors.append(f"model_list验证失败: {e}")
            return False
    
    def validate_bbox(self) -> bool:
        """验证边界框数据"""
        print("📦 验证边界框数据...")
        
        try:
            bbox = self.scene_data['bbox'][()]
            
            # 检查是否为字典类型
            if not isinstance(bbox, dict):
                self.errors.append(f"bbox应为字典类型, 实际{type(bbox)}")
                return False
            
            # 检查每个边界框
            for key, objects in bbox.items():
                if not isinstance(objects, dict):
                    self.errors.append(f"边界框{key}不是字典")
                    continue
                
                for obj_id, obj_data in objects.items():
                    required_fields = ['x', 'y', 'z', 'w', 'h', 'l', 'R']
                    for field in required_fields:
                        if field not in obj_data:
                            self.errors.append(f"边界框{key}[{obj_id}]缺少字段{field}")
                            continue
                    
                    # 检查旋转矩阵
                    if 'R' in obj_data:
                        R = obj_data['R']
                        if R.shape != (3, 3):
                            self.warnings.append(f"边界框{key}[{obj_id}]旋转矩阵形状错误: {R.shape}")
                        elif not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
                            self.warnings.append(f"边界框{key}[{obj_id}]旋转矩阵不正交")
            
            self.validation_results['bbox'] = {
                'valid': True,
                'keys': list(bbox.keys()),
                'is_example_data': all(
                    all(obj_data['x'] == 0 for obj_data in objects.values()) 
                    for objects in bbox.values()
                )
            }
            
            print("✅ 边界框数据验证通过")
            return True
            
        except KeyError:
            self.errors.append("缺少bbox字段")
            return False
        except Exception as e:
            self.errors.append(f"bbox验证失败: {e}")
            return False
    
    def validate_consistency(self) -> bool:
        """验证数据一致性"""
        print("🔗 验证数据一致性...")
        
        try:
            # 检查图像数据之间的一致性
            if 'scene_colors' in self.scene_data and 'scene_points' in self.scene_data:
                colors = self.scene_data['scene_colors']
                points = self.scene_data['scene_points']
                
                # 检查前3维是否一致
                if colors.shape[:3] != points.shape[:3]:
                    self.errors.append(f"scene_colors和scene_points前3维不一致: {colors.shape[:3]} vs {points.shape[:3]}")
                    return False
            
            # 检查深度图与图像的一致性
            if 'scene_colors' in self.scene_data and 'scene_depths' in self.scene_data:
                colors = self.scene_data['scene_colors']
                depths = self.scene_data['scene_depths']
                
                # 检查前3维是否一致
                if colors.shape[:3] != depths.shape:
                    self.errors.append(f"scene_colors和scene_depths维度不一致: {colors.shape[:3]} vs {depths.shape}")
                    return False
            
            # 检查NOCS地图与图像的一致性
            if 'scene_colors' in self.scene_data and 'nocs_maps' in self.scene_data:
                colors = self.scene_data['scene_colors']
                nocs = self.scene_data['nocs_maps']
                
                # 检查前3维是否一致
                if colors.shape[:3] != nocs.shape[:3]:
                    self.errors.append(f"scene_colors和nocs_maps前3维不一致: {colors.shape[:3]} vs {nocs.shape[:3]}")
                    return False
            
            # 检查相机参数与图像数量的一致性
            if 'scene_colors' in self.scene_data and 'extrinsics' in self.scene_data:
                colors = self.scene_data['scene_colors']
                extrinsics = self.scene_data['extrinsics']
                
                if colors.shape[0] != extrinsics.shape[0]:
                    self.errors.append(f"图像数量与相机外参数量不一致: {colors.shape[0]} vs {extrinsics.shape[0]}")
                    return False
            
            # 检查masks和scene_instance_ids的一致性
            if 'masks' in self.scene_data and 'scene_instance_ids' in self.scene_data:
                masks = self.scene_data['masks'][()]
                instance_ids = self.scene_data['scene_instance_ids'][()]
                
                for key in masks.keys():
                    if key in instance_ids:
                        mask_array = masks[key]
                        ids = instance_ids[key]
                        
                        # 检查掩码中的实例ID是否与scene_instance_ids一致
                        unique_mask_ids = np.unique(mask_array)
                        unique_mask_ids = unique_mask_ids[unique_mask_ids != 255]  # 排除背景
                        
                        if not np.array_equal(np.sort(unique_mask_ids), np.sort(ids)):
                            self.warnings.append(f"掩码{key}的实例ID与scene_instance_ids不一致")
            
            # 检查poses和masks的一致性
            if 'scene_poses' in self.scene_data and 'masks' in self.scene_data:
                poses = self.scene_data['scene_poses'][()]
                masks = self.scene_data['masks'][()]
                
                for key in poses.keys():
                    if key in masks:
                        pose_count = poses[key].shape[0]
                        mask_count = masks[key].shape[0]
                        if pose_count != mask_count:
                            self.warnings.append(f"姿态{key}数量({pose_count})与掩码数量({mask_count})不一致")
            
            self.validation_results['consistency'] = {
                'valid': True,
                'image_data_consistent': True,
                'camera_data_consistent': True,
                'masks_instance_ids_consistent': True,
                'poses_masks_consistent': True
            }
            
            print("✅ 数据一致性验证通过")
            return True
            
        except Exception as e:
            self.errors.append(f"数据一致性验证失败: {e}")
            return False
    
    def generate_validation_report(self):
        """生成验证报告"""
        print("\n" + "=" * 60)
        print("📊 数据集验证报告")
        print("=" * 60)
        
        # 统计结果
        total_checks = len(self.validation_results)
        valid_checks = sum(1 for result in self.validation_results.values() if result.get('valid', False))
        
        print(f"总检查项: {total_checks}")
        print(f"通过检查: {valid_checks}")
        print(f"失败检查: {total_checks - valid_checks}")
        print(f"错误数量: {len(self.errors)}")
        print(f"警告数量: {len(self.warnings)}")
        
        # 显示错误
        if self.errors:
            print("\n❌ 错误列表:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        # 显示警告
        if self.warnings:
            print("\n⚠️ 警告列表:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        # 显示示例数据警告
        example_data_fields = []
        for field, result in self.validation_results.items():
            if result.get('is_example_data', False):
                example_data_fields.append(field)
        
        if example_data_fields:
            print(f"\n📝 示例数据字段: {', '.join(example_data_fields)}")
            print("   这些字段包含示例数据（全为0或空），需要替换为真实数据")
        
        print("\n" + "=" * 60)
        
        # 保存报告
        self.validation_results['summary'] = {
            'total_checks': total_checks,
            'valid_checks': valid_checks,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'example_data_fields': example_data_fields
        }
    
    def visualize_data(self, save_path: Optional[str] = None):
        """可视化数据集"""
        print("\n🎨 生成数据可视化...")
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. RGB图像可视化
        ax1 = plt.subplot(3, 4, 1)
        if 'scene_colors' in self.scene_data:
            colors = self.scene_data['scene_colors']
            # 显示第一张图像
            if not np.all(colors[0] == 0):
                ax1.imshow(colors[0])
            else:
                # 创建与原始图像相同尺寸的零图像
                zero_img = np.zeros(colors[0].shape, dtype=np.uint8)
                ax1.imshow(zero_img)
            ax1.set_title('RGB Image (Frame 1)')
            ax1.axis('off')
        
        # 2. 深度图可视化
        ax2 = plt.subplot(3, 4, 2)
        if 'scene_depths' in self.scene_data:
            depths = self.scene_data['scene_depths']
            if not np.all(depths[0] == 0):
                im = ax2.imshow(depths[0], cmap='viridis')
                plt.colorbar(im, ax=ax2)
            else:
                # 创建与原始深度图相同尺寸的零图像
                zero_depth = np.zeros(depths[0].shape, dtype=np.float32)
                ax2.imshow(zero_depth, cmap='viridis')
            ax2.set_title('Depth Map (Frame 1)')
            ax2.axis('off')
        
        # 3. NOCS地图可视化
        ax3 = plt.subplot(3, 4, 3)
        if 'nocs_maps' in self.scene_data:
            nocs = self.scene_data['nocs_maps']
            if not np.all(nocs[0] == 0):
                ax3.imshow(nocs[0])
            else:
                # 创建与原始NOCS地图相同尺寸的零图像
                zero_nocs = np.zeros(nocs[0].shape, dtype=np.uint8)
                ax3.imshow(zero_nocs)
            ax3.set_title('NOCS Map (Frame 1)')
            ax3.axis('off')
        
        # 4. 掩码可视化
        ax4 = plt.subplot(3, 4, 4)
        if 'visible_masks' in self.scene_data:
            masks = self.scene_data['visible_masks']
            if not np.all(masks[0] == 0):
                ax4.imshow(masks[0], cmap='tab10')
                # 在前景部分添加对应值的文字标记
                mask_img = masks[0]
                unique_vals = np.unique(mask_img)
                for val in unique_vals:
                    if val == 0:
                        continue  # 跳过背景
                    # 获取该值的像素坐标
                    ys, xs = np.where(mask_img == val)
                    if len(xs) == 0 or len(ys) == 0:
                        continue
                    # 取该区域的中心点
                    x_mean = int(np.mean(xs))
                    y_mean = int(np.mean(ys))
                    ax4.text(x_mean, y_mean, str(val), color='white', fontsize=12, ha='center', va='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
            else:
                # 创建与原始掩码相同尺寸的零图像
                zero_mask = np.zeros(masks[0].shape, dtype=np.uint8)
                ax4.imshow(zero_mask, cmap='tab10')
            ax4.set_title('Visible Masks (Frame 1)')
            ax4.axis('off')
        
        # 5. 相机轨迹可视化
        ax5 = plt.subplot(3, 4, 5, projection='3d')
        if 'extrinsics' in self.scene_data:
            extrinsics = self.scene_data['extrinsics']
            if not np.all(extrinsics == 0):
                positions = extrinsics[:, :3, 3]  # 提取位置
                ax5.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='red', s=50)
                ax5.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.7)
                ax5.set_xlabel('X')
                ax5.set_ylabel('Y')
                ax5.set_zlabel('Z')
                ax5.set_title('Camera Trajectory')
            else:
                ax5.text(0.5, 0.5, 0.5, 'Example Data\n(All Zeros)', ha='center', va='center')
                ax5.set_title('Camera Trajectory')
        
        # 6. 实例ID分布
        ax6 = plt.subplot(3, 4, 6)
        if 'scene_instance_ids' in self.scene_data:
            instance_ids = self.scene_data['scene_instance_ids'][()]
            all_ids = []
            for ids in instance_ids.values():
                all_ids.extend(ids.tolist())
            
            if all_ids:
                unique_ids, counts = np.unique(all_ids, return_counts=True)
                ax6.bar(unique_ids, counts)
                ax6.set_xlabel('Instance ID')
                ax6.set_ylabel('Count')
                ax6.set_title('Instance ID Distribution')
            else:
                ax6.text(0.5, 0.5, 'Example Data\n(No Instances)', ha='center', va='center')
                ax6.set_title('Instance ID Distribution')
        
        # 7. 尺度分布
        ax7 = plt.subplot(3, 4, 7)
        if 'scene_scales' in self.scene_data:
            scales = self.scene_data['scene_scales']
            if not np.all(scales == 0):
                scale_magnitudes = np.linalg.norm(scales, axis=1)
                ax7.bar(range(len(scale_magnitudes)), scale_magnitudes)
                ax7.set_xlabel('Category')
                ax7.set_ylabel('Scale Magnitude')
                ax7.set_title('Category Scale Distribution')
            else:
                ax7.text(0.5, 0.5, 'Example Data\n(All Zeros)', ha='center', va='center')
                ax7.set_title('Category Scale Distribution')
        
        # 8. 三维边界框、姿态与点云可视化
        ax8 = plt.subplot(3, 4, 8, projection='3d')
        has_bbox = 'bbox' in self.scene_data
        has_pose = 'scene_poses' in self.scene_data
        has_points = 'scene_points' in self.scene_data
        has_colors = 'scene_colors' in self.scene_data
        bbox_drawn = False

        # 先可视化点云
        if False:
            # 获取点云和颜色数据，假设shape为(1, H, W, 3)
            points_array = self.scene_data['scene_points'][0]  # (H, W, 3)
            H, W, _ = points_array.shape

            # 展平成(N, 3)
            points = points_array.reshape(-1, 3)

            # 获取颜色
            if has_colors:
                colors_array = self.scene_data['scene_colors'][0]  # (H, W, 3)
                colors = colors_array.reshape(-1, 3)
                # 归一化到[0,1]
                if colors.max() > 1.0:
                    colors = colors / 255.0
            else:
                colors = None

            # 过滤掉无效点（如z==0或nan）
            valid_mask = np.isfinite(points).all(axis=1) & (points[:,2] != 0)
            points_valid = points[valid_mask]
            if colors is not None:
                colors_valid = colors[valid_mask]
            else:
                colors_valid = None

            # 采样最多2000个点用于可视化
            num_points = points_valid.shape[0]
            if num_points > 2000:
                idx = np.random.choice(num_points, 2000, replace=False)
                points_vis = points_valid[idx]
                if colors_valid is not None and len(colors_valid) == num_points:
                    colors_vis = colors_valid[idx]
                else:
                    colors_vis = None
            else:
                points_vis = points_valid
                colors_vis = colors_valid

            # 绘制点云
            if points_vis.shape[0] > 0:
                if colors_vis is not None and len(colors_vis) == len(points_vis):
                    ax8.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], c=colors_vis, s=0.5, alpha=0.7)
                else:
                    ax8.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], c='gray', s=0.5, alpha=0.7)

            # 过滤掉无效点（如z==0或nan）
            valid_mask = np.isfinite(points).all(axis=1) & (points[:,2] != 0)
            points_valid = points[valid_mask]
            if colors is not None:
                colors_valid = colors[valid_mask]
            else:
                colors_valid = None

            # 绘制点云
            if points_valid.shape[0] > 0:
                if colors_valid is not None and len(colors_valid) == len(points_valid):
                    ax8.scatter(points_valid[:, 0], points_valid[:, 1], points_valid[:, 2], c=colors_valid, s=0.5, alpha=0.7)
                else:
                    ax8.scatter(points_valid[:, 0], points_valid[:, 1], points_valid[:, 2], c='gray', s=0.5, alpha=0.7)

        # 绘制bbox，并用有向线段（箭头）清晰表示bbox的位姿信息
        # whl是位姿变换后，bbox方向下的xyz的方向长度
        if has_bbox:
            bbox = self.scene_data['bbox'][()]
            # 检查是否有非零边界框
            if any(any(obj_data['x'] != 0 for obj_data in objects.values()) for objects in bbox.values()):
                # 只绘制第一个视图（如'0000'）的边界框
                if '0000' in bbox:
                    objects = bbox['0000']
                    for obj_id, obj_data in objects.items():
                        x, y, z = obj_data['x'], obj_data['y'], obj_data['z']
                        w, h, l = obj_data['w'], obj_data['h'], obj_data['l']
                        R = obj_data.get('R', np.eye(3, dtype=np.float32))
                        # 计算8个顶点（以中心为原点，长宽高分别为l,w,h）
                        # 参考construct_data.ipynb注释，l为x方向，w为y方向，h为z方向
                        # 顶点顺序与常见3D bbox一致
                        corners = np.array([
                            [ l/2,  w/2,  h/2],
                            [ l/2, -w/2,  h/2],
                            [-l/2, -w/2,  h/2],
                            [-l/2,  w/2,  h/2],
                            [ l/2,  w/2, -h/2],
                            [ l/2, -w/2, -h/2],
                            [-l/2, -w/2, -h/2],
                            [-l/2,  w/2, -h/2],
                        ], dtype=np.float32)
                        # 旋转+平移
                        corners = (R @ corners.T).T + np.array([x, y, z], dtype=np.float32)
                        # 边的连接顺序
                        edges = [
                            (0,1),(1,2),(2,3),(3,0), # 上面
                            (4,5),(5,6),(6,7),(7,4), # 下面
                            (0,4),(1,5),(2,6),(3,7)  # 竖线
                        ]
                        # 绘制边界框的所有边（无箭头，仅辅助显示）
                        for s, e in edges:
                            ax8.plot(
                                [corners[s,0], corners[e,0]],
                                [corners[s,1], corners[e,1]],
                                [corners[s,2], corners[e,2]],
                                color='red', linewidth=1, alpha=0.5
                            )
                        # 绘制bbox的局部坐标系（有向线段，表示位姿信息）
                        # whl分别是bbox局部坐标系下xyz轴的长度
                        center = np.array([x, y, z], dtype=np.float32)
                        axes = R  # 3x3旋转矩阵
                        # X轴（红色），长度为l
                        ax8.quiver(
                            center[0], center[1], center[2],
                            axes[0,0], axes[1,0], axes[2,0],
                            color='r', length=l, normalize=True, linewidth=2, arrow_length_ratio=0.2, alpha=0.9
                        )
                        ax8.text(
                            center[0] + axes[0,0]*l,
                            center[1] + axes[1,0]*l,
                            center[2] + axes[2,0]*l,
                            'X_bbox', color='r', fontsize=10, ha='center', va='center'
                        )
                        # Y轴（绿色），长度为w
                        ax8.quiver(
                            center[0], center[1], center[2],
                            axes[0,1], axes[1,1], axes[2,1],
                            color='g', length=w, normalize=True, linewidth=2, arrow_length_ratio=0.2, alpha=0.9
                        )
                        ax8.text(
                            center[0] + axes[0,1]*w,
                            center[1] + axes[1,1]*w,
                            center[2] + axes[2,1]*w,
                            'Y_bbox', color='g', fontsize=10, ha='center', va='center'
                        )
                        # Z轴（蓝色），长度为h
                        ax8.quiver(
                            center[0], center[1], center[2],
                            axes[0,2], axes[1,2], axes[2,2],
                            color='b', length=h, normalize=True, linewidth=2, arrow_length_ratio=0.2, alpha=0.9
                        )
                        ax8.text(
                            center[0] + axes[0,2]*h,
                            center[1] + axes[1,2]*h,
                            center[2] + axes[2,2]*h,
                            'Z_bbox', color='b', fontsize=10, ha='center', va='center'
                        )
                        # 标注ID
                        ax8.text(x, y, z, f'ID:{obj_id}', color='red', fontsize=9, ha='center', va='center')
                        bbox_drawn = True

        # 可视化姿态（以相机坐标系为例），并标注xyz轴
        if has_pose:
            poses = self.scene_data['scene_poses'][()]
            if '0000' in poses:
                pose_array = poses['0000']
                for i, pose in enumerate(pose_array):
                    # 坐标系原点
                    origin = pose[:3, 3]
                    # 坐标轴方向
                    axes = pose[:3, :3]
                    scale = 2.0
                    # 绘制X轴
                    ax8.quiver(
                        origin[0], origin[1], origin[2],
                        axes[0,0], axes[1,0], axes[2,0],
                        color='r', length=scale, normalize=True
                    )
                    # 绘制Y轴
                    ax8.quiver(
                        origin[0], origin[1], origin[2],
                        axes[0,1], axes[1,1], axes[2,1],
                        color='g', length=scale, normalize=True
                    )
                    # 绘制Z轴
                    ax8.quiver(
                        origin[0], origin[1], origin[2],
                        axes[0,2], axes[1,2], axes[2,2],
                        color='b', length=scale, normalize=True
                    )
                    # 标注X、Y、Z轴
                    ax8.text(
                        origin[0] + axes[0,0]*scale, 
                        origin[1] + axes[1,0]*scale, 
                        origin[2] + axes[2,0]*scale, 
                        'X_pose', color='r', fontsize=8, ha='center', va='center'
                    )
                    ax8.text(
                        origin[0] + axes[0,1]*scale, 
                        origin[1] + axes[1,1]*scale, 
                        origin[2] + axes[2,1]*scale, 
                        'Y_pose', color='g', fontsize=8, ha='center', va='center'
                    )
                    ax8.text(
                        origin[0] + axes[0,2]*scale, 
                        origin[1] + axes[1,2]*scale, 
                        origin[2] + axes[2,2]*scale, 
                        'Z_pose', color='b', fontsize=8, ha='center', va='center'
                    )
                bbox_drawn = True

        if not bbox_drawn:
            ax8.text(0.5, 0.5, 0.5, 'Example Data\n(All Zeros)', ha='center', va='center')
            ax8.set_title('3D Bounding Boxes & Poses')
        else:
            ax8.set_xlabel('X (m)')
            ax8.set_ylabel('Y (m)')
            ax8.set_zlabel('Z (m)')
            ax8.set_title('3D Bounding Boxes & Poses')
            ax8.grid(True)
        
        # 9. 数据统计
        ax9 = plt.subplot(3, 4, 9)
        if 'validation_results' in self.__dict__:
            summary = self.validation_results.get('summary', {})
            labels = ['Passed', 'Failed', 'Errors', 'Warnings']
            values = [
                summary.get('valid_checks', 0),
                summary.get('total_checks', 0) - summary.get('valid_checks', 0),
                summary.get('error_count', 0),
                summary.get('warning_count', 0)
            ]
            colors = ['green', 'red', 'orange', 'yellow']
            ax9.bar(labels, values, color=colors)
            ax9.set_title('Validation Statistics')
            ax9.set_ylabel('Count')
        
        # 10. 数据完整性
        ax10 = plt.subplot(3, 4, 10)
        if 'validation_results' in self.__dict__:
            fields = list(self.validation_results.keys())
            if 'summary' in fields:
                fields.remove('summary')
            
            valid_fields = [field for field in fields if self.validation_results[field].get('valid', False)]
            invalid_fields = [field for field in fields if not self.validation_results[field].get('valid', False)]
            
            ax10.pie([len(valid_fields), len(invalid_fields)], 
                    labels=['Valid', 'Invalid'], 
                    colors=['lightgreen', 'lightcoral'],
                    autopct='%1.1f%%')
            ax10.set_title('Data Integrity')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 可视化图表已保存至: {save_path}")
        
        plt.show()

    
        


def main():
    """主函数 - 示例用法"""
    # 这里使用您提供的示例数据
    scene_data = {
        'scene_colors': np.zeros((10, 480, 640, 3), dtype=np.uint8),
        'scene_points': np.zeros((10, 480, 640, 3), dtype=np.float32),
        'scene_depths': np.zeros((10, 480, 640), dtype=np.float32),
        'extrinsics': np.zeros((10, 4, 4), dtype=np.float32),
        'intrinsics': np.zeros((10, 3, 3), dtype=np.float32),
        'scene_scales': np.zeros((4, 3), dtype=np.float32),
        'scene_poses': {f'{i:04d}': np.zeros((2, 4, 4), dtype=np.float32) for i in range(10)},
        'visible_masks': np.zeros((10, 480, 640), dtype=np.uint8),
        'masks': {f'{i:04d}': np.zeros((2, 480, 640), dtype=np.uint8) for i in range(10)},
        'nocs_maps': np.zeros((10, 480, 640, 3), dtype=np.uint8),
        'scene_instance_ids': {f'{i:04d}': np.array([1, 4], dtype=np.uint8) for i in range(10)},
        'model_list': np.array(['mushroom_011', 'tamarind_018', 'conch_114', 'nipple_013'], dtype='object'),
        'bbox': {
            '0000': {
                0: {'x': 10.5, 'y': 2.0, 'z': 1.7, 'w': 2.5, 'h': 1.8, 'l': 4.0, 'R': np.eye(3, dtype=np.float32)},
                1: {'x': 8.0, 'y': 1.8, 'z': 0.5, 'w': 0.8, 'h': 1.7, 'l': 0.6, 'R': np.eye(3, dtype=np.float32)}
            }
        }
    }
    
    # 创建验证器
    validator = DatasetValidator(scene_data)
    
    # 执行验证
    results = validator.validate_all()
    
    # 生成可视化
    validator.visualize_data('/baai-cwm-vepfs/cwm/zheng.geng/code/nocs/vggt/tools/dataset_validation_report.png')
    
    return results


if __name__ == "__main__":
    main()

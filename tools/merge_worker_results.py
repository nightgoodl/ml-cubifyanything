#!/usr/bin/env python3
"""
合并多个worker的处理结果
"""

import os
import sys
import argparse
import json
import shutil
import glob
from pathlib import Path
from collections import defaultdict

def merge_worker_results(base_output_dir, split, total_workers, output_dir=None):
    """合并所有worker的结果"""
    
    if output_dir is None:
        output_dir = os.path.join(base_output_dir, f"{split}_merged")
    
    print(f"🔄 开始合并 {total_workers} 个worker的结果...")
    print(f"📁 基础目录: {base_output_dir}")
    print(f"📤 合并输出: {output_dir}")
    
    # 创建合并输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计信息
    total_scenes = 0
    total_instances = 0
    merged_scenes = []
    failed_workers = []
    
    # 处理每个worker的结果
    for worker_id in range(total_workers):
        worker_dir = os.path.join(base_output_dir, split, f"worker_{worker_id}")
        
        if not os.path.exists(worker_dir):
            print(f"⚠️  Worker {worker_id} 目录不存在: {worker_dir}")
            failed_workers.append(worker_id)
            continue
            
        print(f"📂 处理Worker {worker_id}: {worker_dir}")
        
        # 查找场景目录
        scene_dirs = [d for d in os.listdir(worker_dir) 
                     if os.path.isdir(os.path.join(worker_dir, d)) and d.startswith('ca1m-')]
        
        worker_scenes = len(scene_dirs)
        total_scenes += worker_scenes
        
        print(f"   📊 发现 {worker_scenes} 个场景")
        
        # 复制每个场景的结果
        for scene_name in scene_dirs:
            src_scene_dir = os.path.join(worker_dir, scene_name)
            dst_scene_dir = os.path.join(output_dir, scene_name)
            
            # 检查目标是否已存在（避免重复）
            if os.path.exists(dst_scene_dir):
                print(f"   ⚠️  场景 {scene_name} 已存在，跳过")
                continue
            
            # 复制整个场景目录
            try:
                shutil.copytree(src_scene_dir, dst_scene_dir)
                merged_scenes.append(scene_name)
                
                # 统计实例数量
                objects_dir = os.path.join(dst_scene_dir, "objects")
                if os.path.exists(objects_dir):
                    instance_files = glob.glob(os.path.join(objects_dir, "instance_*.ply"))
                    total_instances += len(instance_files)
                
                print(f"   ✅ 复制场景: {scene_name}")
                
            except Exception as e:
                print(f"   ❌ 复制失败 {scene_name}: {e}")
    
    # 创建合并统计报告
    report = {
        "merge_info": {
            "total_workers": total_workers,
            "successful_workers": total_workers - len(failed_workers),
            "failed_workers": failed_workers,
            "merge_timestamp": Path(__file__).stat().st_mtime
        },
        "statistics": {
            "total_scenes": len(merged_scenes),
            "total_instances": total_instances,
            "merged_scenes": sorted(merged_scenes)
        }
    }
    
    # 保存合并报告
    report_file = os.path.join(output_dir, "merge_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # 打印合并结果
    print("\n" + "="*80)
    print("📊 合并完成统计报告")
    print("="*80)
    print(f"✅ 成功worker数: {total_workers - len(failed_workers)}/{total_workers}")
    print(f"📊 合并场景数: {len(merged_scenes)}")
    print(f"🎯 总实例数: {total_instances}")
    print(f"📁 合并输出目录: {output_dir}")
    print(f"📝 合并报告: {report_file}")
    
    if failed_workers:
        print(f"\n❌ 失败的workers: {failed_workers}")
    
    print("="*80)
    
    return len(merged_scenes), total_instances

def verify_completeness(merged_dir, expected_scenes=None):
    """验证合并结果的完整性"""
    print(f"\n🔍 验证合并结果完整性...")
    
    # 统计场景
    scene_dirs = [d for d in os.listdir(merged_dir) 
                 if os.path.isdir(os.path.join(merged_dir, d)) and d.startswith('ca1m-')]
    
    print(f"📊 发现场景数: {len(scene_dirs)}")
    
    # 检查每个场景的完整性
    incomplete_scenes = []
    total_nocs = 0
    total_objects = 0
    
    for scene_name in scene_dirs:
        scene_dir = os.path.join(merged_dir, scene_name)
        
        # 检查必需的子目录
        nocs_dir = os.path.join(scene_dir, "nocs_images")
        objects_dir = os.path.join(scene_dir, "objects")
        bbox_file = os.path.join(scene_dir, "scene_bbox_info.json")
        
        missing_items = []
        if not os.path.exists(nocs_dir):
            missing_items.append("nocs_images")
        else:
            nocs_files = glob.glob(os.path.join(nocs_dir, "*.png"))
            total_nocs += len(nocs_files)
            
        if not os.path.exists(objects_dir):
            missing_items.append("objects")
        else:
            object_files = glob.glob(os.path.join(objects_dir, "instance_*.ply"))
            total_objects += len(object_files)
            
        if not os.path.exists(bbox_file):
            missing_items.append("scene_bbox_info.json")
        
        if missing_items:
            incomplete_scenes.append((scene_name, missing_items))
    
    # 打印验证结果
    print(f"📸 总NOCS图像: {total_nocs}")
    print(f"🎯 总对象文件: {total_objects}")
    
    if incomplete_scenes:
        print(f"\n⚠️  发现 {len(incomplete_scenes)} 个不完整场景:")
        for scene_name, missing in incomplete_scenes:
            print(f"   - {scene_name}: 缺少 {', '.join(missing)}")
    else:
        print(f"\n✅ 所有场景都完整！")
    
    if expected_scenes is not None:
        if len(scene_dirs) < expected_scenes:
            print(f"\n⚠️  预期场景数: {expected_scenes}, 实际: {len(scene_dirs)}")
        else:
            print(f"\n✅ 场景数量符合预期: {len(scene_dirs)}")

def main():
    parser = argparse.ArgumentParser(description="合并分布式worker处理结果")
    parser.add_argument("--base-output-dir", required=True,
                       help="worker结果的基础输出目录")
    parser.add_argument("--split", choices=["train", "val"], required=True,
                       help="数据集划分")
    parser.add_argument("--total-workers", type=int, required=True,
                       help="总worker数量")
    parser.add_argument("--output-dir", default=None,
                       help="合并后的输出目录（默认为base-output-dir/split_merged）")
    parser.add_argument("--verify", action="store_true",
                       help="验证合并结果的完整性")
    parser.add_argument("--expected-scenes", type=int, default=None,
                       help="预期的场景数量（用于验证）")
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔄 CA-1M分布式结果合并工具")
    print("="*80)
    
    # 合并结果
    merged_scenes, total_instances = merge_worker_results(
        args.base_output_dir, 
        args.split, 
        args.total_workers, 
        args.output_dir
    )
    
    # 验证完整性（如果要求）
    if args.verify:
        output_dir = args.output_dir or os.path.join(args.base_output_dir, f"{args.split}_merged")
        verify_completeness(output_dir, args.expected_scenes)
    
    print(f"\n🎉 合并完成！")

if __name__ == "__main__":
    main()

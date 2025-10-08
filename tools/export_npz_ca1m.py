# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

"""仅生成 CA-1M 场景 NPZ 数据的工具脚本。"""

import argparse
import glob
import os
import sys
import time
import numpy as np

from collections import defaultdict
from pathlib import Path
from PIL import Image

from cubifyanything.boxes import GeneralInstance3DBoxes
from cubifyanything.dataset import CubifyAnythingDataset

from tools.demo_multithread_ca1m import (
    extract_bbox_info_from_instances,
    generate_instance_nocs_map,
    unproject,
)


def to_object_array(obj):
    """封装 Python 对象为 0 维 object 数组，便于 np.savez 存储。"""
    wrapper = np.empty((), dtype=object)
    wrapper[()] = obj
    return wrapper


def process_scene_for_npz(scene_path: str, output_root: str) -> dict:
    scene_name = Path(scene_path).stem
    dataset = CubifyAnythingDataset(
        [Path(scene_path).as_uri()],
        yield_world_instances=True,
        load_arkit_depth=True,
        use_cache=False,
    )

    scene_colors = []
    scene_points = []
    scene_depths = []
    extrinsics_list = []
    intrinsics_list = []
    nocs_maps = []
    segmentation_list = []
    scene_instance_ids = {}
    scene_poses = {}
    bbox_dict = {}
    model_names_set = set()
    model_dims = defaultdict(list)
    scene_bbox_info = {}

    frame_count = 0

    for sample in dataset:
        if "gt" not in sample or sample["gt"] is None or "depth" not in sample["gt"]:
            continue

        frame_count += 1
        depth_gt = sample["gt"]["depth"][-1]
        height, width = depth_gt.shape[-2:]

        # 匹配 RGB
        image = np.moveaxis(sample["wide"]["image"][-1].numpy(), 0, -1)
        matched_image_np = np.array(Image.fromarray(image).resize((width, height)))

        RT_camera_to_world = sample["sensor_info"].gt.RT[-1]
        xyz, valid = unproject(
            depth_gt,
            sample["sensor_info"].gt.depth.K[-1],
            RT_camera_to_world,
            max_depth=10.0,
        )

        depth_np = depth_gt.cpu().numpy().astype(np.float32)
        xyz_np = xyz.cpu().numpy().astype(np.float32)
        extrinsic_world_to_camera = np.linalg.inv(RT_camera_to_world.cpu().numpy()).astype(np.float32)
        intrinsic_np = sample["sensor_info"].gt.depth.K[-1].cpu().numpy().astype(np.float32)

        scene_colors.append(matched_image_np.astype(np.uint8))
        scene_depths.append(depth_np)
        scene_points.append(xyz_np)
        extrinsics_list.append(extrinsic_world_to_camera)
        intrinsics_list.append(intrinsic_np)

        frame_key = f"{frame_count-1:04d}"
        frame_segmentation = np.full((height, width), 255, dtype=np.uint8)
        frame_nocs = np.full((height, width, 3), 255, dtype=np.uint8)
        frame_instance_ids = np.zeros((0,), dtype=np.int32)
        frame_poses = np.zeros((0, 4, 4), dtype=np.float32)
        frame_bbox = {}

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
                centers_homogeneous = np.concatenate(
                    [centers, np.ones((centers.shape[0], 1))],
                    axis=1,
                )
                transformed_centers = (RT_np @ centers_homogeneous.T).T[:, :3]

                transformed_R = RT_np[:3, :3] @ original_boxes.R.cpu().numpy()
                transformed_boxes = GeneralInstance3DBoxes(
                    np.concatenate(
                        [transformed_centers, original_boxes.dims.cpu().numpy()],
                        axis=1,
                    ),
                    transformed_R,
                )

                world_gt_instances.set("gt_boxes_3d", transformed_boxes)

        if world_gt_instances is None or len(world_gt_instances) == 0:
            nocs_maps.append(frame_nocs)
            segmentation_list.append(frame_segmentation)
            scene_instance_ids[frame_key] = frame_instance_ids.astype(np.int32)
            scene_poses[frame_key] = frame_poses.astype(np.float32)
            bbox_dict[frame_key] = frame_bbox
            continue

        if frame_count == 1:
            scene_bbox_info = extract_bbox_info_from_instances(world_gt_instances)

        boxes = world_gt_instances.get("gt_boxes_3d") if world_gt_instances.has("gt_boxes_3d") else None
        names = world_gt_instances.get("gt_names") if world_gt_instances.has("gt_names") else []
        if boxes is not None:
            dims_np = boxes.dims.cpu().numpy()
            for name, dims in zip(names, dims_np):
                model_names_set.add(str(name))
                model_dims[str(name)].append(np.array(dims, dtype=np.float32))

        nocs_result = generate_instance_nocs_map(
            xyz_np,
            world_gt_instances,
            intrinsic_np,
            RT_camera_to_world.cpu().numpy(),
            depth_gt.shape[-2:],
            return_aux=True,
        )

        if nocs_result is not None:
            nocs_image, aux_data = nocs_result
            if nocs_image is not None:
                frame_nocs = nocs_image.astype(np.uint8)
            if aux_data is not None:
                frame_segmentation = aux_data.get("segmentation", frame_segmentation)
                frame_instance_ids = aux_data.get("instance_ids", frame_instance_ids).astype(np.int32)
                frame_poses = aux_data.get("poses", frame_poses).astype(np.float32)

                bbox_entries = aux_data.get("bbox", [])
                for entry in bbox_entries:
                    instance_id = int(entry["instance_id"])
                    frame_bbox[instance_id] = entry["bbox"]

                instance_names = aux_data.get("instance_names", [])
                for name in instance_names:
                    model_names_set.add(str(name))

        raw_ids = world_gt_instances.get("gt_ids") if world_gt_instances.has("gt_ids") else []
        if frame_instance_ids.size == 0 and raw_ids:
            fallback_ids = []
            for idx, raw_id in enumerate(raw_ids):
                try:
                    fallback_ids.append(int(raw_id))
                except (TypeError, ValueError):
                    fallback_ids.append(idx + 1)
            frame_instance_ids = np.array(fallback_ids, dtype=np.int32)

        if frame_poses.shape[0] == 0 and boxes is not None and len(boxes) > 0:
            pose_list = []
            centers_np = boxes.gravity_center.cpu().numpy()
            rotations_np = boxes.R.cpu().numpy()
            for center, rotation in zip(centers_np, rotations_np):
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = rotation.astype(np.float32)
                pose[:3, 3] = center.astype(np.float32)
                pose_list.append(pose)
            if pose_list:
                frame_poses = np.stack(pose_list, axis=0)

        if not frame_bbox and boxes is not None and len(boxes) > 0:
            centers_np = boxes.gravity_center.cpu().numpy()
            dims_np = boxes.dims.cpu().numpy()
            rotations_np = boxes.R.cpu().numpy()
            for idx, (center, dims, rotation) in enumerate(zip(centers_np, dims_np, rotations_np)):
                instance_id = int(frame_instance_ids[idx]) if idx < len(frame_instance_ids) else idx
                frame_bbox[instance_id] = {
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "z": float(center[2]),
                    "w": float(dims[0]),
                    "h": float(dims[1]),
                    "l": float(dims[2]),
                    "R": rotation.astype(np.float32),
                }

        # 记录 frame 级别数据
        nocs_maps.append(frame_nocs)
        segmentation_list.append(frame_segmentation)
        scene_instance_ids[frame_key] = frame_instance_ids.astype(np.int32)
        scene_poses[frame_key] = frame_poses.astype(np.float32)
        bbox_dict[frame_key] = frame_bbox

    # 聚合数组
    if scene_colors:
        scene_colors_np = np.stack(scene_colors, axis=0).astype(np.uint8)
    else:
        scene_colors_np = np.zeros((0, 0, 0, 3), dtype=np.uint8)

    if scene_points:
        scene_points_np = np.stack(scene_points, axis=0).astype(np.float32)
    else:
        scene_points_np = np.zeros((0, 0, 0, 3), dtype=np.float32)

    if scene_depths:
        scene_depths_np = np.stack(scene_depths, axis=0).astype(np.float32)
    else:
        scene_depths_np = np.zeros((0, 0, 0), dtype=np.float32)

    if extrinsics_list:
        extrinsics_np = np.stack(extrinsics_list, axis=0).astype(np.float32)
    else:
        extrinsics_np = np.zeros((0, 4, 4), dtype=np.float32)

    if intrinsics_list:
        intrinsics_np = np.stack(intrinsics_list, axis=0).astype(np.float32)
    else:
        intrinsics_np = np.zeros((0, 3, 3), dtype=np.float32)

    if nocs_maps:
        nocs_maps_np = np.stack(nocs_maps, axis=0).astype(np.uint8)
    else:
        nocs_maps_np = np.zeros((0, 0, 0, 3), dtype=np.uint8)

    if segmentation_list:
        masks_np = np.stack(segmentation_list, axis=0).astype(np.uint8)
    else:
        masks_np = np.zeros((0, 0, 0), dtype=np.uint8)

    model_list_sorted = sorted(model_names_set)
    if model_list_sorted:
        scene_scales_np = np.zeros((len(model_list_sorted), 3), dtype=np.float32)
        for idx, name in enumerate(model_list_sorted):
            dims_list = model_dims.get(name, [])
            if dims_list:
                scene_scales_np[idx] = np.mean(np.stack(dims_list, axis=0), axis=0).astype(np.float32)
            else:
                scene_scales_np[idx] = 0.0
        model_list_np = np.array(model_list_sorted, dtype=object)
    else:
        scene_scales_np = np.zeros((0, 3), dtype=np.float32)
        model_list_np = np.array([], dtype=object)

    scene_instance_ids_dict = {key: np.asarray(val, dtype=np.int32) for key, val in scene_instance_ids.items()}
    scene_poses_dict = {key: np.asarray(val, dtype=np.float32) for key, val in scene_poses.items()}
    bbox_serializable = {}
    for key, bbox_per_frame in bbox_dict.items():
        bbox_serializable[key] = {}
        for instance_id, bbox_entry in bbox_per_frame.items():
            bbox_serializable[key][int(instance_id)] = {
                "x": float(bbox_entry["x"]),
                "y": float(bbox_entry["y"]),
                "z": float(bbox_entry["z"]),
                "w": float(bbox_entry["w"]),
                "h": float(bbox_entry["h"]),
                "l": float(bbox_entry["l"]),
                "R": np.asarray(bbox_entry["R"], dtype=np.float32),
            }

    scene_data = dict(
        scene_colors=scene_colors_np,
        scene_points=scene_points_np,
        scene_depths=scene_depths_np,
        extrinsics=extrinsics_np,
        intrinsics=intrinsics_np,
        scene_scales=scene_scales_np,
        scene_poses=to_object_array(scene_poses_dict),
        masks=masks_np,
        nocs_maps=nocs_maps_np,
        scene_instance_ids=to_object_array(scene_instance_ids_dict),
        model_list=model_list_np,
        bbox=to_object_array(bbox_serializable),
    )

    scene_dir = os.path.join(output_root, scene_name)
    os.makedirs(scene_dir, exist_ok=True)
    np.savez_compressed(os.path.join(scene_dir, f"{scene_name}.npz"), **scene_data)

    if scene_bbox_info:
        np.savez_compressed(
            os.path.join(scene_dir, "scene_bbox_info.npz"),
            bbox=to_object_array(scene_bbox_info),
        )

    return {
        "scene": scene_name,
        "frames": frame_count,
        "instances": sum(len(v) for v in scene_instance_ids_dict.values()),
    }


def get_scene_files(data_dir: str, split: str) -> list:
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        raise ValueError(f"Split 目录不存在: {split_dir}")

    scene_files = glob.glob(os.path.join(split_dir, "ca1m-*.tar"))
    if not scene_files:
        raise ValueError(f"在 {split_dir} 中未找到任何 ca1m-*.tar 文件")

    return sorted(scene_files)


def main():
    parser = argparse.ArgumentParser(description="仅生成 CA-1M NPZ 的数据导出工具")
    parser.add_argument(
        "--data-dir",
        default="/baai-cwm-vepfs/cwm/chongjie.ye/data/CA-1M/data",
        help="CA-1M 数据集根目录",
    )
    parser.add_argument(
        "--output-dir",
        default="/baai-cwm-backup/cwm/chongjie.ye/data/CA-1M_npz",
        help="NPZ 输出目录",
    )
    parser.add_argument("--split", choices=["train", "val"], required=True, help="数据划分")
    parser.add_argument("--max-scenes", type=int, default=None, help="仅处理前 N 个场景")

    args = parser.parse_args()

    split_output = os.path.join(args.output_dir, args.split)
    os.makedirs(split_output, exist_ok=True)

    try:
        scene_files = get_scene_files(args.data_dir, args.split)
    except ValueError as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    if args.max_scenes is not None:
        scene_files = scene_files[: args.max_scenes]
        print(f"🔬 测试模式: 仅处理 {len(scene_files)} 个场景")

    start_time = time.time()
    results = []
    failed = []

    for scene_path in scene_files:
        scene_name = Path(scene_path).stem
        scene_dir = os.path.join(split_output, scene_name)
        marker_path = os.path.join(scene_dir, "scene_complete.marker")

        if os.path.exists(marker_path):
            print(f"⏭️ 跳过 {scene_name}: 已存在 marker")
            continue

        print(f"\n🎬 处理场景: {scene_name}")
        try:
            stats = process_scene_for_npz(scene_path, split_output)
            results.append(stats)
            with open(marker_path, "w", encoding="utf-8") as marker:
                marker.write(str(time.time()))
            print(f"✅ 场景 {scene_name} 处理完成")
        except Exception as exc:
            failed.append((scene_name, str(exc)))
            print(f"❌ 场景 {scene_name} 处理失败: {exc}")

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("📊 NPZ 导出统计报告")
    print("=" * 80)
    print(f"🕒 总耗时: {total_time:.2f}s")
    print(f"✅ 成功场景: {len(results)}")
    print(f"❌ 失败场景: {len(failed)}")

    if results:
        total_frames = sum(r["frames"] for r in results)
        total_instances = sum(r["instances"] for r in results)
        print(f"📷 总帧数: {total_frames}")
        print(f"🎯 总实例数: {total_instances}")

    if failed:
        print("\n❌ 失败列表:")
        for scene_name, error in failed:
            print(f"  - {scene_name}: {error}")

    print(f"\n📂 输出目录: {split_output}")


if __name__ == "__main__":
    main()

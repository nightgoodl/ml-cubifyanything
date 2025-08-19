#!/bin/bash

# CA-1M数据集处理脚本
# 使用方法: ./run_ca1m_processing.sh [train|val] [可选参数]

set -e

# 默认参数
DATA_DIR="/baai-cwm-vepfs/cwm/chongjie.ye/data/CA-1M/data"
OUTPUT_DIR="/baai-cwm-vepfs/cwm/chongjie.ye/data/CA-1M_output"
VOXEL_SIZE=0.004
MAX_WORKERS=32
SCENE_WORKERS=16

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 [train|val] [可选参数]"
    echo ""
    echo "可选参数:"
    echo "  --max-scenes N          最大处理场景数（测试用）"
    echo "  --disable-downsampling  禁用体素下采样"
    echo "  --voxel-size SIZE       体素下采样尺寸（默认: 0.004）"
    echo "  --max-workers N         最大线程数（默认: 4）"
    echo "  --scene-workers N       场景处理线程数（默认: 2）"
    echo ""
    echo "示例:"
    echo "  $0 train                # 处理训练集"
    echo "  $0 val --max-scenes 10  # 测试模式，仅处理验证集前10个场景"
    exit 1
fi

SPLIT=$1
shift

# 验证split参数
if [ "$SPLIT" != "train" ] && [ "$SPLIT" != "val" ]; then
    echo "错误: split必须是 'train' 或 'val'"
    exit 1
fi

echo "=================================="
echo "🚀 开始处理CA-1M数据集"
echo "=================================="
echo "📊 处理划分: $SPLIT"
echo "📁 数据目录: $DATA_DIR"
echo "📤 输出目录: $OUTPUT_DIR"
echo "=================================="

# 切换到工具目录
cd "$(dirname "$0")"

# 运行处理脚本
python demo_multithread_ca1m.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --split "$SPLIT" \
    --voxel-size "$VOXEL_SIZE" \
    --max-workers "$MAX_WORKERS" \
    --scene-workers "$SCENE_WORKERS" \
    "$@"

echo ""
echo "=================================="
echo "✅ 处理完成！"
echo "📁 输出目录: $OUTPUT_DIR/$SPLIT"
echo "=================================="
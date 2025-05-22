#!/usr/bin/env bash
set -eo pipefail

# 0) 位置
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "${ROOT_DIR}"

# 1) 產生訓練資料
 python scripts/generate_dataset.py --samples 2000000 --out data/dataset.npz

# 2) 訓練模型
 python scripts/train_mlp.py \
     --data   data/dataset.npz \
     --epochs 10 \
     --lr 1e-2 \
     --hidden 32 \
     --weight-decay 1e-4 \
     --patience 7 \
     --scheduler step \
     --step-size 30 \
     --gamma 0.5 \
     --out data/kalmannet.pt

# 3) 匯出 C++ header（浮點 + INT8 版）
 python scripts/export_weights.py \
     --checkpoint data/kalmannet.pt \
     --float-hpp  core/include/traccc/fitting/kalman_filter/kalman_gru_trained_weights.hpp \
     --int8-hpp   core/include/traccc/fitting/kalman_filter/kalman_int8_gru_trained_weights.hpp

# 4) 建置 traccc（已在 CMakeLists 開啟 TRACCC_LOAD_TRAINED_WEIGHTS）
cmake -S $HOME/project/traccc -B $HOME/project/traccc/build -DCMAKE_BUILD_TYPE=Release \
    --preset cuda-fp32 \
    -DTRACCC_BUILD_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75" \
    -DTRACCC_BUILD_TESTING=ON \
    -DTRACCC_BUILD_EXAMPLES=ON \
    -DCMAKE_CXX_STANDARD=20 \
    -DTRACCC_USE_ROOT=ON | tee ./log/build01.log

cmake --build $HOME/project/traccc/build -- -j$(nproc) 2>&1 | tee ./log/build02.log

# 5) 模擬 traccc（已在 CMakeLists 開啟 TRACCC_LOAD_TRAINED_WEIGHTS）
./build/bin/traccc_throughput_mt_cuda \
  --detector-file=geometries/odd/odd-detray_geometry_detray.json \
  --material-file=geometries/odd/odd-detray_material_detray.json \
  --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
  --use-detray-detector=on \
  --digitization-file=geometries/odd/odd-digi-geometric-config.json \
  --use-acts-geom-source=on \
  --input-directory=odd/geant4_10muon_10GeV/ \
  --input-events=10 \
  --processed-events=1000 \
  --threads=1 \
  
echo "🎉  全流程完成；新權重已嵌入並重新編譯！"
rm -rf $HOME/project/traccc/build
mkdir -p $HOME/project/traccc/build 
cmake -S $HOME/project/traccc -B $HOME/project/traccc/build \
    --preset cuda-fp32 \
    -DTRACCC_BUILD_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75" \
    -DTRACCC_BUILD_TESTING=ON \
    -DTRACCC_BUILD_EXAMPLES=ON \
    -DCMAKE_CXX_STANDARD=20 \
    -DTRACCC_USE_ROOT=ON | tee ./log/build01.log

cmake --build $HOME/project/traccc/build -- -j$(nproc) 2>&1 | tee ./log/build02.log

grep -P "ptxas info.*Used" ./log/build02.log | \
  sed -e 's/^.*ptxas info/ptxas info/' | sort -u | tee ./log/ptxas.log

# 列出所有有 spill 的行
grep -E "bytes spill (stores|loads)" ./log/build02.log | tee ./log/spill_summary.log

# 若要看完整上下文（包含該 kernel 名稱、register count）
grep -nE "ptxas info.*Function properties" -n ./log/build02.log \
  | cut -d: -f1 | while read L; do
      sed -n "$((L-2)),$((L+3))p" ./log/build02.log \
        | grep -E "spill|Used [0-9]+ registers" && echo
  done | tee ./log/spill.log

grep -rn error -A 2 ./log/build02.log | tee ./log/error.log

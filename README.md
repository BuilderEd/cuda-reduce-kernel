## Simple cuda reduction kernel.
**This code was written for [HPCGames](https://hpcgame.pku.edu.cn/), based on [NVIDIA Guides](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)**
### ‼️Remark
  **This kernel is optimized, but not fully. In recent versions of CUDA, you would prefer ___shfl_down_sync()_ over warp reduction**

This is a demo to compare the computation ability of single thread CPU, multi thread CPU and GPU.  
The task is to extract the an image's local high gradient point. A local pixel block has size 7 * 7. We'll find the point that has the maximum image gradient of each 7 * 7 pixel block. An image with size 739 * 458 is provided

## Requirement
OpenCV > 3.0  
CUDA > 7

## Install
```
git clone https://github.com/zhaozhongch/GPU_vs_CPU.git
mkdir build
cd build
cmake ..
make
```

## Use
Single CPU example
```
cd build
./find_gradient
```
Multi CPU example
```
cd build
./find_gradient_multi_thread
```
GPU example
```
cd build
./find_gradient_cuda
```

## Result Example
They'll all get similar image output like ![demo_result](demo_result.png). Those circles just show the where the local high gradient point is in every 7 by 7 pixel block.
On my `ThinkPad-X1-Extreme` laptop with CPU `Intel(R) Core(TM) i7-8850H CPU @ 2.60GHz` and GPU `GTX 1050`, the single CPU computation time is about 100ms, multi CPU computation time is about 50ms, GPU computation time is about 400us.

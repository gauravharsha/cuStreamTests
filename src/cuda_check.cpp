#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<stdexcept>

void check_for_cuda(int &devCount_per_node) {
  if(cudaGetDeviceCount(&devCount_per_node)!= cudaSuccess)
    throw std::runtime_error("error counting cuda devices, typically that means no cuda devices available.");
  if(devCount_per_node==0) throw std::runtime_error("you're starting this code with cuda support but no device is available");
  for (int i = 0; i < devCount_per_node; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::cout<<"Device Number: " << i << std::endl;
    std::cout<<"  Device name: " << prop.name << std::endl;
    std::cout<<"  Peak Memory Bandwidth (GB/s): "<<2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6<<std::endl;
    std::cout<<"  Compute Capability: "<<prop.major<<"."<<prop.minor<<std::endl;
    std::cout<<"  total global mem: "<<prop.totalGlobalMem/(1024.*1024*1024)<<" GB"<<std::endl;
  }
}

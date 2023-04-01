#include <cuda.h>
#include <ostream>
#include <iostream>
#include <cstdlib>
#include <unistd.h>

#define CHKERR(r) {\
  if (r)\
  {\
    std::cerr << "Error " << r << " on line " << __LINE__  <<".\n";\
    throw std::runtime_error("Non-success error code");\
  }\
}

void run_graph(CUdevice device, CUcontext context){

    CHKERR( cuDeviceGet(&device, 0) );
    CHKERR( cuCtxCreate(&context, 0, device) );
    CHKERR( cuCtxSetCurrent(context) );
    
    CUgraph graph;
    CHKERR(cuGraphCreate(&graph, 0));

    CUmodule mod;

    CHKERR(cuModuleLoad(&mod, "set_to_i.cubin"));

    CUfunction func;
    CHKERR(cuModuleGetFunction(&func, mod, "set_to_i"));


    for (int i=0; i < 10; i++){
      // memAllocNode
      CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams = new CUDA_MEM_ALLOC_NODE_PARAMS();
      *nodeParams = {};

      nodeParams->bytesize = 80000000;
      nodeParams->poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
      nodeParams->poolProps.location.id = device;
      nodeParams->poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

      CUgraphNode* memalloc_node = new CUgraphNode();
      std::cout << "Adding memalloc_node" << std::endl;
      CHKERR(cuGraphAddMemAllocNode(memalloc_node, graph, NULL, 0, nodeParams) );

      CUgraphNode* kernel_node = new CUgraphNode();
      void * kernelArgs[] = { &(nodeParams->dptr) };
      CUDA_KERNEL_NODE_PARAMS* kernelParams = new CUDA_KERNEL_NODE_PARAMS();
      *kernelParams = {0};
      kernelParams->func = func;
      kernelParams->gridDimX = 1;
      kernelParams->gridDimY = 1;
      kernelParams->gridDimZ = 1;
      kernelParams->blockDimX = 1;
      kernelParams->blockDimY = 1;
      kernelParams->blockDimZ = 1;
      kernelParams->kernelParams = kernelArgs;
      kernelParams->extra = NULL;
      kernelParams->sharedMemBytes = 0;

      std::cout << "Adding kernel_node" << std::endl;
      CHKERR(cuGraphAddKernelNode(kernel_node, graph, memalloc_node, 1, kernelParams));

      CUgraphNode* memfree_node = new CUgraphNode();
      std::cout << "Adding memfree_node" << std::endl;
      CHKERR(cuGraphAddMemFreeNode(memfree_node, graph, kernel_node, 1, nodeParams->dptr) );

    }

    CUgraphExec exec;
    std::cout << "Instantiating ExecGraph" << std::endl;
    CHKERR(cuGraphInstantiate(&exec, graph, NULL, NULL, 0));
    
    std::cout << "Destroying Graph" << std::endl;
    CHKERR(cuGraphDestroy(graph));

    std::cout << "Launching ExecGraph" << std::endl;
    for (int i=0; i < 10; i++){
      CHKERR(cuGraphLaunch(exec, 0));
    }
    

    std::cout << "Destroying ExecGraph" << std::endl;
    CHKERR(cuGraphExecDestroy(exec));

    std::cout << "Destroying Module" << std::endl;
    CHKERR(cuModuleUnload(mod));
    

}

void run_without_graph(CUdevice device, CUcontext context){

    CHKERR( cuDeviceGet(&device, 0) );
    CHKERR( cuCtxCreate(&context, 0, device) );
    CHKERR( cuCtxSetCurrent(context) );
    
    CUdeviceptr devptr;
    std::cout << "Allocating devptr" << std::endl;
    CHKERR( cuMemAlloc(&devptr, 800));

    CUmodule mod;

    CHKERR(cuModuleLoad(&mod, "set_to_i.cubin"));

    CUfunction func;
    CHKERR(cuModuleGetFunction(&func, mod, "set_to_i"));

    void * kernelArgs[] = { &devptr };
    std::cout << "Launching Kernel" << std::endl;
    CHKERR(cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, NULL, kernelArgs, NULL));

    std::cout << "Destroying Module" << std::endl;
    CHKERR(cuModuleUnload(mod));

    std::cout << "Destroying devptr" << std::endl;
    CHKERR(cuMemFree(devptr));
}

int main(){

    CUresult err;
    CUdevice device = 0;
    CUcontext context = 0;

    CHKERR( cuInit(0) );

    std::cout << "Running with graph" << std::endl;
    run_graph(device, context);

    // std::cout << "Running without graph" << std::endl;
    // run_without_graph(device, context);
}

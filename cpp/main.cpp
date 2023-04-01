#include <cuda.h>
#include <ostream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <unistd.h>
#include <vector>
#include <cstring>
#include <chrono>

using namespace std::chrono;

#define CHKERR(r) {\
  if (r)\
  {\
    std::cerr << "Error " << r << " on line " << __LINE__  <<".\n";\
    throw std::runtime_error("Non-success error code");\
  }\
}

size_t time_cudagraph(CUdevice device, CUcontext context, size_t SIZE){

    CHKERR( cuDeviceGet(&device, 0) );
    CHKERR( cuCtxCreate(&context, 0, device) );
    CHKERR( cuCtxSetCurrent(context) );
    // loading modules

    CUmodule _pt_mod_5;

    CHKERR(cuModuleLoad(&_pt_mod_5, "/home/mitak2/cudagraph_model/kernels/knl_add.cubin"));

    CUfunction func;
    CHKERR(cuModuleGetFunction(&func, _pt_mod_5, "knl_add"));

    CUmodule _pt_mod_14;

    CHKERR(cuModuleLoad(&_pt_mod_14, "/home/mitak2/cudagraph_model/kernels/knl_where.cubin"));
      
    CUfunction func2;
    CHKERR(cuModuleGetFunction(&func2, _pt_mod_14, "knl_where"));

    unsigned BLOCK_SIZE[3];
    BLOCK_SIZE[0] = 128;
    BLOCK_SIZE[1] = 1;
    BLOCK_SIZE[2] = 1;
    unsigned GRID_SIZE[3];
    GRID_SIZE[0] = 2048;
    GRID_SIZE[1] = 1;
    GRID_SIZE[2] = 1;    


    int* _pt_input[9]; // declare an array of 9 integers pointers of size SIZE

    for (int i = 0; i < 9; i++){
        _pt_input[i] = new int[SIZE];
        memset(_pt_input[i], 1, sizeof(float) * SIZE);
    }

    std::cout << "Created array inputs" << std::endl;

    CUgraph graph;
    CHKERR(cuGraphCreate(&graph, 0));
    CUdeviceptr* _ptr_array_layer4[9];
    CUgraphNode* _ptr_kernel_layer4[9];

    // creating 9x4 grid
    for (int i=0; i < 9; i++){

      CUdeviceptr* _ptr_array[4];
      CUgraphNode* _ptr_kernel[4];

      CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams = new CUDA_MEM_ALLOC_NODE_PARAMS();
      *nodeParams = {};

      nodeParams->bytesize = 8*SIZE;
      nodeParams->poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
      nodeParams->poolProps.location.id = device;
      nodeParams->poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

      CUgraphNode* _pt_memalloc = new CUgraphNode();
      std::cout << "Adding _pt_memalloc_first" << std::endl;
      CHKERR(cuGraphAddMemAllocNode(_pt_memalloc, graph, NULL, 0, nodeParams));

      _ptr_array[0]= &(nodeParams->dptr);
  
      // _pt_kernel

      CUgraphNode* _pt_kernel = new CUgraphNode();

      void * kernelArgs[] = { _ptr_array[0], _pt_input[i] };
      CUDA_KERNEL_NODE_PARAMS* kernelParams = new CUDA_KERNEL_NODE_PARAMS();
      *kernelParams = {0};
      kernelParams->func = func;
      kernelParams->gridDimX = GRID_SIZE[0];
      kernelParams->gridDimY = GRID_SIZE[1];
      kernelParams->gridDimZ = GRID_SIZE[2];
      kernelParams->blockDimX = BLOCK_SIZE[0];
      kernelParams->blockDimY = BLOCK_SIZE[1];
      kernelParams->blockDimZ = BLOCK_SIZE[2];
      kernelParams->kernelParams = kernelArgs;
      kernelParams->extra = NULL;
      kernelParams->sharedMemBytes = 0;

      std::cout << "Adding _pt_kernel" << std::endl;
      CHKERR(cuGraphAddKernelNode(_pt_kernel, graph, _pt_memalloc, 1, kernelParams));
      _ptr_kernel[0] = _pt_kernel;

      for (int j=1; j < 4; j++){
        // _pt_memalloc

        CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams = new CUDA_MEM_ALLOC_NODE_PARAMS();
        *nodeParams = {};

        nodeParams->bytesize = 8*SIZE;
        nodeParams->poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
        nodeParams->poolProps.location.id = device;
        nodeParams->poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

        CUgraphNode* _pt_memalloc = new CUgraphNode();
        std::cout << "Adding _pt_memalloc" << std::endl;
        CHKERR(cuGraphAddMemAllocNode(_pt_memalloc, graph, _ptr_kernel[j-1], 1, nodeParams) );

        _ptr_array[j] = &(nodeParams->dptr);

        // _pt_kernel

        CUgraphNode* _pt_kernel = new CUgraphNode();

        void * kernelArgs[] = { _ptr_array[j], _ptr_array[j-1] };
        CUDA_KERNEL_NODE_PARAMS* kernelParams = new CUDA_KERNEL_NODE_PARAMS();
        *kernelParams = {0};
        kernelParams->func = func;
        kernelParams->gridDimX = GRID_SIZE[0];
        kernelParams->gridDimY = GRID_SIZE[1];
        kernelParams->gridDimZ = GRID_SIZE[2];
        kernelParams->blockDimX = BLOCK_SIZE[0];
        kernelParams->blockDimY = BLOCK_SIZE[1];
        kernelParams->blockDimZ = BLOCK_SIZE[2];
        kernelParams->kernelParams = kernelArgs;
        kernelParams->extra = NULL;
        kernelParams->sharedMemBytes = 0;

        std::cout << "Adding _pt_kernel" << std::endl;
        std::vector<CUgraphNode> nodeDependencies;
        nodeDependencies.push_back(*_pt_memalloc);
        nodeDependencies.push_back(*_ptr_kernel[j-1]);
        CHKERR(cuGraphAddKernelNode(_pt_kernel, graph, nodeDependencies.data(), nodeDependencies.size(), kernelParams));
        _ptr_kernel[j] = _pt_kernel;

        if (j == 3){
          std::cout << "Reached last layer" << std::endl;
          _ptr_array_layer4[i] = _ptr_array[j];
          _ptr_kernel_layer4[i] = _pt_kernel;
        }
      
      }

      for (int j=0; j < 3; j++){
        CUgraphNode* memfree_node = new CUgraphNode();
        std::vector<CUgraphNode> nodeDependencies;
        nodeDependencies.push_back(*_ptr_kernel[j]);
        nodeDependencies.push_back(*_ptr_kernel[j+1]);
        CHKERR(cuGraphAddMemFreeNode(memfree_node, graph, nodeDependencies.data(), nodeDependencies.size(), *_ptr_array[j]) );
      }
    }

    // create first knl_where layer

    CUdeviceptr* _ptr_array_layer5[3];
    CUgraphNode* _ptr_kernel_layer5[3];

    for (int i=0; i<3; i++){

        CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams = new CUDA_MEM_ALLOC_NODE_PARAMS();
        *nodeParams = {};

        nodeParams->bytesize = 8*SIZE;
        nodeParams->poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
        nodeParams->poolProps.location.id = device;
        nodeParams->poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

        CUgraphNode* _pt_memalloc = new CUgraphNode();
        std::cout << "Adding _pt_memalloc" << std::endl;
        std::vector<CUgraphNode> nodeDependencies;
        nodeDependencies.push_back(*_ptr_kernel_layer4[3*i]);
        nodeDependencies.push_back(*_ptr_kernel_layer4[3*i + 1]);
        nodeDependencies.push_back(*_ptr_kernel_layer4[3*i + 2]);
        CHKERR(cuGraphAddMemAllocNode(_pt_memalloc, graph, nodeDependencies.data(), nodeDependencies.size(), nodeParams) );

        _ptr_array_layer5[i] = &(nodeParams->dptr);

        // _pt_kernel

        CUgraphNode* _pt_kernel = new CUgraphNode();

        void * kernelArgs[] = { _ptr_array_layer5[i], _ptr_array_layer4[3*i], _ptr_array_layer4[3*i + 1], _ptr_array_layer4[3*i + 2] };
        CUDA_KERNEL_NODE_PARAMS* kernelParams = new CUDA_KERNEL_NODE_PARAMS();
        *kernelParams = {0};
        kernelParams->func = func2;
        kernelParams->gridDimX = GRID_SIZE[0];
        kernelParams->gridDimY = GRID_SIZE[1];
        kernelParams->gridDimZ = GRID_SIZE[2];
        kernelParams->blockDimX = BLOCK_SIZE[0];
        kernelParams->blockDimY = BLOCK_SIZE[1];
        kernelParams->blockDimZ = BLOCK_SIZE[2];
        kernelParams->kernelParams = kernelArgs;
        kernelParams->extra = NULL;
        kernelParams->sharedMemBytes = 0;

        std::cout << "Adding _pt_kernel" << std::endl;
        std::vector<CUgraphNode> nodeDependencies2;
        nodeDependencies2.push_back(*_pt_memalloc);
        nodeDependencies2.push_back(*_ptr_kernel_layer4[3*i]);
        nodeDependencies2.push_back(*_ptr_kernel_layer4[3*i + 1]);
        nodeDependencies2.push_back(*_ptr_kernel_layer4[3*i + 2]);
        CHKERR(cuGraphAddKernelNode(_pt_kernel, graph, nodeDependencies2.data(), nodeDependencies2.size(), kernelParams));
        _ptr_kernel_layer5[i] = _pt_kernel;

    }

    for (int i=0; i < 3; i++){
      for (int j=0; j < 3; j++){

        CUgraphNode* memfree_node = new CUgraphNode();
        std::vector<CUgraphNode> nodeDependencies;
        nodeDependencies.push_back(*_ptr_kernel_layer5[i]);
        nodeDependencies.push_back(*_ptr_kernel_layer4[3*i+j]);
        CHKERR(cuGraphAddMemFreeNode(memfree_node, graph, nodeDependencies.data(), nodeDependencies.size(), *_ptr_array_layer4[3*i + j]) );

      }
    }

    // create last knl_where layer
  
    CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams = new CUDA_MEM_ALLOC_NODE_PARAMS();
    *nodeParams = {};

    nodeParams->bytesize = 8*SIZE;
    nodeParams->poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams->poolProps.location.id = device;
    nodeParams->poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode* _pt_memalloc = new CUgraphNode();
    std::cout << "Adding _pt_memalloc" << std::endl;
    std::vector<CUgraphNode> nodeDependencies;
    nodeDependencies.push_back(*_ptr_kernel_layer5[0]);
    nodeDependencies.push_back(*_ptr_kernel_layer5[1]);
    nodeDependencies.push_back(*_ptr_kernel_layer5[2]);
    CHKERR(cuGraphAddMemAllocNode(_pt_memalloc, graph, nodeDependencies.data(), nodeDependencies.size(), nodeParams) );

    // _pt_kernel

    CUgraphNode* _pt_kernel = new CUgraphNode();

    void * kernelArgs[] = {&(nodeParams->dptr) , _ptr_array_layer5[0], _ptr_array_layer5[1], _ptr_array_layer5[2] };
    CUDA_KERNEL_NODE_PARAMS* kernelParams = new CUDA_KERNEL_NODE_PARAMS();
    *kernelParams = {0};
    kernelParams->func = func2;
    kernelParams->gridDimX = GRID_SIZE[0];
    kernelParams->gridDimY = GRID_SIZE[1];
    kernelParams->gridDimZ = GRID_SIZE[2];
    kernelParams->blockDimX = BLOCK_SIZE[0];
    kernelParams->blockDimY = BLOCK_SIZE[1];
    kernelParams->blockDimZ = BLOCK_SIZE[2];
    kernelParams->kernelParams = kernelArgs;
    kernelParams->extra = NULL;
    kernelParams->sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel" << std::endl;
    std::vector<CUgraphNode> nodeDependencies2;
    nodeDependencies2.push_back(*_pt_memalloc);
    nodeDependencies2.push_back(*_ptr_kernel_layer5[0]);
    nodeDependencies2.push_back(*_ptr_kernel_layer5[1]);
    nodeDependencies2.push_back(*_ptr_kernel_layer5[2]);
    CHKERR(cuGraphAddKernelNode(_pt_kernel, graph, nodeDependencies2.data(), nodeDependencies2.size(), kernelParams));

    for (int i=0; i < 3; i++){
      CUgraphNode* memfree_node = new CUgraphNode();
      std::vector<CUgraphNode> nodeDependencies;
      nodeDependencies.push_back(*_ptr_kernel_layer5[i]);
      nodeDependencies.push_back(*_pt_kernel);
      CHKERR(cuGraphAddMemFreeNode(memfree_node, graph, nodeDependencies.data(), nodeDependencies.size(), *_ptr_array_layer5[i]) );

    }
   
    CUgraphNode* memfree_node = new CUgraphNode();
    CHKERR(cuGraphAddMemFreeNode(memfree_node, graph, _pt_kernel, 1, nodeParams->dptr) );

    std::string filePath = "/home/mitak2/cudagraph_model/cpp/cudagraph_c++.dot";
    CHKERR(cuGraphDebugDotPrint(graph, filePath.c_str(), 1<<1))
    CUgraphExec exec;
    std::cout << "Instantiating ExecGraph" << std::endl;
    CHKERR(cuGraphInstantiate(&exec, graph, NULL, NULL, 0));
    
    std::cout << "warmup rounds" << std::endl;
    for (int i=0; i < 10; i++){
        CHKERR(cuGraphLaunch(exec, 0));
        //CHKERR( cuCtxSynchronize());
        CUresult error = cuCtxSynchronize();
        if (error != CUDA_SUCCESS) {
          const char* errorString;
          cuGetErrorString(error, &errorString);
          printf("Error initializing CUDA driver API: %s\n", errorString);
          return 1;
        }
    }
  
    size_t n_sim_round = 0;
    int total_sim_time = 5;
    auto start_time = std::chrono::steady_clock::now();
        
    while ((std::chrono::steady_clock::now() - start_time < std::chrono::seconds(total_sim_time)) || (n_sim_round < 100)){
        
        // CHKERR( cuCtxSetCurrent(context) );
        // CHKERR( cuCtxSynchronize());
        
        
        std::cout << "entering hot loop" << std::endl;
        
        for (int i=0; i < 100; i++){
          CHKERR(cuGraphLaunch(exec, 0));
        }

        std::cout << "exiting hot loop" << std::endl;
        // CHKERR(cuCtxSynchronize());

        // Get the current time again
        n_sim_round += 100;
    }

    std::cout << "Destroying Graph" << std::endl;
    CHKERR(cuGraphDestroy(graph));

    std::cout << "Destroying ExecGraph" << std::endl;
    CHKERR(cuGraphExecDestroy(exec)); 

    return (total_sim_time/n_sim_round);
}

int main(){

    CUresult err;
    CUdevice device = 0;
    CUcontext context = 0;

    CHKERR( cuInit(0) );
    std::cout << "Running graph" << std::endl;
    size_t sim_time = time_cudagraph(device, context, 1000000);

    std::cout << sim_time << std::endl;


}

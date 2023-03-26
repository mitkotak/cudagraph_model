#include <cuda.h>
#include <ostream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <unistd.h>
#include <vector>
#include <chrono>

using namespace std::chrono;

#define CHKERR(r) {\
  if (r)\
  {\
    std::cerr << "Error " << r << " on line " << __LINE__  <<".\n";\
    throw std::runtime_error("Non-success error code");\
  }\
}

void create_graph(CUdevice device, CUcontext context, CUgraph graph, size_t SIZE){

    CHKERR( cuDeviceGet(&device, 0) );
    CHKERR( cuCtxCreate(&context, 0, device) );
    CHKERR( cuCtxSetCurrent(context) );
    // loading modules

    CUmodule _pt_mod_5;

    CHKERR(cuModuleLoad(&_pt_mod_5, "/home/mitak2/cudagraph_model/kernels/cd34ec14947a699a74accbe93561eeb6.cubin"));

    CUfunction func;
    CHKERR(cuModuleGetFunction(&func, _pt_mod_5, "knl_add_x1_10000000_x2_1"));

    CUmodule _pt_mod_14;

    CHKERR(cuModuleLoad(&_pt_mod_14, "/home/mitak2/cudagraph_model/kernels/fbb9abdeff9f38bd775feb516ef5048c.cubin"));

    CUfunction func2;
    CHKERR(cuModuleGetFunction(&func2, _pt_mod_14, "knl_where"));

    // create data inputs
    int _pt_data[SIZE];
    std::fill(_pt_data, _pt_data+SIZE, 0.0);

    int _pt_data_0[SIZE];
    std::fill(_pt_data_0, _pt_data_0+SIZE, 0.0);

    int _pt_data_1[SIZE];
    std::fill(_pt_data_1, _pt_data_1+SIZE, 0.0);

    int _pt_data_2[SIZE];
    std::fill(_pt_data_2, _pt_data_2+SIZE, 0.0);

    int _pt_data_3[SIZE];
    std::fill(_pt_data_3, _pt_data_3+SIZE, 0.0);

    int _pt_data_4[SIZE];
    std::fill(_pt_data_4, _pt_data_4+SIZE, 0.0);

    int _pt_data_5[SIZE];
    std::fill(_pt_data_5, _pt_data_5+SIZE, 0.0);

    int _pt_data_6[SIZE];
    std::fill(_pt_data_6, _pt_data_6+SIZE, 0.0);

    int _pt_data_7[SIZE];
    std::fill(_pt_data_7, _pt_data_7+SIZE, 0.0);

    std::cout << "Created array inputs" << std::endl;


    std::cout << "Executing on _pt_data_0" << std::endl;

    // _pt_memalloc_4

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams;
    nodeParams = {};

    nodeParams.bytesize = 8*SIZE;
    nodeParams.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams.poolProps.location.id = device;
    nodeParams.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_4;
    std::cout << "Adding _pt_memalloc_4" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_4, graph, NULL, 0, &nodeParams) );

    CUdeviceptr _pt_array_4 = nodeParams.dptr;

    // _pt_kernel_4

    CUgraphNode _pt_kernel_4;

    void * kernelArgs[] = { &_pt_array_4, &_pt_data };
    CUDA_KERNEL_NODE_PARAMS kernelParams = {0};
    kernelParams.func = func;
    kernelParams.gridDimX = 2048;
    kernelParams.gridDimY = 1;
    kernelParams.gridDimZ = 1;
    kernelParams.blockDimX = 128;
    kernelParams.blockDimY = 1;
    kernelParams.blockDimZ = 1;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = NULL;
    kernelParams.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_4" << std::endl;
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_4, graph, &_pt_memalloc_4, 1, &kernelParams));

    // _pt_memalloc_3

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams2;
    nodeParams2 = {};

    nodeParams2.bytesize = 8*SIZE;
    nodeParams2.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams2.poolProps.location.id = device;
    nodeParams2.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_3;
    std::cout << "Adding _pt_memalloc_3" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_3, graph, &_pt_kernel_4, 1, &nodeParams2) );

    CUdeviceptr _pt_array_3 = nodeParams2.dptr;

    // _pt_kernel_3

    CUgraphNode _pt_kernel_3;

    void * kernelArgs2[] = { &_pt_array_3, &_pt_array_4 };
    CUDA_KERNEL_NODE_PARAMS kernelParams2 = {0};
    kernelParams2.func = func;
    kernelParams2.gridDimX = 2048;
    kernelParams2.gridDimY = 1;
    kernelParams2.gridDimZ = 1;
    kernelParams2.blockDimX = 128;
    kernelParams2.blockDimY = 1;
    kernelParams2.blockDimZ = 1;
    kernelParams2.kernelParams = kernelArgs2;
    kernelParams2.extra = NULL;
    kernelParams2.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_3" << std::endl;
    std::vector<CUgraphNode> nodeDependencies;
    nodeDependencies.push_back(_pt_memalloc_3);
    nodeDependencies.push_back(_pt_kernel_4);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_3, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelParams2));

    // _pt_memalloc_2

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams3;
    nodeParams3 = {};

    nodeParams3.bytesize = 8*SIZE;
    nodeParams3.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams3.poolProps.location.id = device;
    nodeParams3.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_2;
    std::cout << "Adding _pt_memalloc_2" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_2, graph, &_pt_kernel_3, 1, &nodeParams3) );

    CUdeviceptr _pt_array_2 = nodeParams3.dptr;

    // _pt_kernel_2

    CUgraphNode _pt_kernel_2;

    void * kernelArgs3[] = { &_pt_array_2, &_pt_array_3 };
    CUDA_KERNEL_NODE_PARAMS kernelParams3 = {0};
    kernelParams3.func = func;
    kernelParams3.gridDimX = 2048;
    kernelParams3.gridDimY = 1;
    kernelParams3.gridDimZ = 1;
    kernelParams3.blockDimX = 128;
    kernelParams3.blockDimY = 1;
    kernelParams3.blockDimZ = 1;
    kernelParams3.kernelParams = kernelArgs3;
    kernelParams3.extra = NULL;
    kernelParams3.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_2" << std::endl;
    std::vector<CUgraphNode> nodeDependencies2;
    nodeDependencies2.push_back(_pt_memalloc_2);
    nodeDependencies2.push_back(_pt_kernel_3);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_2, graph, nodeDependencies2.data(), nodeDependencies2.size(), &kernelParams3));

    // _pt_memalloc_1

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams4;
    nodeParams4 = {};

    nodeParams4.bytesize = 8*SIZE;
    nodeParams4.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams4.poolProps.location.id = device;
    nodeParams4.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_1;
    std::cout << "Adding _pt_memalloc_1" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_1, graph, &_pt_kernel_2, 1, &nodeParams4) );

    CUdeviceptr _pt_array_1 = nodeParams4.dptr;

    // _pt_kernel_1

    CUgraphNode _pt_kernel_1;

    void * kernelArgs4[] = { &_pt_array_1, &_pt_array_2 };
    CUDA_KERNEL_NODE_PARAMS kernelParams4 = {0};
    kernelParams4.func = func;
    kernelParams4.gridDimX = 2048;
    kernelParams4.gridDimY = 1;
    kernelParams4.gridDimZ = 1;
    kernelParams4.blockDimX = 128;
    kernelParams4.blockDimY = 1;
    kernelParams4.blockDimZ = 1;
    kernelParams4.kernelParams = kernelArgs4;
    kernelParams4.extra = NULL;
    kernelParams4.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_1" << std::endl;
    std::vector<CUgraphNode> nodeDependencies3;
    nodeDependencies3.push_back(_pt_memalloc_1);
    nodeDependencies3.push_back(_pt_kernel_2);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_1, graph, nodeDependencies3.data(), nodeDependencies3.size(), &kernelParams4));


    std::cout << "Executing on _pt_data" << std::endl;

    // _pt_memalloc_8

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams5;
    nodeParams5 = {};

    nodeParams5.bytesize = 8*SIZE;
    nodeParams5.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams5.poolProps.location.id = device;
    nodeParams5.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_8;
    std::cout << "Adding _pt_memalloc_8" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_8, graph, NULL, 0, &nodeParams5) );

    CUdeviceptr _pt_array_8 = nodeParams5.dptr;

    // _pt_kernel_8

    CUgraphNode _pt_kernel_8;

    void * kernelArgs5[] = { &_pt_array_8, &_pt_data_0 };
    CUDA_KERNEL_NODE_PARAMS kernelParams5 = {0};
    kernelParams5.func = func;
    kernelParams5.gridDimX = 2048;
    kernelParams5.gridDimY = 1;
    kernelParams5.gridDimZ = 1;
    kernelParams5.blockDimX = 128;
    kernelParams5.blockDimY = 1;
    kernelParams5.blockDimZ = 1;
    kernelParams5.kernelParams = kernelArgs5;
    kernelParams5.extra = NULL;
    kernelParams5.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_8" << std::endl;
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_8, graph, &_pt_memalloc_8, 1, &kernelParams5));

    // _pt_memalloc_7

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams6;
    nodeParams6 = {};

    nodeParams6.bytesize = 8*SIZE;
    nodeParams6.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams6.poolProps.location.id = device;
    nodeParams6.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_7;
    std::cout << "Adding _pt_memalloc_7" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_7, graph, &_pt_kernel_8, 1, &nodeParams6) );

    CUdeviceptr _pt_array_7 = nodeParams6.dptr;

    // _pt_kernel_7

    CUgraphNode _pt_kernel_7;

    void * kernelArgs6[] = { &_pt_array_7, &_pt_array_8 };
    CUDA_KERNEL_NODE_PARAMS kernelParams6 = {0};
    kernelParams6.func = func;
    kernelParams6.gridDimX = 2048;
    kernelParams6.gridDimY = 1;
    kernelParams6.gridDimZ = 1;
    kernelParams6.blockDimX = 128;
    kernelParams6.blockDimY = 1;
    kernelParams6.blockDimZ = 1;
    kernelParams6.kernelParams = kernelArgs6;
    kernelParams6.extra = NULL;
    kernelParams6.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_7" << std::endl;
    std::vector<CUgraphNode> nodeDependencies4;
    nodeDependencies4.push_back(_pt_memalloc_7);
    nodeDependencies4.push_back(_pt_kernel_8);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_7, graph, nodeDependencies4.data(), nodeDependencies4.size(), &kernelParams6));

    // _pt_memalloc_6

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams7;
    nodeParams7 = {};

    nodeParams7.bytesize = 8*SIZE;
    nodeParams7.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams7.poolProps.location.id = device;
    nodeParams7.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_6;
    std::cout << "Adding _pt_memalloc_6" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_6, graph, &_pt_kernel_7, 1, &nodeParams3) );

    CUdeviceptr _pt_array_6 = nodeParams7.dptr;

    // _pt_kernel_6

    CUgraphNode _pt_kernel_6;

    void * kernelArgs7[] = { &_pt_array_6, &_pt_array_7 };
    CUDA_KERNEL_NODE_PARAMS kernelParams7 = {0};
    kernelParams7.func = func;
    kernelParams7.gridDimX = 2048;
    kernelParams7.gridDimY = 1;
    kernelParams7.gridDimZ = 1;
    kernelParams7.blockDimX = 128;
    kernelParams7.blockDimY = 1;
    kernelParams7.blockDimZ = 1;
    kernelParams7.kernelParams = kernelArgs7;
    kernelParams7.extra = NULL;
    kernelParams7.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_6" << std::endl;
    std::vector<CUgraphNode> nodeDependencies5;
    nodeDependencies5.push_back(_pt_memalloc_6);
    nodeDependencies5.push_back(_pt_kernel_7);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_6, graph, nodeDependencies5.data(), nodeDependencies5.size(), &kernelParams7));

    // _pt_memalloc_5

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams8;
    nodeParams8 = {};

    nodeParams8.bytesize = 8*SIZE;
    nodeParams8.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams8.poolProps.location.id = device;
    nodeParams8.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_5;
    std::cout << "Adding _pt_memalloc_5" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_5, graph, &_pt_kernel_6, 1, &nodeParams8) );

    CUdeviceptr _pt_array_5 = nodeParams8.dptr;

    // _pt_kernel_5

    CUgraphNode _pt_kernel_5;

    void * kernelArgs8[] = { &_pt_array_5, &_pt_array_6 };
    CUDA_KERNEL_NODE_PARAMS kernelParams8 = {0};
    kernelParams8.func = func;
    kernelParams8.gridDimX = 2048;
    kernelParams8.gridDimY = 1;
    kernelParams8.gridDimZ = 1;
    kernelParams8.blockDimX = 128;
    kernelParams8.blockDimY = 1;
    kernelParams8.blockDimZ = 1;
    kernelParams8.kernelParams = kernelArgs8;
    kernelParams8.extra = NULL;
    kernelParams8.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_5" << std::endl;
    std::vector<CUgraphNode> nodeDependencies6;
    nodeDependencies6.push_back(_pt_memalloc_5);
    nodeDependencies6.push_back(_pt_kernel_6);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_5, graph, nodeDependencies6.data(), nodeDependencies6.size(), &kernelParams8));

    std::cout << "Executing on _pt_data_1" << std::endl;

    // _pt_memalloc_12

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams9;
    nodeParams9 = {};

    nodeParams9.bytesize = 8*SIZE;
    nodeParams9.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams9.poolProps.location.id = device;
    nodeParams9.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_12;
    std::cout << "Adding _pt_memalloc_12" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_12, graph, NULL, 0, &nodeParams9) );

    CUdeviceptr _pt_array_12 = nodeParams9.dptr;

    // _pt_kernel_12

    CUgraphNode _pt_kernel_12;

    void * kernelArgs9[] = { &_pt_array_12, &_pt_data_1 };
    CUDA_KERNEL_NODE_PARAMS kernelParams9 = {0};
    kernelParams9.func = func;
    kernelParams9.gridDimX = 2048;
    kernelParams9.gridDimY = 1;
    kernelParams9.gridDimZ = 1;
    kernelParams9.blockDimX = 128;
    kernelParams9.blockDimY = 1;
    kernelParams9.blockDimZ = 1;
    kernelParams9.kernelParams = kernelArgs9;
    kernelParams9.extra = NULL;
    kernelParams9.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_12" << std::endl;
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_12, graph, &_pt_memalloc_12, 1, &kernelParams9));

    // _pt_memalloc_11

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams10;
    nodeParams10 = {};

    nodeParams10.bytesize = 8*SIZE;
    nodeParams10.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams10.poolProps.location.id = device;
    nodeParams10.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_11;
    std::cout << "Adding _pt_memalloc_11" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_11, graph, &_pt_kernel_12, 1, &nodeParams10) );

    CUdeviceptr _pt_array_11 = nodeParams10.dptr;

    // _pt_kernel_11

    CUgraphNode _pt_kernel_11;

    void * kernelArgs10[] = { &_pt_array_11, &_pt_array_12 };
    CUDA_KERNEL_NODE_PARAMS kernelParams10 = {0};
    kernelParams10.func = func;
    kernelParams10.gridDimX = 2048;
    kernelParams10.gridDimY = 1;
    kernelParams10.gridDimZ = 1;
    kernelParams10.blockDimX = 128;
    kernelParams10.blockDimY = 1;
    kernelParams10.blockDimZ = 1;
    kernelParams10.kernelParams = kernelArgs10;
    kernelParams10.extra = NULL;
    kernelParams10.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_11" << std::endl;
    std::vector<CUgraphNode> nodeDependencies7;
    nodeDependencies7.push_back(_pt_memalloc_11);
    nodeDependencies7.push_back(_pt_kernel_12);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_11, graph, nodeDependencies7.data(), nodeDependencies7.size(), &kernelParams10));

    // _pt_memalloc_10

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams11;
    nodeParams11 = {};

    nodeParams11.bytesize = 8*SIZE;
    nodeParams11.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams11.poolProps.location.id = device;
    nodeParams11.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_10;
    std::cout << "Adding _pt_memalloc_10" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_10, graph, &_pt_kernel_11, 1, &nodeParams11) );

    CUdeviceptr _pt_array_10 = nodeParams11.dptr;

    // _pt_kernel_10

    CUgraphNode _pt_kernel_10;

    void * kernelArgs11[] = { &_pt_array_10, &_pt_array_11 };
    CUDA_KERNEL_NODE_PARAMS kernelParams11 = {0};
    kernelParams11.func = func;
    kernelParams11.gridDimX = 2048;
    kernelParams11.gridDimY = 1;
    kernelParams11.gridDimZ = 1;
    kernelParams11.blockDimX = 128;
    kernelParams11.blockDimY = 1;
    kernelParams11.blockDimZ = 1;
    kernelParams11.kernelParams = kernelArgs11;
    kernelParams11.extra = NULL;
    kernelParams11.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_10" << std::endl;
    std::vector<CUgraphNode> nodeDependencies8;
    nodeDependencies8.push_back(_pt_memalloc_10);
    nodeDependencies8.push_back(_pt_kernel_11);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_10, graph, nodeDependencies8.data(), nodeDependencies8.size(), &kernelParams11));

    // _pt_memalloc_9

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams12;
    nodeParams12 = {};

    nodeParams12.bytesize = 8*SIZE;
    nodeParams12.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams12.poolProps.location.id = device;
    nodeParams12.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_9;
    std::cout << "Adding _pt_memalloc_9" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_9, graph, &_pt_kernel_10, 1, &nodeParams12) );

    CUdeviceptr _pt_array_9 = nodeParams12.dptr;

    // _pt_kernel_9

    CUgraphNode _pt_kernel_9;

    void * kernelArgs12[] = { &_pt_array_9, &_pt_array_10 };
    CUDA_KERNEL_NODE_PARAMS kernelParams12 = {0};
    kernelParams12.func = func;
    kernelParams12.gridDimX = 2048;
    kernelParams12.gridDimY = 1;
    kernelParams12.gridDimZ = 1;
    kernelParams12.blockDimX = 128;
    kernelParams12.blockDimY = 1;
    kernelParams12.blockDimZ = 1;
    kernelParams12.kernelParams = kernelArgs12;
    kernelParams12.extra = NULL;
    kernelParams12.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_9" << std::endl;
    std::vector<CUgraphNode> nodeDependencies9;
    nodeDependencies9.push_back(_pt_memalloc_9);
    nodeDependencies9.push_back(_pt_kernel_10);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_9, graph, nodeDependencies9.data(), nodeDependencies9.size(), &kernelParams12));

    // _pt_memalloc_0

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams13;
    nodeParams13 = {};

    nodeParams13.bytesize = 8*SIZE;
    nodeParams13.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams13.poolProps.location.id = device;
    nodeParams13.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_0;
    std::cout << "Adding _pt_memalloc_0" << std::endl;
    std::vector<CUgraphNode> nodeDependencies10;
    nodeDependencies10.push_back(_pt_kernel_1);
    nodeDependencies10.push_back(_pt_kernel_5);
    nodeDependencies10.push_back(_pt_kernel_9);
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_0, graph, nodeDependencies10.data(), nodeDependencies10.size(), &nodeParams13) );

    CUdeviceptr _pt_array_0 = nodeParams13.dptr;

    // _pt_kernel_0

    CUgraphNode _pt_kernel_0;

    void * kernelArgs13[] = { &_pt_array_0, &_pt_array_1, &_pt_array_5, &_pt_array_9 };
    CUDA_KERNEL_NODE_PARAMS kernelParams13 = {0};
    kernelParams13.func = func2;
    kernelParams13.gridDimX = 2048;
    kernelParams13.gridDimY = 1;
    kernelParams13.gridDimZ = 1;
    kernelParams13.blockDimX = 128;
    kernelParams13.blockDimY = 1;
    kernelParams13.blockDimZ = 1;
    kernelParams13.kernelParams = kernelArgs13;
    kernelParams13.extra = NULL;
    kernelParams13.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_0" << std::endl;
    std::vector<CUgraphNode> nodeDependencies11;
    nodeDependencies11.push_back(_pt_memalloc_0);
    nodeDependencies11.push_back(_pt_kernel_1);
    nodeDependencies11.push_back(_pt_kernel_5);
    nodeDependencies11.push_back(_pt_kernel_9);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_0, graph, nodeDependencies11.data(), nodeDependencies11.size(), &kernelParams13));

    
    std::cout << "Executing on _pt_data_2" << std::endl;

    // _pt_memalloc_17

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams14;
    nodeParams14 = {};

    nodeParams14.bytesize = 8*SIZE;
    nodeParams14.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams14.poolProps.location.id = device;
    nodeParams14.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_17;
    std::cout << "Adding _pt_memalloc_17" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_17, graph, NULL, 0, &nodeParams14) );

    CUdeviceptr _pt_array_17 = nodeParams14.dptr;

    // _pt_kernel_17

    CUgraphNode _pt_kernel_17;

    void * kernelArgs14[] = { &_pt_array_17, &_pt_data_2 };
    CUDA_KERNEL_NODE_PARAMS kernelParams14 = {0};
    kernelParams14.func = func;
    kernelParams14.gridDimX = 2048;
    kernelParams14.gridDimY = 1;
    kernelParams14.gridDimZ = 1;
    kernelParams14.blockDimX = 128;
    kernelParams14.blockDimY = 1;
    kernelParams14.blockDimZ = 1;
    kernelParams14.kernelParams = kernelArgs14;
    kernelParams14.extra = NULL;
    kernelParams14.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_17" << std::endl;
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_17, graph, &_pt_memalloc_17, 1, &kernelParams14));

    // _pt_memalloc_16

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams15;
    nodeParams15 = {};

    nodeParams15.bytesize = 8*SIZE;
    nodeParams15.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams15.poolProps.location.id = device;
    nodeParams15.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_16;
    std::cout << "Adding _pt_memalloc_16" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_16, graph, &_pt_kernel_17, 1, &nodeParams15) );

    CUdeviceptr _pt_array_16 = nodeParams15.dptr;

    // _pt_kernel_16

    CUgraphNode _pt_kernel_16;

    void * kernelArgs15[] = { &_pt_array_16, &_pt_array_17 };
    CUDA_KERNEL_NODE_PARAMS kernelParams15 = {0};
    kernelParams15.func = func;
    kernelParams15.gridDimX = 2048;
    kernelParams15.gridDimY = 1;
    kernelParams15.gridDimZ = 1;
    kernelParams15.blockDimX = 128;
    kernelParams15.blockDimY = 1;
    kernelParams15.blockDimZ = 1;
    kernelParams15.kernelParams = kernelArgs15;
    kernelParams15.extra = NULL;
    kernelParams15.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_16" << std::endl;
    std::vector<CUgraphNode> nodeDependencies12;
    nodeDependencies12.push_back(_pt_memalloc_16);
    nodeDependencies12.push_back(_pt_kernel_17);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_16, graph, nodeDependencies12.data(), nodeDependencies12.size(), &kernelParams15));

    // _pt_memalloc_15

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams16;
    nodeParams16 = {};

    nodeParams16.bytesize = 8*SIZE;
    nodeParams16.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams16.poolProps.location.id = device;
    nodeParams16.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_15;
    std::cout << "Adding _pt_memalloc_15" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_15, graph, &_pt_kernel_16, 1, &nodeParams16) );

    CUdeviceptr _pt_array_15 = nodeParams16.dptr;

    // _pt_kernel_15

    CUgraphNode _pt_kernel_15;

    void * kernelArgs17[] = { &_pt_array_15, &_pt_array_16 };
    CUDA_KERNEL_NODE_PARAMS kernelParams17 = {0};
    kernelParams17.func = func;
    kernelParams17.gridDimX = 2048;
    kernelParams17.gridDimY = 1;
    kernelParams17.gridDimZ = 1;
    kernelParams17.blockDimX = 128;
    kernelParams17.blockDimY = 1;
    kernelParams17.blockDimZ = 1;
    kernelParams17.kernelParams = kernelArgs17;
    kernelParams17.extra = NULL;
    kernelParams17.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_15" << std::endl;
    std::vector<CUgraphNode> nodeDependencies13;
    nodeDependencies13.push_back(_pt_memalloc_15);
    nodeDependencies13.push_back(_pt_kernel_16);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_15, graph, nodeDependencies13.data(), nodeDependencies13.size(), &kernelParams17));

    // _pt_memalloc_14

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams18;
    nodeParams18 = {};

    nodeParams18.bytesize = 8*SIZE;
    nodeParams18.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams18.poolProps.location.id = device;
    nodeParams18.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_14;
    std::cout << "Adding _pt_memalloc_14" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_14, graph, &_pt_kernel_15, 1, &nodeParams18) );

    CUdeviceptr _pt_array_14 = nodeParams18.dptr;

    // _pt_kernel_14

    CUgraphNode _pt_kernel_14;

    void * kernelArgs18[] = { &_pt_array_14, &_pt_array_15 };
    CUDA_KERNEL_NODE_PARAMS kernelParams18 = {0};
    kernelParams18.func = func;
    kernelParams18.gridDimX = 2048;
    kernelParams18.gridDimY = 1;
    kernelParams18.gridDimZ = 1;
    kernelParams18.blockDimX = 128;
    kernelParams18.blockDimY = 1;
    kernelParams18.blockDimZ = 1;
    kernelParams18.kernelParams = kernelArgs18;
    kernelParams18.extra = NULL;
    kernelParams18.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_14" << std::endl;
    std::vector<CUgraphNode> nodeDependencies14;
    nodeDependencies14.push_back(_pt_memalloc_14);
    nodeDependencies14.push_back(_pt_kernel_15);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_14, graph, nodeDependencies14.data(), nodeDependencies14.size(), &kernelParams18));


    std::cout << "Executing on _pt_data_3" << std::endl;

    // _pt_memalloc_21

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams19;
    nodeParams19 = {};

    nodeParams19.bytesize = 8*SIZE;
    nodeParams19.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams19.poolProps.location.id = device;
    nodeParams19.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_21;
    std::cout << "Adding _pt_memalloc_21" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_21, graph, NULL, 0, &nodeParams19) );

    CUdeviceptr _pt_array_21 = nodeParams19.dptr;

    // _pt_kernel_21

    CUgraphNode _pt_kernel_21;

    void * kernelArgs19[] = { &_pt_array_21, &_pt_data_3 };
    CUDA_KERNEL_NODE_PARAMS kernelParams19 = {0};
    kernelParams19.func = func;
    kernelParams19.gridDimX = 2048;
    kernelParams19.gridDimY = 1;
    kernelParams19.gridDimZ = 1;
    kernelParams19.blockDimX = 128;
    kernelParams19.blockDimY = 1;
    kernelParams19.blockDimZ = 1;
    kernelParams19.kernelParams = kernelArgs19;
    kernelParams19.extra = NULL;
    kernelParams19.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_21" << std::endl;
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_21, graph, &_pt_memalloc_21, 1, &kernelParams19));

    // _pt_memalloc_20

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams20;
    nodeParams20 = {};

    nodeParams20.bytesize = 8*SIZE;
    nodeParams20.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams20.poolProps.location.id = device;
    nodeParams20.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_20;
    std::cout << "Adding _pt_memalloc_20" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_20, graph, &_pt_kernel_21, 1, &nodeParams20) );

    CUdeviceptr _pt_array_20 = nodeParams20.dptr;

    // _pt_kernel_20

    CUgraphNode _pt_kernel_20;

    void * kernelArgs20[] = { &_pt_array_20, &_pt_array_21 };
    CUDA_KERNEL_NODE_PARAMS kernelParams20 = {0};
    kernelParams20.func = func;
    kernelParams20.gridDimX = 2048;
    kernelParams20.gridDimY = 1;
    kernelParams20.gridDimZ = 1;
    kernelParams20.blockDimX = 128;
    kernelParams20.blockDimY = 1;
    kernelParams20.blockDimZ = 1;
    kernelParams20.kernelParams = kernelArgs20;
    kernelParams20.extra = NULL;
    kernelParams20.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_20" << std::endl;
    std::vector<CUgraphNode> nodeDependencies15;
    nodeDependencies15.push_back(_pt_memalloc_20);
    nodeDependencies15.push_back(_pt_kernel_21);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_20, graph, nodeDependencies15.data(), nodeDependencies15.size(), &kernelParams20));

    // _pt_memalloc_19

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams21;
    nodeParams21 = {};

    nodeParams21.bytesize = 8*SIZE;
    nodeParams21.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams21.poolProps.location.id = device;
    nodeParams21.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_19;
    std::cout << "Adding _pt_memalloc_19" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_19, graph, &_pt_kernel_20, 1, &nodeParams21) );

    CUdeviceptr _pt_array_19 = nodeParams21.dptr;

    // _pt_kernel_19

    CUgraphNode _pt_kernel_19;

    void * kernelArgs22[] = { &_pt_array_19, &_pt_array_20 };
    CUDA_KERNEL_NODE_PARAMS kernelParams22 = {0};
    kernelParams22.func = func;
    kernelParams22.gridDimX = 2048;
    kernelParams22.gridDimY = 1;
    kernelParams22.gridDimZ = 1;
    kernelParams22.blockDimX = 128;
    kernelParams22.blockDimY = 1;
    kernelParams22.blockDimZ = 1;
    kernelParams22.kernelParams = kernelArgs22;
    kernelParams22.extra = NULL;
    kernelParams22.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_19" << std::endl;
    std::vector<CUgraphNode> nodeDependencies16;
    nodeDependencies16.push_back(_pt_memalloc_19);
    nodeDependencies16.push_back(_pt_kernel_20);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_19, graph, nodeDependencies16.data(), nodeDependencies16.size(), &kernelParams22));

    // _pt_memalloc_18

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams23;
    nodeParams23 = {};

    nodeParams23.bytesize = 8*SIZE;
    nodeParams23.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams23.poolProps.location.id = device;
    nodeParams23.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_18;
    std::cout << "Adding _pt_memalloc_18" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_18, graph, &_pt_kernel_19, 1, &nodeParams23) );

    CUdeviceptr _pt_array_18 = nodeParams23.dptr;

    // _pt_kernel_18

    CUgraphNode _pt_kernel_18;

    void * kernelArgs24[] = { &_pt_array_18, &_pt_array_19 };
    CUDA_KERNEL_NODE_PARAMS kernelParams24 = {0};
    kernelParams24.func = func;
    kernelParams24.gridDimX = 2048;
    kernelParams24.gridDimY = 1;
    kernelParams24.gridDimZ = 1;
    kernelParams24.blockDimX = 128;
    kernelParams24.blockDimY = 1;
    kernelParams24.blockDimZ = 1;
    kernelParams24.kernelParams = kernelArgs24;
    kernelParams24.extra = NULL;
    kernelParams24.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_18" << std::endl;
    std::vector<CUgraphNode> nodeDependencies17;
    nodeDependencies17.push_back(_pt_memalloc_18);
    nodeDependencies17.push_back(_pt_kernel_19);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_18, graph, nodeDependencies17.data(), nodeDependencies17.size(), &kernelParams24));

    std::cout << "Executing on _pt_data_4" << std::endl;

    // _pt_memalloc_25

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams25;
    nodeParams25 = {};

    nodeParams25.bytesize = 8*SIZE;
    nodeParams25.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams25.poolProps.location.id = device;
    nodeParams25.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_25;
    std::cout << "Adding _pt_memalloc_25" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_25, graph, NULL, 0, &nodeParams25) );

    CUdeviceptr _pt_array_25 = nodeParams25.dptr;

    // _pt_kernel_12

    CUgraphNode _pt_kernel_25;

    void * kernelArgs25[] = { &_pt_array_25, &_pt_data_4 };
    CUDA_KERNEL_NODE_PARAMS kernelParams25 = {0};
    kernelParams25.func = func;
    kernelParams25.gridDimX = 2048;
    kernelParams25.gridDimY = 1;
    kernelParams25.gridDimZ = 1;
    kernelParams25.blockDimX = 128;
    kernelParams25.blockDimY = 1;
    kernelParams25.blockDimZ = 1;
    kernelParams25.kernelParams = kernelArgs25;
    kernelParams25.extra = NULL;
    kernelParams25.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_25" << std::endl;
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_25, graph, &_pt_memalloc_25, 1, &kernelParams25));

    // _pt_memalloc_24

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams26;
    nodeParams26 = {};

    nodeParams26.bytesize = 8*SIZE;
    nodeParams26.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams26.poolProps.location.id = device;
    nodeParams26.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_24;
    std::cout << "Adding _pt_memalloc_24" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_24, graph, &_pt_kernel_25, 1, &nodeParams26) );

    CUdeviceptr _pt_array_24 = nodeParams26.dptr;

    // _pt_kernel_24

    CUgraphNode _pt_kernel_24;

    void * kernelArgs27[] = { &_pt_array_24, &_pt_array_25 };
    CUDA_KERNEL_NODE_PARAMS kernelParams27 = {0};
    kernelParams27.func = func;
    kernelParams27.gridDimX = 2048;
    kernelParams27.gridDimY = 1;
    kernelParams27.gridDimZ = 1;
    kernelParams27.blockDimX = 128;
    kernelParams27.blockDimY = 1;
    kernelParams27.blockDimZ = 1;
    kernelParams27.kernelParams = kernelArgs27;
    kernelParams27.extra = NULL;
    kernelParams27.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_24" << std::endl;
    std::vector<CUgraphNode> nodeDependencies18;
    nodeDependencies18.push_back(_pt_memalloc_24);
    nodeDependencies18.push_back(_pt_kernel_25);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_24, graph, nodeDependencies18.data(), nodeDependencies18.size(), &kernelParams27));

    // _pt_memalloc_23

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams28;
    nodeParams28 = {};

    nodeParams28.bytesize = 8*SIZE;
    nodeParams28.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams28.poolProps.location.id = device;
    nodeParams28.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_23;
    std::cout << "Adding _pt_memalloc_23" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_23, graph, &_pt_kernel_24, 1, &nodeParams28) );

    CUdeviceptr _pt_array_23 = nodeParams28.dptr;

    // _pt_kernel_23

    CUgraphNode _pt_kernel_23;

    void * kernelArgs28[] = { &_pt_array_23, &_pt_array_24 };
    CUDA_KERNEL_NODE_PARAMS kernelParams28 = {0};
    kernelParams28.func = func;
    kernelParams28.gridDimX = 2048;
    kernelParams28.gridDimY = 1;
    kernelParams28.gridDimZ = 1;
    kernelParams28.blockDimX = 128;
    kernelParams28.blockDimY = 1;
    kernelParams28.blockDimZ = 1;
    kernelParams28.kernelParams = kernelArgs28;
    kernelParams28.extra = NULL;
    kernelParams28.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_23" << std::endl;
    std::vector<CUgraphNode> nodeDependencies19;
    nodeDependencies19.push_back(_pt_memalloc_23);
    nodeDependencies19.push_back(_pt_kernel_24);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_23, graph, nodeDependencies19.data(), nodeDependencies19.size(), &kernelParams28));

    // _pt_memalloc_22

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams29;
    nodeParams29 = {};

    nodeParams29.bytesize = 8*SIZE;
    nodeParams29.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams29.poolProps.location.id = device;
    nodeParams29.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_22;
    std::cout << "Adding _pt_memalloc_22" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_22, graph, &_pt_kernel_23, 1, &nodeParams29) );

    CUdeviceptr _pt_array_22 = nodeParams29.dptr;

    // _pt_kernel_22

    CUgraphNode _pt_kernel_22;

    void * kernelArgs29[] = { &_pt_array_22, &_pt_array_23 };
    CUDA_KERNEL_NODE_PARAMS kernelParams29 = {0};
    kernelParams29.func = func;
    kernelParams29.gridDimX = 2048;
    kernelParams29.gridDimY = 1;
    kernelParams29.gridDimZ = 1;
    kernelParams29.blockDimX = 128;
    kernelParams29.blockDimY = 1;
    kernelParams29.blockDimZ = 1;
    kernelParams29.kernelParams = kernelArgs29;
    kernelParams29.extra = NULL;
    kernelParams29.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_22" << std::endl;
    std::vector<CUgraphNode> nodeDependencies20;
    nodeDependencies20.push_back(_pt_memalloc_22);
    nodeDependencies20.push_back(_pt_kernel_23);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_22, graph, nodeDependencies20.data(), nodeDependencies20.size(), &kernelParams29));

    // _pt_memalloc_13

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams30;
    nodeParams30 = {};

    nodeParams30.bytesize = 8*SIZE;
    nodeParams30.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams30.poolProps.location.id = device;
    nodeParams30.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_13;
    std::cout << "Adding _pt_memalloc_13" << std::endl;
    std::vector<CUgraphNode> nodeDependencies21;
    nodeDependencies21.push_back(_pt_kernel_22);
    nodeDependencies21.push_back(_pt_kernel_18);
    nodeDependencies21.push_back(_pt_kernel_14);
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_13, graph, nodeDependencies21.data(), nodeDependencies21.size(), &nodeParams30) );

    CUdeviceptr _pt_array_13 = nodeParams30.dptr;

    // _pt_kernel_13

    CUgraphNode _pt_kernel_13;

    void * kernelArgs30[] = { &_pt_array_13, &_pt_array_14, &_pt_array_18, &_pt_array_22 };
    CUDA_KERNEL_NODE_PARAMS kernelParams30 = {0};
    kernelParams30.func = func2;
    kernelParams30.gridDimX = 2048;
    kernelParams30.gridDimY = 1;
    kernelParams30.gridDimZ = 1;
    kernelParams30.blockDimX = 128;
    kernelParams30.blockDimY = 1;
    kernelParams30.blockDimZ = 1;
    kernelParams30.kernelParams = kernelArgs30;
    kernelParams30.extra = NULL;
    kernelParams30.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_13" << std::endl;
    std::vector<CUgraphNode> nodeDependencies22;
    nodeDependencies22.push_back(_pt_memalloc_13);
    nodeDependencies22.push_back(_pt_kernel_22);
    nodeDependencies22.push_back(_pt_kernel_18);
    nodeDependencies22.push_back(_pt_kernel_14);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_13, graph, nodeDependencies22.data(), nodeDependencies22.size(), &kernelParams30));

    std::cout << "Executing on _pt_data_5" << std::endl;

    // _pt_memalloc_30

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams31;
    nodeParams31 = {};

    nodeParams31.bytesize = 8*SIZE;
    nodeParams31.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams31.poolProps.location.id = device;
    nodeParams31.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_30;
    std::cout << "Adding _pt_memalloc_30" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_30, graph, NULL, 0, &nodeParams31) );

    CUdeviceptr _pt_array_30 = nodeParams31.dptr;

    // _pt_kernel_30

    CUgraphNode _pt_kernel_30;

    void * kernelArgs31[] = { &_pt_array_30, &_pt_data_5 };
    CUDA_KERNEL_NODE_PARAMS kernelParams31 = {0};
    kernelParams31.func = func;
    kernelParams31.gridDimX = 2048;
    kernelParams31.gridDimY = 1;
    kernelParams31.gridDimZ = 1;
    kernelParams31.blockDimX = 128;
    kernelParams31.blockDimY = 1;
    kernelParams31.blockDimZ = 1;
    kernelParams31.kernelParams = kernelArgs31;
    kernelParams31.extra = NULL;
    kernelParams31.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_30" << std::endl;
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_30, graph, &_pt_memalloc_30, 1, &kernelParams31));

    // _pt_memalloc_29

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams32;
    nodeParams32 = {};

    nodeParams32.bytesize = 8*SIZE;
    nodeParams32.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams32.poolProps.location.id = device;
    nodeParams32.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_29;
    std::cout << "Adding _pt_memalloc_29" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_29, graph, &_pt_kernel_30, 1, &nodeParams32) );

    CUdeviceptr _pt_array_29 = nodeParams32.dptr;

    // _pt_kernel_29

    CUgraphNode _pt_kernel_29;

    void * kernelArgs32[] = { &_pt_array_29, &_pt_array_30 };
    CUDA_KERNEL_NODE_PARAMS kernelParams32 = {0};
    kernelParams32.func = func;
    kernelParams32.gridDimX = 2048;
    kernelParams32.gridDimY = 1;
    kernelParams32.gridDimZ = 1;
    kernelParams32.blockDimX = 128;
    kernelParams32.blockDimY = 1;
    kernelParams32.blockDimZ = 1;
    kernelParams32.kernelParams = kernelArgs32;
    kernelParams32.extra = NULL;
    kernelParams32.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_29" << std::endl;
    std::vector<CUgraphNode> nodeDependencies23;
    nodeDependencies23.push_back(_pt_memalloc_29);
    nodeDependencies23.push_back(_pt_kernel_30);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_29, graph, nodeDependencies23.data(), nodeDependencies23.size(), &kernelParams32));

    // _pt_memalloc_28

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams33;
    nodeParams33 = {};

    nodeParams33.bytesize = 8*SIZE;
    nodeParams33.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams33.poolProps.location.id = device;
    nodeParams33.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_28;
    std::cout << "Adding _pt_memalloc_28" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_28, graph, &_pt_kernel_29, 1, &nodeParams33) );

    CUdeviceptr _pt_array_28 = nodeParams33.dptr;

    // _pt_kernel_28

    CUgraphNode _pt_kernel_28;

    void * kernelArgs33[] = { &_pt_array_28, &_pt_array_29 };
    CUDA_KERNEL_NODE_PARAMS kernelParams33 = {0};
    kernelParams33.func = func;
    kernelParams33.gridDimX = 2048;
    kernelParams33.gridDimY = 1;
    kernelParams33.gridDimZ = 1;
    kernelParams33.blockDimX = 128;
    kernelParams33.blockDimY = 1;
    kernelParams33.blockDimZ = 1;
    kernelParams33.kernelParams = kernelArgs33;
    kernelParams33.extra = NULL;
    kernelParams33.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_28" << std::endl;
    std::vector<CUgraphNode> nodeDependencies24;
    nodeDependencies24.push_back(_pt_memalloc_28);
    nodeDependencies24.push_back(_pt_kernel_29);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_28, graph, nodeDependencies24.data(), nodeDependencies24.size(), &kernelParams33));

    // _pt_memalloc_27

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams34;
    nodeParams34 = {};

    nodeParams34.bytesize = 8*SIZE;
    nodeParams34.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams34.poolProps.location.id = device;
    nodeParams34.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_27;
    std::cout << "Adding _pt_memalloc_27" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_27, graph, &_pt_kernel_28, 1, &nodeParams34) );

    CUdeviceptr _pt_array_27 = nodeParams34.dptr;

    // _pt_kernel_27

    CUgraphNode _pt_kernel_27;

    void * kernelArgs34[] = { &_pt_array_27, &_pt_array_28 };
    CUDA_KERNEL_NODE_PARAMS kernelParams34 = {0};
    kernelParams34.func = func;
    kernelParams34.gridDimX = 2048;
    kernelParams34.gridDimY = 1;
    kernelParams34.gridDimZ = 1;
    kernelParams34.blockDimX = 128;
    kernelParams34.blockDimY = 1;
    kernelParams34.blockDimZ = 1;
    kernelParams34.kernelParams = kernelArgs34;
    kernelParams34.extra = NULL;
    kernelParams34.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_27" << std::endl;
    std::vector<CUgraphNode> nodeDependencies25;
    nodeDependencies25.push_back(_pt_memalloc_27);
    nodeDependencies25.push_back(_pt_kernel_28);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_27, graph, nodeDependencies25.data(), nodeDependencies25.size(), &kernelParams34));


    std::cout << "Executing on _pt_data_6" << std::endl;

    // _pt_memalloc_34

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams35;
    nodeParams35 = {};

    nodeParams35.bytesize = 8*SIZE;
    nodeParams35.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams35.poolProps.location.id = device;
    nodeParams35.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_34;
    std::cout << "Adding _pt_memalloc_34" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_34, graph, NULL, 0, &nodeParams35) );

    CUdeviceptr _pt_array_34 = nodeParams35.dptr;

    // _pt_kernel_34

    CUgraphNode _pt_kernel_34;

    void * kernelArgs36[] = { &_pt_array_34, &_pt_data_6 };
    CUDA_KERNEL_NODE_PARAMS kernelParams36 = {0};
    kernelParams36.func = func;
    kernelParams36.gridDimX = 2048;
    kernelParams36.gridDimY = 1;
    kernelParams36.gridDimZ = 1;
    kernelParams36.blockDimX = 128;
    kernelParams36.blockDimY = 1;
    kernelParams36.blockDimZ = 1;
    kernelParams36.kernelParams = kernelArgs36;
    kernelParams36.extra = NULL;
    kernelParams36.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_34" << std::endl;
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_34, graph, &_pt_memalloc_34, 1, &kernelParams36));

    // _pt_memalloc_33

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams37;
    nodeParams37= {};

    nodeParams37.bytesize = 8*SIZE;
    nodeParams37.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams37.poolProps.location.id = device;
    nodeParams37.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_33;
    std::cout << "Adding _pt_memalloc_33" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_33, graph, &_pt_kernel_34, 1, &nodeParams37) );

    CUdeviceptr _pt_array_33 = nodeParams37.dptr;

    // _pt_kernel_33

    CUgraphNode _pt_kernel_33;

    void * kernelArgs37[] = { &_pt_array_33, &_pt_array_34 };
    CUDA_KERNEL_NODE_PARAMS kernelParams37 = {0};
    kernelParams37.func = func;
    kernelParams37.gridDimX = 2048;
    kernelParams37.gridDimY = 1;
    kernelParams37.gridDimZ = 1;
    kernelParams37.blockDimX = 128;
    kernelParams37.blockDimY = 1;
    kernelParams37.blockDimZ = 1;
    kernelParams37.kernelParams = kernelArgs37;
    kernelParams37.extra = NULL;
    kernelParams37.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_33" << std::endl;
    std::vector<CUgraphNode> nodeDependencies26;
    nodeDependencies26.push_back(_pt_memalloc_33);
    nodeDependencies26.push_back(_pt_kernel_34);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_33, graph, nodeDependencies26.data(), nodeDependencies26.size(), &kernelParams37));

    // _pt_memalloc_32

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams38;
    nodeParams38 = {};

    nodeParams38.bytesize = 8*SIZE;
    nodeParams38.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams38.poolProps.location.id = device;
    nodeParams38.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_32;
    std::cout << "Adding _pt_memalloc_32" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_32, graph, &_pt_kernel_33, 1, &nodeParams38) );

    CUdeviceptr _pt_array_32 = nodeParams38.dptr;

    // _pt_kernel_32

    CUgraphNode _pt_kernel_32;

    void * kernelArgs38[] = { &_pt_array_32, &_pt_array_33 };
    CUDA_KERNEL_NODE_PARAMS kernelParams38 = {0};
    kernelParams38.func = func;
    kernelParams38.gridDimX = 2048;
    kernelParams38.gridDimY = 1;
    kernelParams38.gridDimZ = 1;
    kernelParams38.blockDimX = 128;
    kernelParams38.blockDimY = 1;
    kernelParams38.blockDimZ = 1;
    kernelParams38.kernelParams = kernelArgs38;
    kernelParams38.extra = NULL;
    kernelParams38.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_32" << std::endl;
    std::vector<CUgraphNode> nodeDependencies27;
    nodeDependencies27.push_back(_pt_memalloc_32);
    nodeDependencies27.push_back(_pt_kernel_33);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_32, graph, nodeDependencies27.data(), nodeDependencies27.size(), &kernelParams38));

    // _pt_memalloc_31

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams39;
    nodeParams39 = {};

    nodeParams39.bytesize = 8*SIZE;
    nodeParams39.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams39.poolProps.location.id = device;
    nodeParams39.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_31;
    std::cout << "Adding _pt_memalloc_31" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_31, graph, &_pt_kernel_32, 1, &nodeParams39) );

    CUdeviceptr _pt_array_31 = nodeParams39.dptr;

    // _pt_kernel_31

    CUgraphNode _pt_kernel_31;

    void * kernelArgs39[] = { &_pt_array_31, &_pt_array_32 };
    CUDA_KERNEL_NODE_PARAMS kernelParams39 = {0};
    kernelParams39.func = func;
    kernelParams39.gridDimX = 2048;
    kernelParams39.gridDimY = 1;
    kernelParams39.gridDimZ = 1;
    kernelParams39.blockDimX = 128;
    kernelParams39.blockDimY = 1;
    kernelParams39.blockDimZ = 1;
    kernelParams39.kernelParams = kernelArgs39;
    kernelParams39.extra = NULL;
    kernelParams39.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_31" << std::endl;
    std::vector<CUgraphNode> nodeDependencies28;
    nodeDependencies28.push_back(_pt_memalloc_31);
    nodeDependencies28.push_back(_pt_kernel_32);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_31, graph, nodeDependencies28.data(), nodeDependencies28.size(), &kernelParams39));

    std::cout << "Executing on _pt_data_7" << std::endl;

    // _pt_memalloc_38

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams40;
    nodeParams40 = {};

    nodeParams40.bytesize = 8*SIZE;
    nodeParams40.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams40.poolProps.location.id = device;
    nodeParams40.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_38;
    std::cout << "Adding _pt_memalloc_38" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_38, graph, NULL, 0, &nodeParams40) );

    CUdeviceptr _pt_array_38 = nodeParams40.dptr;

    // _pt_kernel_38

    CUgraphNode _pt_kernel_38;

    void * kernelArgs40[] = { &_pt_array_38, &_pt_data_7 };
    CUDA_KERNEL_NODE_PARAMS kernelParams40 = {0};
    kernelParams40.func = func;
    kernelParams40.gridDimX = 2048;
    kernelParams40.gridDimY = 1;
    kernelParams40.gridDimZ = 1;
    kernelParams40.blockDimX = 128;
    kernelParams40.blockDimY = 1;
    kernelParams40.blockDimZ = 1;
    kernelParams40.kernelParams = kernelArgs40;
    kernelParams40.extra = NULL;
    kernelParams40.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_38" << std::endl;
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_38, graph, &_pt_memalloc_38, 1, &kernelParams40));

    // _pt_memalloc_37

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams41;
    nodeParams41 = {};

    nodeParams41.bytesize = 8*SIZE;
    nodeParams41.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams41.poolProps.location.id = device;
    nodeParams41.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_37;
    std::cout << "Adding _pt_memalloc_37" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_37, graph, &_pt_kernel_38, 1, &nodeParams41) );

    CUdeviceptr _pt_array_37 = nodeParams41.dptr;

    // _pt_kernel_37

    CUgraphNode _pt_kernel_37;

    void * kernelArgs41[] = { &_pt_array_37, &_pt_array_38 };
    CUDA_KERNEL_NODE_PARAMS kernelParams41 = {0};
    kernelParams41.func = func;
    kernelParams41.gridDimX = 2048;
    kernelParams41.gridDimY = 1;
    kernelParams41.gridDimZ = 1;
    kernelParams41.blockDimX = 128;
    kernelParams41.blockDimY = 1;
    kernelParams41.blockDimZ = 1;
    kernelParams41.kernelParams = kernelArgs41;
    kernelParams41.extra = NULL;
    kernelParams41.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_37" << std::endl;
    std::vector<CUgraphNode> nodeDependencies29;
    nodeDependencies29.push_back(_pt_memalloc_37);
    nodeDependencies29.push_back(_pt_kernel_38);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_37, graph, nodeDependencies29.data(), nodeDependencies29.size(), &kernelParams41));

    // _pt_memalloc_36

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams42;
    nodeParams42= {};

    nodeParams42.bytesize = 8*SIZE;
    nodeParams42.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams42.poolProps.location.id = device;
    nodeParams42.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_36;
    std::cout << "Adding _pt_memalloc_36" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_36, graph, &_pt_kernel_37, 1, &nodeParams42) );

    CUdeviceptr _pt_array_36 = nodeParams42.dptr;

    // _pt_kernel_36

    CUgraphNode _pt_kernel_36;

    void * kernelArgs42[] = { &_pt_array_36, &_pt_array_37 };
    CUDA_KERNEL_NODE_PARAMS kernelParams42 = {0};
    kernelParams42.func = func;
    kernelParams42.gridDimX = 2048;
    kernelParams42.gridDimY = 1;
    kernelParams42.gridDimZ = 1;
    kernelParams42.blockDimX = 128;
    kernelParams42.blockDimY = 1;
    kernelParams42.blockDimZ = 1;
    kernelParams42.kernelParams = kernelArgs42;
    kernelParams42.extra = NULL;
    kernelParams42.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_36" << std::endl;
    std::vector<CUgraphNode> nodeDependencies30;
    nodeDependencies30.push_back(_pt_memalloc_36);
    nodeDependencies30.push_back(_pt_kernel_37);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_36, graph, nodeDependencies30.data(), nodeDependencies30.size(), &kernelParams42));

    // _pt_memalloc_35

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams43;
    nodeParams43 = {};

    nodeParams43.bytesize = 8*SIZE;
    nodeParams43.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams43.poolProps.location.id = device;
    nodeParams43.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_35;
    std::cout << "Adding _pt_memalloc_35" << std::endl;
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_35, graph, &_pt_kernel_36, 1, &nodeParams43) );

    CUdeviceptr _pt_array_35 = nodeParams43.dptr;

    // _pt_kernel_35

    CUgraphNode _pt_kernel_35;

    void * kernelArgs43[] = { &_pt_array_35, &_pt_array_36 };
    CUDA_KERNEL_NODE_PARAMS kernelParams43 = {0};
    kernelParams43.func = func;
    kernelParams43.gridDimX = 2048;
    kernelParams43.gridDimY = 1;
    kernelParams43.gridDimZ = 1;
    kernelParams43.blockDimX = 128;
    kernelParams43.blockDimY = 1;
    kernelParams43.blockDimZ = 1;
    kernelParams43.kernelParams = kernelArgs43;
    kernelParams43.extra = NULL;
    kernelParams43.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_35" << std::endl;
    std::vector<CUgraphNode> nodeDependencies31;
    nodeDependencies31.push_back(_pt_memalloc_35);
    nodeDependencies31.push_back(_pt_kernel_36);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_35, graph, nodeDependencies31.data(), nodeDependencies31.size(), &kernelParams43));

    // _pt_memalloc_26

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams44;
    nodeParams44 = {};

    nodeParams44.bytesize = 8*SIZE;
    nodeParams44.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams44.poolProps.location.id = device;
    nodeParams44.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc_26;
    std::cout << "Adding _pt_memalloc_26" << std::endl;
    std::vector<CUgraphNode> nodeDependencies32;
    nodeDependencies32.push_back(_pt_kernel_31);
    nodeDependencies32.push_back(_pt_kernel_27);
    nodeDependencies32.push_back(_pt_kernel_35);
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc_26, graph, nodeDependencies32.data(), nodeDependencies32.size(), &nodeParams44) );

    CUdeviceptr _pt_array_26 = nodeParams44.dptr;

    // _pt_kernel_26

    CUgraphNode _pt_kernel_26;

    void * kernelArgs44[] = { &_pt_array_26, &_pt_array_27, &_pt_array_31, &_pt_array_35 };
    CUDA_KERNEL_NODE_PARAMS kernelParams44 = {0};
    kernelParams44.func = func2;
    kernelParams44.gridDimX = 2048;
    kernelParams44.gridDimY = 1;
    kernelParams44.gridDimZ = 1;
    kernelParams44.blockDimX = 128;
    kernelParams44.blockDimY = 1;
    kernelParams44.blockDimZ = 1;
    kernelParams44.kernelParams = kernelArgs44;
    kernelParams44.extra = NULL;
    kernelParams44.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel_26" << std::endl;
    std::vector<CUgraphNode> nodeDependencies33;
    nodeDependencies33.push_back(_pt_memalloc_26);
    nodeDependencies33.push_back(_pt_kernel_27);
    nodeDependencies33.push_back(_pt_kernel_31);
    nodeDependencies33.push_back(_pt_kernel_35);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel_26, graph, nodeDependencies33.data(), nodeDependencies33.size(), &kernelParams44));

    // _pt_memalloc

    CUDA_MEM_ALLOC_NODE_PARAMS nodeParams45;
    nodeParams45 = {};

    nodeParams45.bytesize = 8*SIZE;
    nodeParams45.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    nodeParams45.poolProps.location.id = device;
    nodeParams45.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    CUgraphNode _pt_memalloc;
    std::cout << "Adding _pt_memalloc" << std::endl;
    std::vector<CUgraphNode> nodeDependencies34;
    nodeDependencies34.push_back(_pt_kernel_0);
    nodeDependencies34.push_back(_pt_kernel_26);
    nodeDependencies34.push_back(_pt_kernel_13);
    CHKERR(cuGraphAddMemAllocNode(&_pt_memalloc, graph, nodeDependencies33.data(), nodeDependencies33.size(), &nodeParams45) );

    CUdeviceptr _pt_array = nodeParams45.dptr;

    // _pt_kernel

    CUgraphNode _pt_kernel;

    void * kernelArgs45[] = { &_pt_array, &_pt_array_0, &_pt_array_13, &_pt_array_26 };
    CUDA_KERNEL_NODE_PARAMS kernelParams45 = {0};
    kernelParams45.func = func2;
    kernelParams45.gridDimX = 2048;
    kernelParams45.gridDimY = 1;
    kernelParams45.gridDimZ = 1;
    kernelParams45.blockDimX = 128;
    kernelParams45.blockDimY = 1;
    kernelParams45.blockDimZ = 1;
    kernelParams45.kernelParams = kernelArgs45;
    kernelParams45.extra = NULL;
    kernelParams45.sharedMemBytes = 0;

    std::cout << "Adding _pt_kernel" << std::endl;
    std::vector<CUgraphNode> nodeDependencies35;
    nodeDependencies35.push_back(_pt_memalloc);
    nodeDependencies35.push_back(_pt_kernel_0);
    nodeDependencies35.push_back(_pt_kernel_26);
    nodeDependencies35.push_back(_pt_kernel_13);
    CHKERR(cuGraphAddKernelNode(&_pt_kernel, graph, nodeDependencies35.data(), nodeDependencies35.size(), &kernelParams45)); 

}

int main(){

    CUresult err;
    CUdevice device = 0;
    CUcontext context = 0;

    CHKERR( cuInit(0) );
  
    CUgraph graph;
    CHKERR(cuGraphCreate(&graph, 0));

    std::cout << "Creating graph" << std::endl;
    create_graph(device, context, graph, 100000);

    CUgraphExec exec;
    std::cout << "Instantiating ExecGraph" << std::endl;
    CHKERR(cuGraphInstantiate(&exec, graph, NULL, NULL, 0));

    CHKERR(cuGraphLaunch(exec, 0));
    
    // std::cout << "warmup rounds" << std::endl;
    // for (int i=0; i < 10; i++){
        
    // }

    // size_t n_sim_round = 0;
    // size_t total_sim_time = 5.0;
    
    // while ((total_sim_time < 5.0) || (n_sim_round < 100)){
        
    //     CHKERR(cuCtxSynchronize());
    //     // Get the current time
    //     auto start = high_resolution_clock::now();
        
    //     for (int i=0; i < 100; i++){
    //         CHKERR(cuGraphLaunch(exec, 0));
    //     }

    //     CHKERR(cuCtxSynchronize());

    //     // Get the current time again
    //     auto end = high_resolution_clock::now();

    //     n_sim_round += 100;

    //     auto duration = duration_cast<seconds>(end - start);

    //     total_sim_time += duration.count();
    // }

    std::cout << "Destroying Graph" << std::endl;
    CHKERR(cuGraphDestroy(graph));

    std::cout << "Destroying ExecGraph" << std::endl;
    CHKERR(cuGraphExecDestroy(exec)); 
}

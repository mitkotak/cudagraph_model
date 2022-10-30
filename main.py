import numpy as np
import matplotlib.pyplot as plt
# from arraycontext import PyCUDAArrayContext
from grudge.array_context import PyOpenCLArrayContext
import pytato as pt
import multiprocessing
from dataclasses import dataclass
from arraycontext import PytatoCUDAGraphArrayContext
from arraycontext import (
    freeze, thaw,
    ArrayContainer, ArrayContext,
    with_container_arithmetic, dataclass_array_container)

def vector_add(a, b, c, d):
    e = a + b
    f = b + c
    g = c + d
    h = d + a
    i = e + f
    j = g + h
    k = i + j
    return k

def mem_checker():
    free_bytes, total_bytes = drv.mem_get_info()
    print(free_bytes/total_bytes)

    
def run_cudagraph_kernel(actx_class=PyOpenCLArrayContext):
    # if issubclass(actx_class, PyCUDAArrayContext):
    if issubclass(actx_class, PytatoCUDAGraphArrayContext):
        import pycuda.driver as drv
        drv.init()
        global context
        from pycuda.tools import make_default_context,DeviceMemoryPool  # noqa: E402
        context = make_default_context()
        device = context.get_device()
        actx = actx_class(allocator=DeviceMemoryPool().allocate)
    else:    
        import pyopencl as cl
        import pyopencl.tools as cl_tools
        import os
        os.environ["PYOPENCL_CTX"] = "0:1"
        cl_ctx = cl.create_some_context()
        queue = cl.CommandQueue(cl_ctx)
        actx = actx_class(
            queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
        )    

    @with_container_arithmetic(bcast_obj_array=True, rel_comparison=True)
    @dataclass_array_container
    @dataclass(frozen=True)
    class Width5:
        x1: ArrayContainer
        x2: ArrayContainer
        x3: ArrayContainer
        x4: ArrayContainer
        x5: ArrayContainer
        array_context: ArrayContext

    @with_container_arithmetic(bcast_obj_array=True, rel_comparison=True)
    @dataclass_array_container
    @dataclass(frozen=True)
    class Width6:
        x1: ArrayContainer
        x2: ArrayContainer
        x3: ArrayContainer
        x4: ArrayContainer
        x5: ArrayContainer
        x6: ArrayContainer
        array_context: ArrayContext
        
    @with_container_arithmetic(bcast_obj_array=True, rel_comparison=True)
    @dataclass_array_container
    @dataclass(frozen=True)
    class Width7:
        x1: ArrayContainer
        x2: ArrayContainer
        x3: ArrayContainer
        x4: ArrayContainer
        x5: ArrayContainer
        x6: ArrayContainer
        x7: ArrayContainer
        array_context: ArrayContext

    def width_5(data_container):
        from arraycontext import rec_map_array_container
        actx = data_container.array_context
        doubled_data_container = rec_map_array_container(lambda x: 2 * x,
                                            data_container)
        return Width5(doubled_data_container.x1,
                             doubled_data_container.x2,
                             doubled_data_container.x3,
                             doubled_data_container.x4,
                             doubled_data_container.x5,
                             actx)
    
    def width_6(data_container):
        from arraycontext import rec_map_array_container
        actx = data_container.array_context
        doubled_data_container = rec_map_array_container(lambda x: 2 * x,
                                            data_container)
        return Width6(doubled_data_container.x1,
                             doubled_data_container.x2,
                             doubled_data_container.x3,
                             doubled_data_container.x4,
                             doubled_data_container.x5,
                             doubled_data_container.x6,
                             actx)
    

    def width_7(data_container):
        from arraycontext import rec_map_array_container
        actx = data_container.array_context
        doubled_data_container = rec_map_array_container(lambda x: 2 * x,
                                            data_container)
        return Width7(doubled_data_container.x1,
                             doubled_data_container.x2,
                             doubled_data_container.x3,
                             doubled_data_container.x4,
                             doubled_data_container.x5,
                             doubled_data_container.x6,
                             doubled_data_container.x7,
                             actx)

    fig = plt.figure()
    timings_5, timings_6, timings_7 = [],[],[]
    for size in [1000,2000,3000,4000,5000]:
        width_5_compiled = actx.compile(width_5)
        width_6_compiled = actx.compile(width_6)
        width_7_compiled = actx.compile(width_7)

        input_5 = actx.from_numpy(Width5(np.zeros(shape=(size,1), dtype=np.float64) + 1,
                                np.zeros(shape=(size,1), dtype=np.float64) + 1,
                                np.zeros(shape=(size,1), dtype=np.float64) + 1,
                                np.zeros(shape=(size,1), dtype=np.float64) + 1,
                                np.zeros(shape=(size,1), dtype=np.float64) + 1,
                                actx))
        input_6 = actx.from_numpy(Width6(np.zeros(shape=(size,1), dtype=np.float64) + 1,
                                np.zeros(shape=(size,1), dtype=np.float64) + 1,
                                np.zeros(shape=(size,1), dtype=np.float64) + 1,
                                np.zeros(shape=(size,1), dtype=np.float64) + 1,
                                np.zeros(shape=(size,1), dtype=np.float64) + 1,
                                np.zeros(shape=(size,1), dtype=np.float64) + 1,
                                actx))
        input_7 = actx.from_numpy(Width7(np.zeros(shape=(size,1), dtype=np.float64) + 1,
                                        np.zeros(shape=(size,1), dtype=np.float64) + 1,
                                        np.zeros(shape=(size,1), dtype=np.float64) + 1,
                                        np.zeros(shape=(size,1), dtype=np.float64) + 1,
                                        np.zeros(shape=(size,1), dtype=np.float64) + 1,
                                        np.zeros(shape=(size,1), dtype=np.float64) + 1,
                                        np.zeros(shape=(size,1), dtype=np.float64) + 1,
                                        actx))

        from time import time
        c_time_start = time()
        for i in range(2):
            # if isinstance(actx, PytatoCUDAGraphArrayContext):
            #   a = thaw(freeze(a, actx), actx)
            # a = f_compiled(a, b, c, d)
            input_5 = width_5_compiled(input_5)

        c_time_end = time()
        print("Compiletime in secs", (c_time_end - c_time_start))
        # proc = multiprocessing.Process(target=mem_checker)
        # proc.start()
        # for i in range(2):
        #     result = f_compiled(a, b, c, d)
        #actx.freeze(result)
        #mem_checker()
        #proc.join()

        t_start = time()
        for i in range(10):
            #if isinstance(actx, PytatoCUDAGraphArrayContext):
            #    a = thaw(freeze(a, actx), actx)
            # a = f_compiled(a, b, c, d)
            input_5 = width_5_compiled(input_5)


        # a = thaw(freeze(a, actx), actx)
        t_end = time()
        print("Runtime in secs for ",size,"sized array: ", (t_end - t_start)/10)
        timings_5.append((t_end - t_start)/10)
        
        c_time_start = time()
        for i in range(2):
            # if isinstance(actx, PytatoCUDAGraphArrayContext):
            #   a = thaw(freeze(a, actx), actx)
            # a = f_compiled(a, b, c, d)
            input_6 = width_6_compiled(input_6)

        c_time_end = time()
        print("Compiletime in secs", (c_time_end - c_time_start))
        # proc = multiprocessing.Process(target=mem_checker)
        # proc.start()
        # for i in range(2):
        #     result = f_compiled(a, b, c, d)
        #actx.freeze(result)
        #mem_checker()
        #proc.join()

        t_start = time()
        for i in range(10):
            #if isinstance(actx, PytatoCUDAGraphArrayContext):
            #    a = thaw(freeze(a, actx), actx)
            # a = f_compiled(a, b, c, d)
            input_6 = width_6_compiled(input_6)


        # a = thaw(freeze(a, actx), actx)
        t_end = time()
        print("Runtime in secs for ",size,"sized array: ", (t_end - t_start)/10)
        timings_6.append((t_end - t_start)/10)

        c_time_start = time()
        for i in range(2):
            # if isinstance(actx, PytatoCUDAGraphArrayContext):
            #   a = thaw(freeze(a, actx), actx)
            # a = f_compiled(a, b, c, d)
            input_7 = width_7_compiled(input_7)

        c_time_end = time()
        print("Compiletime in secs", (c_time_end - c_time_start))
        # proc = multiprocessing.Process(target=mem_checker)
        # proc.start()
        # for i in range(2):
        #     result = f_compiled(a, b, c, d)
        #actx.freeze(result)
        #mem_checker()
        #proc.join()

        t_start = time()
        for i in range(10):
            #if isinstance(actx, PytatoCUDAGraphArrayContext):
            #    a = thaw(freeze(a, actx), actx)
            # a = f_compiled(a, b, c, d)
            input_7 = width_7_compiled(input_7)


        # a = thaw(freeze(a, actx), actx)
        t_end = time()
        print("Runtime in secs for ",size,"sized array: ", (t_end - t_start)/10)
        timings_7.append((t_end - t_start)/10)
    
    plt.plot([1000,2000,3000,4000,5000], timings_5, label="width=5")    
    plt.plot([1000,2000,3000,4000,5000], timings_6, label="width=6")
    plt.plot([1000,2000,3000,4000,5000], timings_7, label="width=7")

    from datetime import datetime
    import pytz
    filename = (datetime
                .now(pytz.timezone("America/Chicago"))
                .strftime("archive/case_%Y_%m_%d_%H%M.png"))
    plt.legend()
    plt.title(actx_class.__name__[:-len("ArrayContext")])
    plt.xlabel("Array Size")
    plt.ylabel("Runtime in secs")
    plt.savefig(filename)

    if issubclass(actx_class, PytatoCUDAGraphArrayContext):
    # if issubclass(actx_class, PyCUDAArrayContext):
        context.pop()
        context = None

        from pycuda.tools import clear_context_caches

        clear_context_caches()
        import gc
        gc.collect()

if __name__ == "__main__":
    
    run_cudagraph_kernel(actx_class=PytatoCUDAGraphArrayContext)
    # run_cudagraph_kernel()
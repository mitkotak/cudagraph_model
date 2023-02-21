import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from arraycontext import (PyCUDAArrayContext, PytatoCUDAGraphArrayContext, PytatoJAXArrayContext,
                        rec_map_array_container)
from meshmode.array_context import FusionContractorArrayContext
from pytools.obj_array import make_obj_array
import os
import pytato as pt
import multiprocessing
from dataclasses import dataclass
from arraycontext import (
    freeze, thaw,
    ArrayContainer, ArrayContext,
    with_container_arithmetic, dataclass_array_container)


def mem_checker():
    free_bytes, total_bytes = drv.mem_get_info()
    print(free_bytes/total_bytes)

def time_kernel(f_compiled):
    from time import time
    for _ in range(10):
        f_compiled()

    n_sim_round = 0
    total_sim_time = 0.

    while ((total_sim_time < 5.0)
            or (n_sim_round < 100)):

        context.synchronize()

        t_start = time()

        for _ in range(100):
            f_compiled()
        
        context.synchronize()

        t_end = time()

        n_sim_round += 100
        total_sim_time += (t_end - t_start)

    return total_sim_time / n_sim_round

def run_cudagraph_kernel(actx_class=FusionContractorArrayContext):
    import pycuda.driver as drv
    from pycuda.tools import make_default_context,DeviceMemoryPool  # noqa: E402
    drv.init()
    global context
    context = make_default_context()
    device = context.get_device()
    if issubclass(actx_class, (PyCUDAArrayContext)):
        actx = actx_class(allocator=DeviceMemoryPool().allocate)
    elif issubclass(actx_class, PytatoJAXArrayContext):
        actx = actx_class()
    else:  
        import pyopencl as cl
        import pyopencl.tools as cl_tools
        os.environ["PYOPENCL_CTX"] = "0:1"
        cl_ctx = cl.create_some_context()
        queue = cl.CommandQueue(cl_ctx)
        actx = actx_class(
            queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
        )
    cudagraph_actx = PytatoCUDAGraphArrayContext(allocator=DeviceMemoryPool().allocate)
    
    sim_time = 5.0
    #sizes = [1000,10000]
    sizes = [30000]
    widths = [3**1, 3**2, 3**3]
    # widths = np.arange(1,3)
    # heights = np.arange(1,3)
    heights = [1]
    #speedup = np.zeros(shape=(len(sizes), len(widths), len(heights), 2))
    speedup = np.zeros(shape=(len(sizes),2))
    for s_idx, size in enumerate(sizes):
        # for i in range(len(heights)):
            # height = heights[i]
            # width  = widths[i]
        height = 3
        width = 9
        # for h_idx, height in enumerate(heights):
        #     for w_idx, width in enumerate(widths):
        def run_kernel(actx):
            def f(xs):
                #xs[::3] = actx.np.where(xs[::3], xs[1::3], xs[2::3])
                xs[::3] = actx.np.where(xs[::3], xs[1::3], xs[2::3])
                xs[::3] = actx.np.where(xs[::3], xs[1::3], xs[2::3])
                #xs = actx.np.where(xs[::3], xs[1::3], xs[2::3])
                xs = actx.np.where(xs[::3], xs[1::3], xs[2::3])
                xs = actx.np.where(xs[::3], xs[1::3], xs[2::3])
                return xs
            xs = actx.thaw(actx.freeze(make_obj_array([actx.zeros(size, np.float64) for i in range(width)])))
            f_compiled = actx.compile(lambda: f(xs))
            return time_kernel(f_compiled)
        
            # def run_kernel(actx):
            #     def f(xs):
            #         for _ in range(height):
            #             #xs = actx.np.where(xs[::3], xs[1::3], xs[2::3])
            #             xs[::3] = actx.np.where(xs[::3], xs[1::3], xs[2::3])
            #         return xs
            #     xs = actx.thaw(actx.freeze(make_obj_array([actx.zeros(size, np.float64) for i in range(width)])))
            #     f_compiled = actx.compile(lambda: f(xs))
            #     return time_kernel(f_compiled)
            
            # def run_kernel(actx):
            #     def f(xs):
            #         for _ in range(height):
            #             xs = xs + 1
            #         return xs
            #     xs = actx.thaw(actx.freeze(make_obj_array([actx.zeros(size,  np.float64)+1 for _ in range(width)])))
            #     f_compiled = actx.compile(lambda: f(xs))
            #     return time_kernel(f_compiled)
            
        final_cudagraph_time = run_kernel(cudagraph_actx)
        final_base_time = run_kernel(actx)
        
        print("Speedup for ",size,"sized array, cudagraph_time ", final_cudagraph_time, " , base_time ", final_base_time, " :", (final_base_time/final_cudagraph_time))
        speedup[s_idx,0] = final_base_time
        speedup[s_idx,1] = final_cudagraph_time

    from datetime import datetime
    import pytz
    os.mkdir(datetime
            .now(pytz.timezone("America/Chicago"))
            .strftime("archive/%Y_%m_%d_%H%M"))
    filename = (datetime
                .now(pytz.timezone("America/Chicago"))
                .strftime("archive/%Y_%m_%d_%H%M/speedup_%Y_%m_%d_%H%M.npz"))
    np.savez(filename, widths=widths, heights=heights, sizes=sizes, speedup=speedup)
    print("Saved ",filename)

    context.pop()
    context = None

    from pycuda.tools import clear_context_caches

    clear_context_caches()
    import gc
    gc.collect()

if __name__ == "__main__":

    run_cudagraph_kernel(actx_class=PyCUDAArrayContext)
    # run_cudagraph_kernel(actx_class=PytatoJAXArrayContext)
    # run_cudagraph_kernel()

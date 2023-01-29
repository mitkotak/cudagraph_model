import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from arraycontext import (PyCUDAArrayContext, PytatoCUDAGraphArrayContext, PytatoJAXArrayContext)
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
    for _ in range(2):
        f_compiled()

    n_sim_round = 0
    total_sim_time = 0.

    while ((total_sim_time < 5.0)
            or (n_sim_round < 100)):

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
    drv.init()
    global context
    from pycuda.tools import make_default_context,DeviceMemoryPool  # noqa: E402
    context = make_default_context()
    device = context.get_device()
    if issubclass(actx_class, (PytatoCUDAGraphArrayContext, PyCUDAArrayContext)):
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

    fig, ax = plt.subplots(2)
    sim_time = 5.0
    widths = np.arange(50)
    heights = np.arange(50)
    W,H = np.meshgrid(widths, heights)
    speedup = np.zeros(W.shape)
    for size in [1000]:
        for h_idx, height in enumerate(heights):
            for w_idx, width in enumerate(widths):
                def run_kernel(actx):
                    def f(xs, ys):
                        assert isinstance(xs, np.ndarray)
                        assert isinstance(ys, np.ndarray)
                        for _ in range(height):
                            xs = make_obj_array([xs[i] + ys[i] for i in range(len(xs))])
                        return xs

                    x = make_obj_array([actx.zeros(size,  np.float64)+1
                                        for _ in range(width)])
                    y = make_obj_array([actx.zeros(size,  np.float64)+1
                                        for _ in range(width)])
                    f_compiled = actx.compile(lambda : f(actx.thaw(actx.freeze(x)), actx.thaw(actx.freeze(y)))) 
                    
                    return time_kernel(f_compiled)

                final_den_time = run_kernel(actx)
                final_num_time = run_kernel(cudagraph_actx)

                print("Speedup for ",size,"sized array, height ", height, " , width ", width, " :", (final_num_time/final_den_time))
                speedup[h_idx,w_idx] = final_num_time/final_den_time
    
    from datetime import datetime
    import pytz
    os.mkdir(datetime
            .now(pytz.timezone("America/Chicago"))
            .strftime("archive/%Y_%m_%d_%H%M"))
    filename = (datetime
                .now(pytz.timezone("America/Chicago"))
                .strftime("archive/%Y_%m_%d_%H%M/speedup_%Y_%m_%d_%H%M.npz"))
    np.savez(filename, W=W, H=H, speedup=speedup)

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

"""
Perform UMPA (and especially the unwarping procedure) with multiprocessing
and use of shared memory.
There are non-parallelized processes for loading data and sending it to
shared memory, then we have a pool of workers for unwarping and UMPA
which send processed images to a Queue, and finally a single process for
saving data from the Queue to disk.

Putting UMPA in the parallelized process is kind of unnecessary (bad?),
because it uses multiple cores on its own, so that should be changed.
"""

from multiprocessing import Process, Queue, shared_memory, Lock, Pool
import time
import sys
import numpy as np
import imageio
import os
import UMPA
sys.path.insert(0, '/data/speckle/20201210_starfish/scripts/')
import unwarp

def read_log(log):
    '''
    Function that reads and splits the log file in
    pr: prefix
    sX: hexapod X coordinate
    sZ: hezapod Z coordinate
    dX: sandpaper holder X coordinate
    dZ: sandpaper holder Z coordinate
    rt: rotator angle
    I: current
    t: time in (y,M,d,h,m,s) format
    '''
    pr, sX, sZ, dX, dZ, rt, I, t = ([] for i in range(8))
    with open(log) as f:
        for line in f:
            entries = line.split(',')
            pr.append(entries[0])
            sX.append(float(entries[1]))
            sZ.append(float(entries[2]))
            dX.append(float(entries[3]))
            dZ.append(float(entries[4]))
            rt.append(float(entries[5]))
            I.append(float(entries[6]))
            t.append(entries[7])
    return {'pr': pr, 'sX': sX, 'sZ': sZ, 'dX': dX,'dZ': dZ, 'rt': rt, 'I': I, 't': t}


def get_image(path, filename, do_unwarp=True, dtype=float):
    img = imageio.imread(os.path.join(path, filename + '.tif')).astype(dtype)
    if do_unwarp:
        img = unwarp.unwarp(img)
    return img
    

def fill_list(names, prefix_list, path, do_unwarp=True):
    '''
    Read images from a folder 'path' from a prefix list and fill them in a
    single img list if containing 'name'.
    Updated version: load first image, create 3D array of predicted size,
    write remaining images in that array.
    '''
    if type(names) is str:
        names = [names]
    # Find all files `prefix_list` matching each of the entries in `names`:
    matches = [[pr for pr in prefix_list if (name in pr)] for name in names]
    # Check if same number of matches for each name:
    assert np.all([len(m)==len(matches[0]) for m in matches[1:]])
    # todo: should `matches` be sorted?
    img0 = get_image(path, matches[0][0], do_unwarp=do_unwarp)
    out_array = np.zeros((len(names), len(matches), *img0.shape))
    out_array[0,0] = img0
    start = True  # only True for the first image
    for i, matches_name in enumerate(matches):
        for j, filename in enumerate(matches_name):
            if start:
                # create the output array in the first iteration
                img0 = get_image(path, filename, do_unwarp=do_unwarp)
                out_array = np.zeros((len(matches), len(matches[0]), *img0.shape))
                out_array[i,j] = img0
                start = False
            else:
                out_array[i,j] = get_image(path, filename, do_unwarp=do_unwarp)
    return out_array

  
def share(local_arr):
    """
    Produce a shared memory object from an array.
    Returns dictionary with shared_memory object (key "mem"), shape of the array
    (key "shape"), and data type of the array (key "dtype").
    """
    shm = shared_memory.SharedMemory(create=True, size=local_arr.nbytes)
    shared_arr = np.ndarray(local_arr.shape, dtype=local_arr.dtype, buffer=shm.buf)
    shared_arr[:] = local_arr[:]
    return dict(mem=shm, shape=local_arr.shape, dtype=local_arr.dtype)


def unwarp_and_umpa(in_queue, out_queue, spiral_ref, flat, dark, ref_nums, window, max_shift, lock):
    """
    Read elements from `in_queue`, do unwarping, do flat correction with nearest flat, perform UMPA,
    and write UMPA output to `out_queue`.

    in_queue, out_queue: Queue objects
    dark, spiral_ref, flat: shared memory dicts, as returned by share()
    
    Read sample projection from in_queue,
    perform flatfield correction with data from shared memory,
    write UMPA output to out_queue.
    """
    #print('started unwarp_and_umpa')
    while True:
        try:
            msg = in_queue.get()
        except OSError:
            print('Queue unavailable, worker shutting down')
            break
        if type(msg) is str and msg == 'DONE':
            print('received DONE in unwarp_and_umpa')
            out_queue.put('DONE') #pass the sentinel to out_queue
            break
        else:
            #print('found input!')
            proj_num, proj = msg
            
            # Do unwarp: slowest step by far!
            for frame in range(proj.shape[0]):
                #print('started an unwarp')
                proj[frame] = unwarp.unwarp(proj[frame])

            # Get nearest reference
            refnum = np.argmin(abs(float(proj_num) - np.array(ref_nums)))  # todo: get ref_nums in here
            
            # Do flat correction
            print('do flat corr')
            # reconstruct the numpy arrays from shared memory:
            spiral_ref_arr = np.ndarray(spiral_ref['shape'], dtype=spiral_ref['dtype'], buffer=spiral_ref['mem'].buf)
            flat_arr = np.ndarray(flat['shape'], dtype=flat['dtype'], buffer=flat['mem'].buf)
            dark_arr = np.ndarray(dark['shape'], dtype=dark['dtype'], buffer=dark['mem'].buf)
            
            #print(proj.shape, dark_arr.shape, flat_arr.shape)
            
            sam = (proj - dark_arr) / flat_arr[refnum]
            ref = spiral_ref_arr[refnum]  # (spiral_ref_arr[refnum] - dark_arr) / flat_arr[refnum]
            
            # Do UMPA
            print('do umpa')
            U = UMPA.model.UMPAModelDF(sam, ref, window_size=window, max_shift=max_shift)
            res = U.match()
            
            # Write result to out_queue
            out_queue.put([proj_num, res])
            
            # Print message to console
            lock.acquire()
            try:
                print("Processed proj# %03d" % proj_num)
            finally:
                lock.release()


def save(out_queue, num_procs, save_path, save_filename, lock):
    """
    Read elements from `out_queue` and write to file.
    Queue elements are tuples (proj_num, res), where proj_num is the projection
    number and `res` is the output dictionary from UMPAModelxxx.match()

    Stops reading when the string "DONE" has been received `num_procs` times
    (since each worker process will send this string when terminating).
    """
    num_finished = 0
    while True:
        msg = out_queue.get()
        # each of the worker processes (running unwarp_and_umpa) must
        # return 'DONE' for this process to shut down!
        #print('num_finished =', num_finished, 'msg=', msg)
        if type(msg) is str and msg == 'DONE':
            num_finished += 1
            print('num_finished =', num_finished)
            if num_finished == num_procs:
                break
        else:
            proj_num, res = msg
            np.savez(os.path.join(save_path, save_filename % proj_num), **res)
            lock.acquire()when
            try:
                print("Saved proj# %03d" % proj_num)
            finally:
                lock.release()


if __name__=='__main__':
    
    # Data paths:
    # ===========
    base_path = '/data/speckle/20201210_starfish/rawdata/'
    base_save_path = '/data/speckle/20201210_starfish/processed/'
    scan_name = '3Dstar_UMPA_30keV_pos0'
    
    path = os.path.join(base_path, scan_name)
    save_path = os.path.join(base_save_path, scan_name, 'test_parallel_fabio')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_filename = 'test_%03d'
    
    log_file = os.path.join(path, '3Dstar_UMPA_30keV_pos0.log')
    log_list = read_log(log_file)
    prefix_list = log_list['pr']


    # Parameters for refs:
    # ====================
    nprojs = 721  # Total number of projections
    projs_to_process = np.arange(721) # which projections to actually process?
    nrefs = 60 # Refs and flats are taken every <nrefs> projections
    ref_nums = np.arange(0, nprojs, nrefs)  # Indices of reference scans

    
    # Load refs:
    # ==========

    # Get the names:
    spiral_ref_names = ['spiral_reference_%03d' % r for r in ref_nums]
    flat_names       = ['flat_field_%03d' % r for r in ref_nums]
    #print(ref_nums, spiral_ref_names, flat_names)
    
    cont = dict(end="", flush="True") # params for print() to allow appending something.

    # Load the data:
    print("Loading dark frames...        ", **cont)
    dark = fill_list('dark_frame', prefix_list, path, do_unwarp=True)
    dark = np.mean(dark, axis=1)
    print("done")
    print("Loading spiral_ref...         ", **cont)
    spiral_ref = fill_list(spiral_ref_names, prefix_list, path, do_unwarp=True)
    print("done")
    print("Loading flat...               ", **cont)
    flat       = fill_list(flat_names, prefix_list, path, do_unwarp=True)
    print("done")

    # Combine flats, dark, and spiral_ref for corrected flats
    flat          = np.mean(flat, axis=1) - dark
    flat[flat<=0] = 1

    print("Correcting flats...           ", **cont)
    spiral_ref    = (spiral_ref - dark[:,None,:,:]) / flat[:,None,:,:]
    print("done")

    #sys.exit()
    # we need spiral_reference, dark, and flat in shared mem.

    print("Loading data to shared mem... ", **cont)
    dark_sh = share(dark)
    spiral_ref_sh = share(spiral_ref)
    flat_sh = share(flat)
    print("done")
    
    try:  # this try-except bracket is to ensure the shared memory is released even if something crashes
        lock = Lock()
        in_queue = Queue()  # input queue, containing intensity data
        out_queue = Queue()  # output queue, containing UMPA data
        num_procs = 5  # how many worker processes?
        window, max_shift = 3, 5  # UMPA parameters
        # Pool of workers to do UMPA: each worker runs unwarp_and_umpa(),
        # where data is read from in_queue and results are written to out_queue.
        pool = Pool(num_procs, unwarp_and_umpa, (in_queue, out_queue, spiral_ref_sh, flat_sh, dark_sh, ref_nums, window, max_shift, lock))

        # Process to save output in out_queue to file:
        saver = Process(target=save, args=(out_queue, num_procs, save_path, save_filename, lock))
        saver.daemon = True  # do I need this?
        saver.start() 

        # Push the input to the queue:
        for proj_num in projs_to_process:
            # load data from disk (without unwarp!)
            proj = fill_list('spiral_sample_%03d' % proj_num, prefix_list, path, do_unwarp=False).squeeze()
            in_queue.put([proj_num, proj])
            lock.acquire()
            try:
                print("Loaded proj# %03d" % proj_num)
            finally:
                lock.release()

        # Put a bunch of "DONE"s at the end so the worker processes know they can quit
        for j in range(num_procs):
            in_queue.put('DONE')

        pool.close()  # allow no more tasks to be submitted to pool
        pool.join()  # wait for all workers to exit

        in_queue.close()
        in_queue.join_thread()

        saver.join()  # wait until saver is done
        
        out_queue.close()
        out_queue.join_thread()
    except Exception as e:
        print(e)
    finally:
        # close the shared memory objects:
        for shm in [spiral_ref_sh, flat_sh, dark_sh]:
            shm['mem'].close()
            shm['mem'].unlink()


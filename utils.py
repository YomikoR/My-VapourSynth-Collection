import os
import multiprocessing as mp
import runpy
import inspect
import html
import vapoursynth as vs

__all__ = [
    'file_path',
    'u2a',
    'a2u',
    'get_param',
    'read_vpy',
    'mp_worker'
]

def file_path(vol, num, path, patch=''):
    return path + f'{vol}{patch}{os.path.sep}BDMV{os.path.sep}STREAM{os.path.sep}{num:05}.m2ts'

# UTF-8 -> ASCII
def u2a(stu):
    return stu.encode('ascii', 'xmlcharrefreplace')

# ASCII -> UTF-8
def a2u(sta):
    return html.unescape(sta.decode('utf8', 'xmlcharrefreplace'))

# Get global variable in VPY
def get_param(key, decode=True):
    key_b = inspect.currentframe().f_back.f_globals[f'{key}']
    return a2u(key_b) if decode else key_b

# Read VPY from script
# Example for using params
#   In python:
#       params = dict(vol='1', num='3', ext='明')
#       output_idx = 2
#   In vpy:
#       globals()['vol'].decode('utf-8') # gets '1'
#       num.decode('utf-8')              # gets '3'
#       from ... import get_param
#       get_param('ext')                 # gets '明' which is non-ASCII
#       clip.set_output(2)               # output with index 2
def read_vpy(script, params=dict(), output_idx=0):
    params_ascii = dict()
    for i, k in params.items():
        params_ascii[i] = u2a(k)
    runpy.run_path(script, params_ascii, '__vapoursynth__')
    return vs.get_output(output_idx)

# The multiprocessing wrapper to recycle hardware resources (esp. GPU memory)
def mp_worker(func):
    p = mp.Pool(processes=1)
    data = p.apply(func)
    p.close()
    return data

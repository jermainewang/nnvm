flags = [
'-std=c++11',
'-DMSHADOW_FORCE_STREAM',
'-Wall',
'-O3',
'-I/home/jermaine/git-repo/dmlc/mxnet/nnvm/include',
'-Iinclude',
'-msse3',
'-funroll-loops',
'-Wno-unused-parameter',
'-Wno-unknown-pragmas',
'-DMSHADOW_USE_CUDA=0',
'-DMSHADOW_USE_CBLAS=1',
'-DMSHADOW_USE_MKL=0',
'-DMSHADOW_RABIT_PS=0',
'-DMSHADOW_DIST_PS=0',
'-DMSDHADOW_USE_PASCAL=0',
'-DMXNET_USE_OPENCV=1',
'-DDMLC_USE_CXX11=1',
'-fopenmp',
'-DMXNET_USE_NVRTC=0',
]

def FlagsForFile(filename):
    return { 'flags' : flags, 'do_cache': True }

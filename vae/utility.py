import sys
from collections import OrderedDict

param_type = OrderedDict({
    'drop_rate':float,
    'epoch':int,
    'model':int,
    'dense_size':int,
    'batch_size':int,
    'name':str
    })

def read_command_line():

    cmd = {}
    if len(sys.argv) > 1:
        argv = sys.argv[1:]
        for arg in argv:
            k,v = arg.split('=')
            t = param_type[k]
            if t is int:
                cmd[k] = int(float(v))
            else:
                cmd[k] = t(v) 
    return cmd

def make_file_name(param):
    p = ["dr=%.2f"%param['drop_rate'],"epoch=%i"%param['epoch'],"mod=%i"%param['model'],"i"%param['dense_size'],"i"%param['batch_size']]
    return "VAE_" + "_".join(p) + ".h5"
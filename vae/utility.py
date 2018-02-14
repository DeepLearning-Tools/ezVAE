import sys

def read_command_line():
    param_type = {
    'drop_rate':float,
    'epoch':int,
    'model':int,
    'dense_size':int,
    'batch_size':int,
    'name':str
    }
    
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





import sys


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def llprint(message):

    sys.stdout.write(message)
    sys.stdout.flush()
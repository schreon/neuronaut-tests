'''
Created on Jul 7, 2014

@author: schreon
'''
import cPickle
import logging
import os

import numpy


log = logging.getLogger("abalone dataset")

def _download_file(url, file_name):
    import urllib2
    u = urllib2.urlopen(url)
    dirname = os.path.dirname(file_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    log.info("Downloading: %s Bytes: %s" % (file_name, file_size))

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        log.info(status)

    f.close()


def _load(path, file_name):
    file_path = os.path.join(path, file_name)
    if not os.path.exists(file_path):
        _download_file("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/"+str(file_name), file_path)


def get_patterns():
    abalone_pik = os.path.join('data', 'abalone.pik')
    if not os.path.exists(abalone_pik):
        log.info("creating pickle file")
        # make sure the abalone files are present
        _load(os.path.join('data'), 'abalone.data')
        
        patterns = []
        sexmap = {'M': 0.0, 'I' : 0.5, 'F': 1.0}
        filename = os.path.join('data', 'abalone.data')
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                values = line.split(",")
                # convert character to float representation
                values[0] = sexmap[values[0]]
                patterns.append(values)
        
        patterns = numpy.array(patterns, dtype=numpy.float32)
        
        minimum = patterns.min(axis=0)
        patterns -= minimum
        maximum = patterns.max(axis=0)
        patterns /= maximum
                
        with open(abalone_pik, "wb") as f:            
            cPickle.dump((patterns, minimum, maximum), f, protocol=-1)
    else:
        log.info("loading existing pickle file")
        with open(abalone_pik, "rb") as f:
            (patterns, minimum, maximum) = cPickle.load(f)
        
    inputs = patterns[:, :-1].astype(numpy.float32)
    targets = patterns[:, -1:].astype(numpy.float32)

    n = int(0.5*inputs.shape[0])

    return inputs[:n], targets[:n], inputs[n:], targets[n:], minimum[-1], maximum[-1]
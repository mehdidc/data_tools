import random
from clize import run
import glob
from subprocess import call
import os

def split(folder, test_size=0.5):
        pattern = '{}/**/**/*'.format(folder)
        filenames = list(glob.glob(pattern))
        
        random.shuffle(filenames)
        nb_test = int(test_size * len(filenames))
        nb_train  = len(filenames) - nb_test

        train = filenames[0:nb_train] 
        test = filenames[nb_train:]

        os.mkdir('train')
        o = 'train'
        for t in train:
            call('cp {} {}'.format(t, o), shell=True)
        os.mkdir('test')
        o = 'test'
        for t in test:
            call('cp {} {}'.format(t, o), shell=True)



run(split)




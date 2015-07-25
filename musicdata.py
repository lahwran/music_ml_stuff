import os
import py

datadir = os.path.expanduser("~/.musicdata")
try:
    os.mkdir(datadir)
except:
    # open up your mouth and EAT IT
    pass

datadir = py.path.local(datadir)

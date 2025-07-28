"""
This is where everything comes together. You can preprocess (create labels) for the training (North California) and 
test (Hawaii, Ridgecrest, Yellowstone) datasets, train the models, and perform evaluation on the test data. All 
functions are connected to command line programs. Check the readme for help on how to run these programs. The 
functions can also be imported from here and run in notebooks for development purposes. Recommended to run commands 
in terminal in production.
"""


from seisnet.pipelines.cluster_events import *
from seisnet.pipelines.cross_corr import *
from seisnet.pipelines.evaluation import *
from seisnet.pipelines.preprocessing import *
from seisnet.pipelines.sparse_sampler import *
from seisnet.pipelines.training import *

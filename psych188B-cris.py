# there is a 6-8 second lag for BOLD signal

import os
import numpy as n

# data.pymvpa.org/datasets/haxby2001
def load_haxby_data(datapath, sub, mask=None):
    # input arguments:
    # datapath (string): path to the root directory
    # sub (string): subject ID (e.g. subj1, subj2, etc)
    # output:
    # maskeddata (numpy array): samples x voxels data matrix
    # fmrilabel (pandas dataframe): length samples
    import nibabel as nib
    import pandas as pd
    import numpy as np

    fmriobj = nib.load(os.path.join(datapath, sub, 'train.nii.gz'))
    fmridata, fmriheader = fmriobj.get_data(), fmriobj.header
    fmridata = np.rollaxis(fmridata, -1)
    # shift last axis to the first
    fmrilabel = pd.read_table(os.path.join(datapath, sub, 'labels.txt'), delim_whitespace=True)
    if mask is not None:
        maskobj = nib.load(os.path.join(datapath, sub, mask + '.nii.gz'))
        maskdata, maskheader = maskobj.get_data(), maskobj.header
        maskeddata = fmridata[:, maskdata > 0]  # timepoints axis 0, voxels axis 1
        # need to figure out how to mask features back to original geometry
        print maskeddata.shape
    else:
        maskeddata = fmridata

    return maskeddata, fmrilabel[fmrilabel.chunks != 11]  # not loading the testing run that we've set aside

# Setup working directory & load files
cwd = os.getcwd()+'/haxby2001-188B'
files = os.listdir(cwd)
data = np.array([])
for f in files:
	if f.startswith('subj'):
		data = np.array([data, [load_haxby_data(cwd, f, 'mask4_vt')]])



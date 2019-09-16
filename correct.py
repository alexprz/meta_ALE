import os
from os import listdir
from os.path import isfile, join
import pickle
import nimare
import numpy as np

from nilearn import plotting
import nibabel as nib

import matplotlib
matplotlib.use('MacOsx')

from matplotlib import pyplot as plt

path_res = os.path.abspath('save/res_meta/KDA_res')
path_obj = os.path.abspath('save/res_meta/KDA_obj')
path_t = os.path.abspath('save/res_meta/KDA_t')

with open(path_res, 'rb') as file:
    res = pickle.load(file)

with open(path_obj, 'rb') as file:
    obj = pickle.load(file)

# img = res.get_map('of')

# plotting.plot_stat_map(img)
# plt.show()

# KDA = nimare.meta.cbma.mkda.KDA()
# KDA.dat
arr_p = obj._fwe_correct_permutation(res, n_iters=10000)['logp_level-voxel']

arr_p = np.exp(-arr_p)

arr_t = res.get_map('of', return_type='array')
# print(arr_t)
arr_t[arr_p > 0.05] = 0

mask = obj.mask
masker = nimare.utils.get_masker(mask)

img_t = masker.inverse_transform(arr_t)
# print(arr_t)

# img_t = nib.Nifti1Image(arr_t, res.get_map('of').affine)
plotting.plot_stat_map(img_t)
plt.show()

with open(path_t, 'wb') as file:
    pickle.dump(img_t, file)


## MKDA

path_res = os.path.abspath('save/res_meta/MKDA_res')
path_obj = os.path.abspath('save/res_meta/MKDA_obj')
path_t = os.path.abspath('save/res_meta/MKDA_t')

with open(path_res, 'rb') as file:
    res = pickle.load(file)

with open(path_obj, 'rb') as file:
    obj = pickle.load(file)

# img = res.get_map('of')

# plotting.plot_stat_map(img)
# plt.show()

# KDA = nimare.meta.cbma.mkda.KDA()
# KDA.dat
arr_p = obj._fwe_correct_permutation(res, n_iters=10000)['logp_level-voxel']

arr_p = np.exp(-arr_p)

arr_t = res.get_map('of', return_type='array')
# print(arr_t)
arr_t[arr_p > 0.05] = 0

mask = obj.mask
masker = nimare.utils.get_masker(mask)

img_t = masker.inverse_transform(arr_t)
# print(arr_t)

# img_t = nib.Nifti1Image(arr_t, res.get_map('of').affine)
plotting.plot_stat_map(img_t)
plt.show()

with open(path_t, 'wb') as file:
    pickle.dump(img_t, file)

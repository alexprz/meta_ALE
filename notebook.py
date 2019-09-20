import os
import nilearn
import numpy as np
from nilearn import masking, plotting
from nipy.labs.statistical_mapping import get_3d_peaks
import nibabel as nib
import scipy

import matplotlib
matplotlib.use('MacOsx')

from matplotlib import pyplot as plt

o_dir = 'data-narps/proc/'  # Processed studies' directory
hyp_file = 'hypo1_unthresh.nii.gz'  # File to look for in each study's folder

template = nilearn.datasets.load_mni152_template()
affine = template.affine
gray_mask = masking.compute_gray_matter_mask(template)


def retrieve_imgs(dir_path, filename):
    """
    Return a dict of path to images.

    This function is specific to the studied dataset structure.

    Args:
        dir_path (str): Path to the folder containing the studies folders

    Returns:
        (list): List of absolute paths (string) to the found images.

    """
    # List of folders contained in dir_path folder
    Dir = [f'{dir_path}{dir}' for dir in next(os.walk(dir_path))[1]]
    try:
        # On some OS the root dict is also in the list, must be removed
        Dir.remove(dir_path)
    except ValueError:
        pass

    path_dict = dict()
    for dir in Dir:
        name = os.path.basename(os.path.normpath(dir))  # Extract name of study
        path = os.path.abspath(dir)  # Turn into absolute paths
        path_dict[name] = f'{path}/{filename}'

    return path_dict

def get_activations(path, threshold):
    """
    Retrieve the xyz activation coordinates from an image.

    Args:
        path (string or Nifti1Image): Path to or object of a
            nibabel.Nifti1Image from which to extract coordinates.
        threshold (float): Same as the extract_from_paths function.

    Returns:
        (tuple): Size 3 tuple of lists storing respectively the X, Y and
            Z coordinates

    """
    I, J, K = [], [], []
    try:
        img = nilearn.image.load_img(path)
    except ValueError:  # File path not found
        print(f'File {path} not found. Ignored.')
        return None

    if np.isnan(img.get_fdata()).any():
        print(f'Img {path} contains Nan. Ignored.')
        return None

    img = nilearn.image.resample_to_img(img, template)

    peaks = get_3d_peaks(img, mask=gray_mask, threshold=threshold)

    if not peaks:
        return I, J, K

    for peak in peaks:
        I.append(peak['ijk'][0])
        J.append(peak['ijk'][1])
        K.append(peak['ijk'][2])

    del peaks
    return np.array(I).astype(int), np.array(J).astype(int), np.array(K).astype(int)


img_paths = retrieve_imgs(o_dir, hyp_file)

binary_imgs = []
for I, J, K in [get_activations(path, 1.96) for path in list(img_paths.values())[:2]]:
    arr = np.zeros(template.shape)
    arr[I, J, K] = 1
    # arr = scipy.ndimage.gaussian_filter(arr, sigma=2)
    # print(arr[arr > 0])
    img = nib.Nifti1Image(arr, affine)
    binary_imgs.append(img)

# binary_imgs = [get_activations(path, 1.96) for path in list(img_paths.values())[:2]]
print(binary_imgs)

for img in binary_imgs:
    plotting.plot_glass_brain(img, threshold=0.002)
plt.show()

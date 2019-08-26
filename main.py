"""Perform data extraction, run ALE and plot results."""
import extract
import nimare
import matplotlib
import os
import nibabel as nib
from nilearn import plotting
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import copy

# Set backend for matplotlib
load_dotenv()
if 'MPL_BACKEND' in os.environ:
    matplotlib.use(os.environ['MPL_BACKEND'])
else:
    matplotlib.use('TkAgg')


def run_ALE(ds_dict):
    """Extract and run ALE on specified data."""
    # Performing ALE
    ds = nimare.dataset.Dataset(ds_dict)
    ALE = nimare.meta.cbma.ale.ALE()
    res = ALE.fit(ds)

    img_ale = res.get_map('ale')
    img_p = res.get_map('p')
    img_z = res.get_map('z')

    return img_ale, img_p, img_z


def fdr_threshold(img_list, img_p, q=0.05):
    """Compute FDR and threshold same-sized images."""
    arr_list = [copy.copy(img.get_fdata()) for img in img_list]
    arr_p = img_p.get_fdata()
    aff = img_p.affine

    fdr = nimare.stats.fdr(arr_p.ravel(), q=q)

    for arr in arr_list:
        arr[arr_p > fdr] = 0

    res_list = [nib.Nifti1Image(arr, aff) for arr in arr_list]

    return res_list


if __name__ == '__main__':
    # Folder and file names
    data_dir = 'data-narps/orig/'  # Data folder
    hyp_file = 'hypo1_unthresh.nii.gz'  # Hypothesis name

    # Extracting coordinates from data
    ds_dict = extract.extract(data_dir, hyp_file, threshold=1.96, load=True)

    img_ale, img_p, img_z = run_ALE(ds_dict)
    img_ale_t, img_p_t, img_z_t = fdr_threshold([img_ale, img_p, img_z], img_p)

    plotting.plot_stat_map(img_ale, title='ALE')
    plotting.plot_stat_map(img_p, title='p')
    plotting.plot_stat_map(img_z, title='z')
    plotting.plot_stat_map(img_ale_t, title='ALE thresholded')
    plotting.plot_stat_map(img_z_t, title='z thresholded')
    plotting.plot_stat_map(img_p_t, title='p thresholded')
    plt.show()

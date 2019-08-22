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


if __name__ == '__main__':
    # Folder and file names
    data_dir = 'data-narps/orig/'  # Data folder
    hyp_file = 'hypo1_unthresh.nii.gz'  # Hypothesis name

    # Extracting coordinates from data
    ds_dict = extract.extract(data_dir, hyp_file, threshold=.95, load=False)

    # Performing ALE
    ds = nimare.dataset.Dataset(ds_dict)
    ALE = nimare.meta.cbma.ale.ALE()
    res = ALE.fit(ds)


    # Plotting results of meta analysis
    img_ale = res.get_map('ale')
    img_p = res.get_map('p')
    img_z = res.get_map('z')

    arr_ale = img_ale.get_fdata()
    arr_p = img_p.get_fdata()
    arr_z = img_z.get_fdata()

    fdr = nimare.stats.fdr(arr_p.ravel(), q=0.05)
    print(f'FDR : {fdr}')

    arr_ale_thresholded = copy.copy(arr_ale)
    arr_z_thresholded = copy.copy(arr_z)
    arr_p_thresholded = copy.copy(arr_p)

    arr_ale_thresholded[arr_p > fdr] = 0
    arr_z_thresholded[arr_p > fdr] = 0
    arr_p_thresholded[arr_p > fdr] = 0

    img_ale_thresholded = nib.Nifti1Image(arr_ale_thresholded, img_ale.affine)
    img_z_thresholded = nib.Nifti1Image(arr_z_thresholded, img_z.affine)
    img_p_thresholded = nib.Nifti1Image(arr_p_thresholded, img_p.affine)

    plotting.plot_stat_map(img_ale, title='ALE')
    plotting.plot_stat_map(img_p, title='p')
    plotting.plot_stat_map(img_z, title='z')
    plotting.plot_stat_map(img_ale_thresholded, title='ALE thresholded')
    plotting.plot_stat_map(img_z_thresholded, title='z thresholded')
    plotting.plot_stat_map(img_p_thresholded, title='p thresholded')
    plt.show()

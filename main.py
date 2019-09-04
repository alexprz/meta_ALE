"""Perform data extraction, run ALE and plot results."""
from extract import extract_from_paths, process
import nimare
from nimare.dataset import Dataset
import matplotlib
import os
import nibabel as nib
from nilearn import plotting
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import copy
import nilearn

# Set backend for matplotlib
load_dotenv()
if 'MPL_BACKEND' in os.environ:
    matplotlib.use(os.environ['MPL_BACKEND'])
else:
    matplotlib.use('TkAgg')


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

    # Extract extension
    # base, filename = ntpath.split(path)
    file, ext = filename.split('.', 1)

    paths = dict()
    for dir in Dir:
        name = os.path.basename(os.path.normpath(dir))  # Extract name of study
        path = os.path.abspath(dir)  # Turn into absolute paths
        # paths[name] = f'{path}/{filename}'
        paths[name] = {
            'z': f'{path}/{filename}',
            'con': f'{path}/{file}_con.{ext}',
            'se': f'{path}/{file}_se.{ext}'
        }

        # Filter to keep only existing files
        # paths[name] = dict(filter(os.path.isfile, paths.items()))
        paths[name] = {k: v for
                       k, v in paths[name].items() if os.path.isfile(v)}

    # Remove empty dict values
    paths = {k: v for k, v in paths.items() if v}
        # for file_path in [f'{path}/{filename}',
        #             f'{path}/{file}_con.{ext}',
        #             f'{path}/{file}_se.{ext}']:

            # if os.path.isfile(file_path):
            #     paths[name][''] = file_path

        # paths[name] = {
        #     'z': f'{path}/{filename}',
        #     'con': f'{path}/{file}_con.{ext}',
        #     'se': f'{path}/{file}_se.{ext}'

    return paths


def run_ALE(ds_dict):
    """Run ALE on given data."""
    ds = Dataset(ds_dict)
    ibma = nimare.meta.cbma.ale.ALE()
    res = ibma.fit(ds)

    img_ale = res.get_map('ale')
    img_p = res.get_map('p')
    img_z = res.get_map('z')

    return img_ale, img_p, img_z


def run_RFX_GLM(ds_dict):
    """Run RFX_GLM on given data."""
    ds = Dataset(ds_dict)
    ibma = nimare.meta.ibma.RFX_GLM()
    res = ibma.fit(ds)

    return res.get_map('t')


def run_MFX_GLM(ds_dict):
    """Run MFX_GLM on given data."""
    ds = Dataset(ds_dict)
    ibma = nimare.meta.ibma.MFX_GLM()
    res = ibma.fit(ds)

    return res.get_map('t')


def run_FFX_GLM(ds_dict):
    """Run FFX_GLM on given data."""
    ds = Dataset(ds_dict)
    ibma = nimare.meta.ibma.FFX_GLM()
    res = ibma.fit(ds)

    return res.get_map('t')


def run_Fishers(ds_dict):
    """Run Fishers on given data."""
    ds = Dataset(ds_dict)
    ibma = nimare.meta.ibma.Fishers()
    res = ibma.fit(ds)

    return res.get_map('z')


def run_Stouffers(ds_dict):
    """Run Stouffers on given data."""
    ds = Dataset(ds_dict)
    ibma = nimare.meta.ibma.Stouffers()
    res = ibma.fit(ds)

    return res.get_map('z')


def run_WeightedStouffers(ds_dict):
    """Run Weighted Stouffers on given data."""
    ds = Dataset(ds_dict)
    ibma = nimare.meta.ibma.WeightedStouffers()
    res = ibma.fit(ds)

    return res.get_map('z')


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
    # Parameters
    data_dir = 'data-narps/orig/'  # Data folder
    hyp_file = 'hypo1_unthresh.nii.gz'  # Hypothesis name
    # con_file = 'hypo1_unthresh_con.nii.gz'  # Hypothesis name
    # se_file = 'hypo1_unthresh_se.nii.gz'  # Hypothesis name
    # hyp_file_proc = 'hypo1_unthresh_resampled.nii.gz'  # Hypothesis name
    o_dir = 'data-narps/proc/'

    threshold = 1.96
    tag = f'{hyp_file}-thr-{threshold}'
    load = False
    RS = 0

    # Retrieve image paths from data folder
    path_dict = retrieve_imgs(data_dir, hyp_file)

    # print(path_dict)
    # exit()

    # print(path_dict)

    process(path_dict, o_dir=o_dir, n_sub=119, s1=10., s2=1.5, rmdir=True,
            ignore_if_exist=True, random_state=RS)
    # exit()

    # for study in ['ADFZYYLQ_C88N', 'BPZDIIWY_VG39']:
    #     img_mean = nilearn.image.load_img(f'{o_dir}{study}/{con_file}')
    #     img_se = nilearn.image.load_img(f'{o_dir}{study}/{se_file}')
    #     img_sub1 = nilearn.image.load_img(f'{o_dir}{study}/sub-001/{hyp_file}')
    #     img_sub2 = nilearn.image.load_img(f'{o_dir}{study}/sub-002/{hyp_file}')
    #     plotting.plot_stat_map(img_mean, title=f'mean {study}')
    #     plotting.plot_stat_map(img_se, title=f'std {study}')
    #     plotting.plot_stat_map(img_sub1, title=f'sub001 {study}')
    #     plotting.plot_stat_map(img_sub2, title=f'sub002 {study}')
    #     plt.show()
    # exit()

    proc_path_dict = retrieve_imgs(o_dir, hyp_file)
    # print(proc_path_dict)
    # exit()

    # Extract data from files
    ds_dict = extract_from_paths(proc_path_dict, data=['path', 'coord'],
                                 threshold=threshold, tag=tag, load=load)

    print(ds_dict)
    # exit()

    # Perform meta analysis
    img_ale, img_p, img_z = run_ALE(ds_dict)
    img_t_MFX = run_MFX_GLM(ds_dict)
    img_t_FFX = run_FFX_GLM(ds_dict)
    img_t_RFX = run_RFX_GLM(ds_dict)
    img_z_F = run_Fishers(ds_dict)
    img_z_S = run_Stouffers(ds_dict)
    img_z_WS = run_WeightedStouffers(ds_dict)
    # ds = nimare.dataset.Dataset(ds_dict)
    # img_t = nimare.meta.ibma.WeightedStouffers().fit(ds).get_map('t')
    img_ale_t, img_p_t, img_z_t = fdr_threshold([img_ale, img_p, img_z], img_p)

    plotting.plot_stat_map(img_ale, title='ALE')
    plotting.plot_stat_map(img_p, title='p')
    plotting.plot_stat_map(img_z, title='z')
    plotting.plot_stat_map(img_ale_t, title='ALE thresholded')
    plotting.plot_stat_map(img_z_t, title='z thresholded')
    plotting.plot_stat_map(img_p_t, title='p thresholded')
    plotting.plot_stat_map(img_t_MFX, title='t MFX')
    plotting.plot_stat_map(img_t_FFX, title='t FFX')
    plotting.plot_stat_map(img_t_RFX, title='t RFX')
    plotting.plot_stat_map(img_z_F, title='z Fishers')
    plotting.plot_stat_map(img_z_S, title='z Stouffers')
    plotting.plot_stat_map(img_z_WS, title='z Weighted Stouffers')
    plt.show()

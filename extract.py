"""Contain function for extracting coordinates from folders."""
import os
import shutil
import numpy as np
import nilearn
from nilearn import masking, datasets, image
from nipy.labs.statistical_mapping import get_3d_peaks
import multiprocessing
from joblib import Parallel, delayed
import ntpath
import nibabel as nib
import re
import scipy

from tools import pickle_dump, pickle_load

save_dir = 'save/'

template = datasets.load_mni152_template()
gray_mask = masking.compute_gray_matter_mask(template)


def get_sub_dict(XYZ, path):
    """
    Build sub dictionnary of a study using the nimare structure.

    Args:
        XYZ (tuple): Size 3 tuple of list storing the X Y Z coordinates.
        path (string): Absolute path to the full image.

    Returns:
        (dict): Dictionary storing the coordinates for a
            single study using the Nimare structure.

    """
    d = {
        'contrasts': {
            '0': {}
        }
    }

    if XYZ is not None:
        d['contrasts']['0']['coords'] = {
                    'x': XYZ[0],
                    'y': XYZ[1],
                    'z': XYZ[2],
                    'space': 'MNI'
                    }
        d['contrasts']['0']['sample_sizes'] = 50

    if path is not None:
        d['contrasts']['0']['images'] = {
            'con': path
        }

    return d


def get_activations(img, threshold):
    """
    Retrieve the xyz activation coordinates from an image.

    Args:
        img (string or Nifti1Image): Path to or object of a
            nibabel.Nifti1Image from which to extract coordinates.
        threshold (float): Same as the extract_from_paths function.

    Returns:
        (tuple): Size 3 tuple of lists storing respectively the X, Y and
            Z coordinates

    """
    X, Y, Z = [], [], []

    try:
        img = nilearn.image.load_img(img)
    except ValueError:  # File path not found
        print(f'File {img} not found. Ignored.')
        return None

    if np.isnan(img.get_fdata()).any():
        print(f'Img {img} contains Nan. Ignored.')
        return None

    img = image.resample_to_img(img, template)

    peaks = get_3d_peaks(img, mask=gray_mask, threshold=threshold)

    if not peaks:
        return X, Y, Z

    for peak in peaks:
        X.append(peak['pos'][0])
        Y.append(peak['pos'][1])
        Z.append(peak['pos'][2])

    del peaks
    return X, Y, Z


def extract_from_paths(path_dict, data=['coord', 'path'],
                       threshold=1.96, tag=None, load=True):
    """
    Extract data from given images.

    Extracts data (coordinates, paths...) from the data and put it in a
        dictionnary using Nimare structure.

    Args:
        path_dict (dict): Dict which keys are study names and values
            absolute paths (string).
        data (list): Data to extract. 'coord' and 'path' available.
        threshold (float): value below threshold are ignored. Used for
            peak detection.
        tag (str): Name of the file to load/dump.
        load (bool): If True, load a potential existing result.
            If False or not found, compute again.

    Returns:
        (dict): Dictionnary storing the coordinates using the Nimare
            structure.

    """
    if tag is not None:
        # Loading previously computed dict if any
        ds_dict = pickle_load(save_dir+tag, load=load)
        if ds_dict is not None:
            return ds_dict

    # Computing a new dataset dictionary
    def extract_pool(name, path):
        """Extract activation for multiprocessing."""
        print(f'Extracting {path}...')

        XYZ = None
        if 'coord' in data:
            XYZ, img = get_activations(path, threshold)
            if XYZ is None:
                return

        if 'path' in data:
            return get_sub_dict(XYZ, img)

        if XYZ is not None:
            return get_sub_dict(XYZ, None)

        return

    n_jobs = multiprocessing.cpu_count()
    res = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(extract_pool)(name, path) for name, path in path_dict.items())

    # Removing potential None values
    res = list(filter(None, res))
    # Merging all dictionaries
    ds_dict = {k: v for k, v in enumerate(res)}

    if tag is not None:
        pickle_dump(ds_dict, save_dir+tag)  # Dumping
    return ds_dict


# def process(Path, suffix='_resampled'):
#     """
#     Process images to resample them into MNI template.

#     Args:
#         Path (list): List of paths (string) of images.
#         suffix (string): Suffix added to the original file. Note that the
#             output file is stored in the same dir as the input one.

#     """
#     for path in Path:
#         try:
#             img = nilearn.image.load_img(path)
#         except ValueError:  # File path not found
#             print(f'File {path} not found. Ignored.')
#             continue

#         if np.isnan(img.get_fdata()).any():
#             print(f'File {path} contains Nan. Ignored.')
#             continue

#         img = image.resample_to_img(img, template)
#         var = nib.Nifti1Image(img.get_fdata(), img.affine)

#         base, filename = ntpath.split(path)
#         file, ext = filename.split('.', 1)

#         print(f'Resampling {path}...')
#         img.to_filename(f'{base}/{file}{suffix}.{ext}')
#         var.to_filename(f'{base}/{file}_var.{ext}')

def process(studies, o_dir, n_sub, s1, s2, rmdir=False, ignore_if_exist=False):
    """
    Process data by simulating subjects from studies' avg contrasts.

    Args:
        studies (dict): Dict with studies' alphanum names as keys and
            absolute path to avg contrast as values.
        o_dir (string): Path to output directory in which to store
            processed data.
        n_sub (int): Number of subjects to simulate.
        s1 (float): Standard deviation of the gaussian noise.
        s2 (float): Standard deviation of the gaussian kernel.
        rmdir (bool, optional): Whether to delete existing output directory
            before writing process data.
        ignore_if_exist (bool, optional): Whether to ignore this process
            if the output directory exists.

    """
    if ignore_if_exist and os.path.exists(o_dir):
        print(f'Dir {o_dir} exists. Process ignored.')
        return

    if rmdir and os.path.exists(o_dir):
        shutil.rmtree(o_dir)

    o_dir = os.path.join(o_dir, '')  # Adds trailing slash if not already

    # for study_name, path in studies.items():
    def process_pool(study_name, path):
        print(f'Processing {study_name}...')
        try:
            img = nilearn.image.load_img(path)
        except ValueError:  # File path not found
            print(f'File {path} not found. Ignored.')
            return

        if np.isnan(img.get_fdata()).any():
            print(f'Image {path} contains Nan. Ignored.')
            return

        if not re.match(r'^\w+$', study_name):
            raise ValueError('Study {study_name} contains invalid caracters.')
        o_study_path = f'{o_dir}{study_name}/'
        os.makedirs(o_study_path, exist_ok=True)

        _, filename = ntpath.split(path)
        file, ext = filename.split('.', 1)

        img = image.resample_to_img(img, template)
        img.to_filename(f'{o_study_path}{filename}')

        sub_imgs = []
        for i in range(1, n_sub+1):
            print(f'Simulating subject {i}...', end='\r')
            sub_img = sim_sub(img, s1, s2)
            sub_dir = f'{o_study_path}sub-{str(i).zfill(len(str(n_sub)))}/'
            os.makedirs(sub_dir, exist_ok=True)
            sub_img.to_filename(f'{sub_dir}{filename}')
            sub_imgs.append(sub_img)

        std_img = nilearn.image.math_img('np.std(imgs, axis=3)', imgs=sub_imgs)
        std_img.to_filename(f'{o_study_path}{file}_se.{ext}')

    n_jobs = multiprocessing.cpu_count()
    Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(process_pool)(name, path) for name, path in studies.items())


def sim_sub(img, s1, s2):
    """
    Simulate a subject's contrast from an average map.

    Generate random gaussian noise of the shape of the given image, convolve
    this noise with a gaussian kernel and add the result to the given image.

    Args:
        img (nibabel.Nifti1Image): Base image.
        s1 (float): Standard deviation of the gaussian noise.
        s2 (float): Standard deviation of the gaussian kernel applied to the
            generated noise.

    Returns:
        (nibabel.Nifti1Image): Noised image.

    """
    data = img.get_fdata()
    noise = np.random.normal(scale=s1, size=data.shape)
    noise = scipy.ndimage.gaussian_filter(noise, s2)
    noise = np.ma.masked_array(noise, np.logical_not(gray_mask.get_fdata()))

    return nib.Nifti1Image(data+noise, img.affine)

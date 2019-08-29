"""Contain function for extracting coordinates from folders."""
import os
import numpy as np
import nilearn
from nilearn import masking, datasets, image
from nipy.labs.statistical_mapping import get_3d_peaks
import multiprocessing
from joblib import Parallel, delayed
import ntpath

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


def get_activations(filepath, threshold):
    """
    Retrieve the xyz activation coordinates from an image.

    Args:
        filepath (stirng or Nifti1Image): Path to or object of a
            nibabel.Nifti1Image from which to extract coordinates.
        threshold (float): Same as the extract_from_paths function.

    Returns:
        (tuple): Size 3 tuple of lists storing respectively the X, Y and
            Z coordinates

    """
    X, Y, Z = [], [], []

    try:
        img = nilearn.image.load_img(filepath)
    except ValueError:  # File path not found
        print(f'File {filepath} not found. Ignored.')
        return None

    if np.isnan(img.get_fdata()).any():
        print(f'File {filepath} contains Nan. Ignored.')
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


def extract_from_paths(Path, data=['coord', 'path'], threshold=1.96, tag=None, load=True):
    """
    Extract data from given images.

    Extracts data (coordinates, paths...) from the data and put it in a
        dictionnary using Nimare structure.

    Args:
        Path (list): List of absolute paths (string).
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
    def extract_pool(path):
        """Extract activation for multiprocessing."""
        print(f'Extracting {path}...')

        XYZ = None
        if 'coord' in data:
            XYZ = get_activations(path, threshold)
            if XYZ is None:
                return

        if 'path' in data:
            return get_sub_dict(XYZ, path)

        if XYZ is not None:
            return get_sub_dict(XYZ, None)

        return

    n_jobs = multiprocessing.cpu_count()
    res = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(extract_pool)(path) for path in Path)

    # Removing potential None values
    res = list(filter(None, res))
    # Merging all dictionaries
    ds_dict = {k: v for k, v in enumerate(res)}

    if tag is not None:
        pickle_dump(ds_dict, save_dir+tag)  # Dumping
    return ds_dict


def process(Path, suffix='_resampled'):
    """
    Process images to resample them into MNI template.

    Args:
        Path (list): List of paths (string) of images.
        suffix (string): Suffix added to the original file. Note that the
            output file is stored in the same dir as the input one.

    """
    for path in Path:
        try:
            img = nilearn.image.load_img(path)
        except ValueError:  # File path not found
            print(f'File {path} not found. Ignored.')
            continue

        if np.isnan(img.get_fdata()).any():
            print(f'File {path} contains Nan. Ignored.')
            continue

        img = image.resample_to_img(img, template)
        base, filename = ntpath.split(path)
        file, ext = filename.split('.', 1)

        print(f'Resampling {path}...')
        img.to_filename(f'{base}/{file}{suffix}.{ext}')

"""Contain function for extracting coordinates from folders."""
import os
import numpy as np
import nilearn
from nilearn import masking, datasets, image
from nipy.labs.statistical_mapping import get_3d_peaks
import multiprocessing
from joblib import Parallel, delayed

from tools import pickle_dump, pickle_load

save_dir = 'save/'

template = datasets.load_mni152_template()
gray_mask = masking.compute_gray_matter_mask(template)


def get_sub_dict(X, Y, Z):
    """
    Build sub dictionnary of a study using the nimare structure.

    Args:
        X (list): Store the X coordinates
        Y (list): Store the Y coordinates
        Z (list): Store the Z coordinates

    Returns:
        (dict): Dictionary storing the coordinates for a
            single study using the Nimare structure.

    """
    return {
        'contrasts': {
            '0': {
                'coords':
                    {
                        'x': X,
                        'y': Y,
                        'z': Z,
                        'space': 'MNI'
                    },
                'sample_sizes': 50
            }
        }
    }


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

    for peak in peaks:
        X.append(peak['pos'][0])
        Y.append(peak['pos'][1])
        Z.append(peak['pos'][2])

    del peaks
    return X, Y, Z


def retrieve_imgs(dir_path, filename):
    """
    Return a list of path to images.

    Args:
        dir_path (str): Path to the folder containing the studies folders
        filename (str): Name of the image to look for inside each study folder

    Returns:
        (list): List of paths (string) to the found images.

    """
    # List of folders contained in dir_path folder
    Dir = [f'{dir_path}{dir}' for dir in next(os.walk(dir_path))[1]]
    try:
        # On some OS the root dict is also in the list, must be removed
        Dir.remove(dir_path)
    except ValueError:
        pass

    return [f'{dir}/{filename}' for dir in Dir]


def extract(dir_path, filename, threshold=1.96, load=True):
    """
    Extract coordinates from found images.

    Extracts coordinates from found images and put it in a
        dictionnary using Nimare structure. See retrieve_imgs
        and extract_from_paths doc for aguments.

    Args:
        dir_path (str): See retrieve_imgs doc.
        filename (str): See retrieve_imgs doc.
        threshold (float): See extract_from_paths doc.
        load (bool): If True, try to load a previously dumped dict if any.
            If not or False, compute a new one.

    Returns:
        (dict): Dictionnary storing the coordinates using the Nimare
            structure.

    """
    Path = retrieve_imgs(dir_path, filename)
    tag = f'{filename}-thr-{threshold}' if load else None
    return extract_from_paths(Path, threshold=threshold, tag=tag, load=load)


def extract_from_paths(Path, threshold=1.96, tag=None, load=True):
    """
    Extract coordinates from given images.

    Extracts coordinates from the data and put it in a
        dictionnary using Nimare structure.

    Args:
        Path (list): List of paths (string) or images (Nifti1Image).
        threshold (float): value below threshold are ignored.
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
        XYZ = get_activations(f'{path}', threshold)
        if XYZ is not None:
            return get_sub_dict(XYZ[0], XYZ[1], XYZ[2])
        return None

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

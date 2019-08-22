import os
import nibabel as nib
import numpy as np
from nilearn import masking, datasets, plotting, image
from nipy.labs.statistical_mapping import get_3d_peaks
import multiprocessing
from joblib import Parallel, delayed

from tools import pickle_dump, pickle_load

save_dir = 'save/'

template = datasets.load_mni152_template()
gray_mask = masking.compute_gray_matter_mask(template)


def get_sub_dict(X, Y, Z):
    '''
        Build sub dictionnary of a study using the
        nimare structure

        Args:
            X (list): Store the X coordinates
            Y (list): Store the Y coordinates
            Z (list): Store the Z coordinates

        Returns:
            (dict): Dictionary storing the coordinates for a
                single study using the Nimare structure.
    '''
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
    '''
        Given the path of a Nifti1Image storing a contrast,
        retrieve the xyz activation coordinates.

        Args:
            filepath (stirng): Path to a nibabel.Nifti1Image from which to
                extract coordinates.
            threshold (float): Same as the extract function

        Returns:
            (tuple): Size 3 tuple of lists storing respectively the X, Y and
                Z coordinates
    '''
    X, Y, Z = [], [], []

    try:
        img = nilearn.image.load_img(filepath)
    except ValueError:  # File path not found
        print(f'File {filepath} not found.')
        return None

    img = image.resample_to_img(img, template)
    abs_img = nib.Nifti1Image(np.absolute(img.get_data()), img.affine)
    del img

    abs_data = abs_img.get_data()
    threshold = np.percentile(abs_data[abs_data > 0], threshold)
    del abs_data

    peaks = get_3d_peaks(abs_img, mask=gray_mask)

    for peak in peaks:
        X.append(peak['pos'][0])
        Y.append(peak['pos'][1])
        Z.append(peak['pos'][2])


    del peaks
    return X, Y, Z

def extract(dir_path, filename, threshold=0., load=True):
    '''
        Extracts coordinates from the data and put it in a
        dictionnary using Nimare structure.

        Args:
            dir_path (str): Path to the folder containing the studies folders
            filename (str): Name of the image to look for inside each study folder
            threshold (float): float between 0 and 1 representing the percentile of
                the non-zero values.
            load (bool): If True, try to load a previously dumped dict if any.
                If not or False, compute a new one.

        Returns:
            (dict): Dictionnary storing the coordinates using the Nimare
                structure.
    '''
    tag = f'{filename}-thr-{threshold}'

    # Loading previously computed dict if any
    ds_dict = pickle_load(save_dir+tag, load=load)
    if ds_dict is not None:
        return ds_dict

    # Computing a new dataset dictionary
    ds_dict = {}

    # List of folders contained in dir_path folder
    dir_list = [f'{dir_path}{dir}' for dir in next(os.walk(dir_path))[1]]
    try:
        # On some OS the root dict is also in the list, must be removed
        dir_list.remove(dir_path)
    except ValueError:
        pass

    def extract_pool(directory):
        '''
            Function used for multiprocessing
        '''
        print(f'Extracting {directory}...')
        XYZ = get_activations(f'{directory}/{filename}', threshold)
        if not XYZ is None:
            return get_sub_dict(XYZ[0], XYZ[1], XYZ[2])
        return None

    n_jobs = multiprocessing.cpu_count()
    res = Parallel(n_jobs=n_jobs, backend='threading')(delayed(extract_pool)(dir) for dir in dir_list)

    # Removing potential None values
    res = list(filter(None, res))
    # Merging all dictionaries
    ds_dict = {k: v for k, v in enumerate(res)}

    # Dumping
    pickle_dump(ds_dict, save_dir+tag)
    return ds_dict

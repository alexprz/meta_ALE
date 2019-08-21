import os
import nilearn
import nibabel as nib
import numpy as np
from nilearn import masking, datasets, plotting
from nipy.labs.statistical_mapping import get_3d_peaks

from tools import pickle_dump, pickle_load

save_dir = 'save/'
# template = datasets.load_mni152_template()
# gray_mask = masking.compute_gray_matter_mask(template)

def get_activations(filepath, threshold):
    X, Y, Z = [], [], []

    try:
        img = nilearn.image.load_img(filepath)
    except ValueError:  # File path not found
        return None

    gray_mask = masking.compute_gray_matter_mask(img)
    shape1 = img.get_data().shape
    shape2 = gray_mask.get_data().shape
    assert(shape1 == shape2)
    img2 = nib.Nifti1Image(np.absolute(img.get_data()), img.affine)
    # plotting.plot_stat_map(img2)
    # plt.show()
    # return
    abs_data = img2.get_data()
    threshold = np.percentile(abs_data[abs_data > 0], threshold)
    print(threshold)
    peaks = get_3d_peaks(img2, threshold=threshold)#, mask=gray_mask)
    # print(peaks)
    for peak in peaks:
        X.append(peak['pos'][0])
        Y.append(peak['pos'][1])
        Z.append(peak['pos'][2])
    # print(peaks)
    # print(img.get_fdata())
    return X, Y, Z

def extract(dir_path, filename, threshold=0., load=True):
    tag = f'{filename}-thr-{threshold}'
    ds_dict = pickle_load(save_dir+tag, load=load)
    if ds_dict is not None:
        return ds_dict

    ds_dict = {}

    dir_list = [x[0] for x in os.walk(dir_path)]
    dir_list.remove(dir_path)

    n_tot = len(dir_list)
    for pid, dir in enumerate(dir_list):
        print(f'Computing activations of map {pid+1} out of {n_tot}.')
        XYZ = get_activations(f'{dir}/{filename}', threshold)
        if not XYZ is None:
            ds_dict[pid] = {
                'contrasts': {
                    '0': {
                        'coords':
                            {
                                'x': XYZ[0],
                                'y': XYZ[1],
                                'z': XYZ[2],
                                'space': 'MNI'
                            },
                        'sample_sizes': len(XYZ[0])
                    }
                }
            }
        # break

    pickle_dump(ds_dict, save_dir+tag)
    return ds_dict


if __name__ == '__main__':
    ds_dict = extract('data-narps/orig/', 'hypo1_unthresh.nii.gz', .95)
    print(ds_dict[0]['contrasts']['0']['sample_size'])
"""Test if the metaanalysis results are coherents."""
from nilearn import plotting, datasets
from matplotlib import pyplot as plt
import pandas as pd
import scipy
import numpy as np

from meta_analysis import Maps
# from brain_mapping import build_df_from_keyword
# from brain_mapping import template

from extract import extract_from_paths
from main import run_ALE, fdr_threshold

template = datasets.load_mni152_template()
input_path = 'brain_mapping/minimal/'
corpus_tfidf = scipy.sparse.load_npz(input_path+'corpus_tfidf.npz')


def build_index(file_path):
    '''
        Build decode & encode dictionnary of the given file_name.

        encode : dict
            key : line number
            value : string at the specified line number
        decode : dict (reverse of encode)
            key : string found in the file
            value : number of the line containing the string

        Used for the files pmids.txt & feature_names.txt
    '''
    decode = dict(enumerate(line.strip() for line in open(file_path)))
    encode = {v: k for k, v in decode.items()}

    return encode, decode

encode_feature, decode_feature = build_index(input_path+'feature_names.txt')
encode_pmid, decode_pmid = build_index(input_path+'pmids.txt')

coordinates = pd.read_csv(input_path+'coordinates.csv')


def build_df_from_keyword(keyword):
    nonzero_pmids = np.array([int(decode_pmid[index]) for index in corpus_tfidf[:, encode_feature[keyword]].nonzero()[0]])
    df = coordinates[coordinates['pmid'].isin(nonzero_pmids)]
    df['weight'] = 1
    return df


def simulate_maps(coords, sigma, size, random_state=None):
    """
    Create maps the same noised activation point

    Args:
        coords (tuple): Tuple of size 3 storing x y z coordinates.
        sigma (float): Standard deviation of the gaussian kernel.
        N (int): Number of maps to generate.

    Returns:
        (Maps): Instance of Maps object containing generated maps.

    """
    x, y, z = coords
    Ni, Nj, Nk = template.shape
    p = Maps.zeros(template=template)
    p.set_coord(0, x, y, z, 1)
    p.smooth(sigma=sigma, inplace=True)

    # plotting.plot_glass_brain(p.to_img())
    # plt.show()

    # rand_maps = Maps.zeros(Ni*Nj*Nk, template=template)
    # rand_maps.randomize(size=10*np.ones(N).astype(int), p=p, inplace=True)
    # return rand_maps

    return Maps.random(
        size=size,
        p=p,
        random_state=random_state,
        template=template
    )

if __name__ == '__main__':
    # keyword = 'prosopagnosia'
    sigma = 2.
    # df = build_df_from_keyword(keyword)
    # maps = Maps(df, template=template, groupby_col='pmid')
    # Img = maps.to_img(sequence=True, verbose=True)

    # Img = ['data-narps/orig/YZFBWTVU_Q6O0/hypo1_unthresh.nii.gz']
    size = 1000*np.ones(10).astype(int)
    maps = simulate_maps((34, -52, 44), 2*sigma, size, random_state=0)
    maps.smooth(sigma=sigma, inplace=True)

    if True:
        maps = Maps.concatenate((maps, Maps.zeros(n_maps=1, template=template)))

    plotting.plot_stat_map(maps.avg().to_img(), threshold=0.0010)
    plt.show()

    # exit()

    Img = maps.to_img(sequence=True)

    # plotting.plot_stat_map(Img[0])
    # plotting.plot_stat_map(Img[4])
    # plt.show()
    # exit()

    print('extract')
    ds_dict = extract_from_paths(Img, tag=f'test', threshold=0.003, load=False)
    # exit()
    print('run ale')
    img_ale, img_p, img_z = run_ALE(ds_dict)
    print('threshold')
    img_ale_t, img_p_t, img_z_t = fdr_threshold([img_ale, img_p, img_z], img_p)

    plotting.plot_stat_map(img_ale, title='ALE')
    plotting.plot_stat_map(img_p, title='p')
    plotting.plot_stat_map(img_z, title='z')
    plotting.plot_stat_map(img_ale_t, title='ALE thresholded')
    plotting.plot_stat_map(img_z_t, title='z thresholded')
    plotting.plot_stat_map(img_p_t, title='p thresholded')
    plt.show()

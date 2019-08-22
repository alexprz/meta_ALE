import extract
import nimare
import matplotlib
import os
from nilearn import plotting
from dotenv import load_dotenv
from matplotlib import pyplot as plt

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
    ds_dict = extract.extract(data_dir, hyp_file, threshold=.95, load=True)

    # Performing ALE
    ds = nimare.dataset.Dataset(ds_dict)
    ALE = nimare.meta.cbma.ale.ALE()
    res = ALE.fit(ds)


    # Plotting results of meta analysis
    img_ale = res.get_map('ale')
    img_p = res.get_map('p')
    img_z = res.get_map('z')

    plotting.plot_stat_map(img_ale, title='ALE')
    plotting.plot_stat_map(img_p, title='p')
    plotting.plot_stat_map(img_z, title='z')
    plt.show()

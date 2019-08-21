import extract
import nimare

data_dir = 'data-narps/orig/'  # Data folder
hyp_file = 'hypo1_unthresh.nii.gz'  # Hypothesis name

if __name__ == '__main__':
    # Extracting coordinates from data
    ds_dict = extract.extract(data_dir, hyp_file, threshold=.95, load=True)

    # Performing ALE
    ds = nimare.dataset.Dataset(ds_dict)
    ALE = nimare.meta.cbma.ale.ALE()
    res = ALE.fit(ds)

    print(res.maps)
import extract
import meta
import nimare

data_dir = 'data-narps/orig/'
hyp_file = 'hypo1_unthresh.nii.gz'

if __name__ == '__main__':
    ds_dict = extract.extract(data_dir, hyp_file, threshold=.95, load=False)
    # print(ds_dict)

    ds = nimare.dataset.Dataset(ds_dict)

    ALE = nimare.meta.cbma.ale.ALE()
    res = ALE.fit(ds)
    print(res.maps)
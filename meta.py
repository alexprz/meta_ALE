import nimare

if __name__ == '__main__':
    print('hello')

    input_dict = {
        '0': {
            'contrasts': {
                '0': {
                    'coords': {
                        'x': [1., 2., 3.],
                        'y': [1., 2., 3.],
                        'z': [1., 2., 3.],
                        'space': 'MNI'
                    },
                    'sample_sizes': 3
                }
            }
        }
    }

    ds = nimare.dataset.Dataset(input_dict)
    print(ds)

    ALE = nimare.meta.cbma.ale.ALE()
    res = ALE.fit(ds)
    print(res.maps)
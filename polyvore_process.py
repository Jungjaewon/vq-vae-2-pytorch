
import pickle
import os.path as osp
import shutil
import os

from glob import glob
from tqdm import tqdm

def write_polydataset(mode):

    img_dir = f'../deep_fashion_desinger_last/outfitdata_set3_4598/{mode}'

    target_dir = f'polyvore/{mode}'
    os.makedirs(target_dir, exist_ok=True)
    outfit_list = glob(osp.join(img_dir, '*'))
    dataset = list()

    for outfit_path in outfit_list:
        t_idx_list = list(range(1, 6))
        for t_idx in t_idx_list:
            base_list = list(range(1, 6))
            base_list.remove(t_idx)
            assert len(base_list) == 4
            dataset.append([outfit_path, t_idx, base_list])

    with open('outfitdata_set3_tagged.plk', 'rb') as fp:
        tagged_dict = pickle.load(fp)[mode]

    for outfit_path, t_idx, base_list in dataset:

        source_img_list, outfit_id = list(), osp.basename(outfit_path)
        t_cat = tagged_dict[f'{outfit_id}_{str(t_idx)}']['cate_idx']

        cat_dir = osp.join(target_dir, t_cat)
        os.makedirs(cat_dir, exist_ok=True)

        print(f'outfit_path : {outfit_path}')
        target_path = osp.join(outfit_path, f'{t_idx}.jpg')
        shutil.copy(target_path, osp.join(cat_dir, osp.basename(target_path)))







if __name__ == '__main__':
    write_polydataset("train")
    write_polydataset('val')
    write_polydataset('test')
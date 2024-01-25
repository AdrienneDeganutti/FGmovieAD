import h5py
import torch
import json
import os
import logging
import pickle as pkl
from torch.utils.data import Dataset, DataLoader

class MADdataset(Dataset):
    
    def __init__(self, visual_file, language_file, do_train, train_json_file):
        self.visual_h5 = h5py.File(visual_file, 'r')
        self.language_h5 = h5py.File(language_file, 'r')

        # Use visual file keys to define length
        self.visual_keys = list(self.visual_h5.keys())
        self.length = len(self.visual_keys)

        if do_train == True:
            SPLIT = 'train'
            train_annotations = json.load(open(train_json_file, 'r'))
            
            train_ann_directory = train_json_file.split('.')
            cache = train_ann_directory[0] + '.pickle'
            if os.path.exists(cache):
                self.load_pickle_data(cache)
            else:
                self.compute_annotations(train_annotations, cache)
    

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError('Index out of range')
        
        # Get corresponding visual and language features
        visual_key = self.visual_keys[idx]
        visual_feats = torch.tensor(self.visual_h5[visual_key])
        
        # Assuming there's a way to map visual keys to language keys
        language_key = self.map_visual_to_language_key(visual_key)
        language_feats = torch.tensor(self.language_h5[language_key])

        return visual_feats, language_feats


    def load_pickle_data(self,cache):
        '''
            The function loads preprocessed annotations and computes the max length of the sentences.

        '''
        logger = logging.getLogger("log")
        logger.info("Load cache data, please wait...")
        self.annos = pkl.load(open(cache, 'rb'))


    def close(self):
        self.visual_h5.close()
        self.language_h5.close()


def create_dataloader(args):
    dataset = MADdataset(args.DATASET.visual_file, args.DATASET.language_file, args.MODEL.do_train, args.DATASET.train_json_file)
    
    if args.MODEL.do_train == True:
        shuffle = True
    else:
        shuffle = False

    return DataLoader(dataset, args.MODEL.batch_size, shuffle)
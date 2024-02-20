import h5py
import torch
import pickle as pkl
import csv

video_features_file = '/mnt/welles/scratch/adrienne/MAD/features/CLIP_B32_frames_features_5fps.h5'
annotations_file = '/home/adrienne/FGmovieAD/output/data_cache/MAD_train_annotations.pickle'
output_file = '/home/adrienne/FGmovieAD/dataset/visual_features_tsv/train_segmented_features.tsv'

print('Loading cached annotations...')
annotations = pkl.load(open(annotations_file, 'rb'))
movies = {a['movie']:a['movie_duration'] for a in annotations}

print('Loading video features...')
with h5py.File(video_features_file, 'r') as f:
    video_feats = {m: torch.from_numpy(f[m][:]) for m in movies}

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')

    for annotation in annotations:
        annotation_ID = annotation['id']
        movie_ID = annotation['movie']
        start_frame, end_frame = annotation['frames_idx']

        print(f'Processing movie ID: {movie_ID}')
        full_feature_tensor = video_feats[movie_ID]
        segmented_feature_tensor = full_feature_tensor[start_frame:end_frame + 1]

        flattened_feature_list = segmented_feature_tensor.flatten().tolist()

        print(f'Writing annotation {annotation_ID}')
        writer.writerow([annotation_ID, flattened_feature_list])






import pickle as pkl
import csv


annotations_file = '/home/adrienne/FGmovieAD/output/data_cache/MAD_train_annotations.pickle'
output_file = '/home/adrienne/FGmovieAD/dataset/metadata/train.label.tsv'

print('Loading cached annotations...')
annotations = pkl.load(open(annotations_file, 'rb'))

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')

    for annotation in annotations:
        annotation_ID = annotation['id']
        caption = annotation['sentence']
        caption = '[{"caption": "' + caption + '"}]'

        writer.writerow([annotation_ID, caption])






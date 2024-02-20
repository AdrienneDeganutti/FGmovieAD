import pickle as pkl
import json


annotations_file = '/home/adrienne/FGmovieAD/output/data_cache/MAD_train_annotations.pickle'
output_file = '/home/adrienne/FGmovieAD/dataset/metadata/train.caption_coco_format.json'

print('Loading cached annotations...')
annotations = pkl.load(open(annotations_file, 'rb'))

# Prepare the list to hold all annotation data
annotations_list = []
id_counter = 0

 
for annotation in annotations:
    annotation_ID = annotation['id']
    caption = annotation['sentence']
    
    annotations_list.append({
        "image_id": annotation_ID,
        "caption": caption,
        "id": id_counter
    })

    id_counter += 1

final_json = {"annotations": annotations_list}

with open(output_file, 'w') as f:
    json.dump(final_json, f, indent=4)

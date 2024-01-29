"""Centralized catalog of paths."""
import os

class DatasetCatalog(object):
    DATA_DIR = "/mnt/welles/scratch/adrienne/"

    DATASETS = {
        "MAD_train":{
            "video_dir": "MAD/",
            "ann_file" : "MAD/annotations/MAD-v1/MAD_train.json",
            "feat_file": "MAD/features/CLIP_B32_frames_features_5fps.h5",
            "lang_feat": "MAD/features/CLIP_B32_language_tokens_features.h5",
            "tokenizer_folder": "stanford-corenlp-4.0.0",
            "fps" : 5,
            "split" : "train",
        },
        "MAD_val":{
            "video_dir": "MAD/",
            "ann_file":  "MAD/annotations/MAD-v1/MAD_val.json",
            "feat_file": "MAD/features/CLIP_B32_frames_features_5fps.h5",
            "lang_feat": "MAD/features/CLIP_B32_language_tokens_features.h5",
            "tokenizer_folder": "stanford-corenlp-4.0.0",
            "fps" : 5,
            "split" : "val",
        },
        "MAD_test":{
            "video_dir": "MAD/",
            "ann_file":  "MAD/annotations/MAD-v1/MAD_test.json",
            "feat_file": "MAD/features/CLIP_B32_frames_features_5fps.h5",
            "lang_feat": "MAD/features/CLIP_B32_language_tokens_features.h5",
            "tokenizer_folder": "stanford-corenlp-4.0.0",
            "fps" : 5,
            "split" : "test",
        }
    }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        args = dict(
            root=os.path.join(data_dir, attrs["video_dir"]),
            annotations_file=os.path.join(data_dir, attrs["ann_file"]),
            visual_feats_file=os.path.join(data_dir, attrs["feat_file"]),
            language_feats_file=os.path.join(data_dir, attrs["lang_feat"]),
            tokenizer_folder=os.path.join(data_dir, attrs["tokenizer_folder"]),
            split=attrs["split"],
            fps=attrs["fps"],
        )
        if "MAD" in name:
            return dict(
                factory = "MADdataset",
                args = args
            )
        raise RuntimeError("Dataset not available: {}".format(name))
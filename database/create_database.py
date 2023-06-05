import numpy as np
import json
import os
import matplotlib.pyplot as plt
import time
import pickle

'''
Database organization

db = {
    'video_name': {
        'frames': [0, 1, 2, ...], # target pedestrian appeared frames
        'nlp_annotations': {
            vid_uid_pair: {'speed': [], 'direction': [], 'description': [], 'key_frame': []},
            ...
        }
        'speed': [],
        'gps': []
    }
}
'''


def create_database(args):
    for split_name in ['train', 'val', 'test']:
        with open(args.video_splits) as f:
            datasplits = json.load(f)
        db_log = os.path.join(args.database_path, split_name + '_db_log.txt')
        with open(db_log, 'w') as f:
            f.write(f"Initialize {split_name} database \n")
            f.write(time.strftime("%d%b%Y-%Hh%Mm%Ss") + "\n")
        # 1. Init db
        db = init_db(sorted(datasplits[split_name]), db_log, args)
        # 2. get intent, remove missing frames
        # update_db_annotations(db, db_log, args)
        # 3. cut sequences, remove early frames before the first key frame, and after last key frame
        # cut_sequence(db, db_log, args)

        database_name = 'driving_database_' + split_name + '.pkl'
        with open(os.path.join(args.database_path, database_name), 'wb') as fid:
            pickle.dump(db, fid)

    print("Finished collecting database!")


def add_case(db, video_name, cog_annotation, cv_annotation, db_log):
    if video_name not in db:
        db[video_name] = {}

        # cog_annotation = annotation['pedestrians']['cognitive_annotations']
        # nlp_vid_uid_pairs = cog_annotation.keys()
    frame_list = list(cog_annotation['frames'].keys())
    frames = [int(f.split('_')[1]) for f in frame_list]

    db[video_name] = {  # ped_name is 'track_id' in cv-annotation
        'frames': frames,  # [] list of frame_idx of the target pedestrian appear
        'nlp_annotations': {
            # [vid_uid_pair: {'speed': [], 'direction': [], 'description': [], 'key_frame': []}]
        },
        'speed': [],
        'gps': []
    }

    nlp_vid_uid_pairs = list(cog_annotation['frames']['frame_0']['cognitive_annotation'].keys())
    for vid_uid in nlp_vid_uid_pairs:
        db[video_name]['nlp_annotations'][vid_uid] = {
            'driving_speed': [],
            'driving_direction': [],
            'description': [],
            'key_frame': []
            # 0: not key frame (expanded from key frames with NLP annotations)
            # 1: key frame (labeled by NLP annotations)
        }

    first_ann_idx = len(frame_list) - 1
    last_ann_idx = -1
    for i in range(len(frame_list)):
        fname = frame_list[i]
        for vid_uid in nlp_vid_uid_pairs:
            db[video_name]['nlp_annotations'][vid_uid]['driving_speed'].append(
                cog_annotation['frames'][fname]['cognitive_annotation'][vid_uid]['driving_decision_speed'])
            db[video_name]['nlp_annotations'][vid_uid]['driving_direction'].append(
                cog_annotation['frames'][fname]['cognitive_annotation'][vid_uid]['driving_decision_direction'])
            db[video_name]['nlp_annotations'][vid_uid]['description'].append(
                cog_annotation['frames'][fname]['cognitive_annotation'][vid_uid]['explanation'])
            db[video_name]['nlp_annotations'][vid_uid]['key_frame'].append(
                cog_annotation['frames'][fname]['cognitive_annotation'][vid_uid]['key_frame'])
            # record first/last ann frame idx
            if cog_annotation['frames'][fname]['cognitive_annotation'][vid_uid]['key_frame'] == 1:
                first_ann_idx = min(first_ann_idx, i)
                last_ann_idx = max(last_ann_idx, i)
        try:
            db[video_name]['speed'].append(float(cv_annotation['frames'][fname]['speed(km/hr)']))
            db[video_name]['gps'].append(cv_annotation['frames'][fname]['gps'])
        except:
            with open(db_log, 'a') as f:
                f.write(f"NO speed and gps information:  {video_name} frame {fname} \n")

    # Cut sequences, only keep frames containing both driving decision & explanations
    db[video_name]['frames'] = db[video_name]['frames'][first_ann_idx: last_ann_idx + 1]
    db[video_name]['speed'] = db[video_name]['speed'][first_ann_idx: last_ann_idx + 1]
    db[video_name]['gps'] = db[video_name]['gps'][first_ann_idx: last_ann_idx + 1]
    for vid_uid in nlp_vid_uid_pairs:
        db[video_name]['nlp_annotations'][vid_uid]['driving_speed'] \
            = db[video_name]['nlp_annotations'][vid_uid]['driving_speed'][first_ann_idx: last_ann_idx + 1]
        db[video_name]['nlp_annotations'][vid_uid]['driving_direction'] \
            = db[video_name]['nlp_annotations'][vid_uid]['driving_direction'][first_ann_idx: last_ann_idx + 1]
        db[video_name]['nlp_annotations'][vid_uid]['description'] \
            = db[video_name]['nlp_annotations'][vid_uid]['description'][first_ann_idx: last_ann_idx + 1]
        db[video_name]['nlp_annotations'][vid_uid]['key_frame'] \
            = db[video_name]['nlp_annotations'][vid_uid]['key_frame'][first_ann_idx: last_ann_idx + 1]


def init_db(video_list, db_log, args):
    db = {}
#     data_split = 'train' # 'train', 'val', 'test'
    dataroot = args.dataset_root_path
    # key_frame_folder = 'cognitive_annotation_key_frame'
    if args.dataset == 'PSI2.0':
        extended_folder = 'PSI2.0_TrainVal/annotations/cognitive_annotation_extended'
    elif args.dataset == 'PSI1.0':
        extended_folder = 'PSI1.0/annotations/cognitive_annotation_extended'

    for video_name in sorted(video_list):
        try:
            with open(os.path.join(dataroot, extended_folder, video_name, 'driving_decision.json'), 'r') as f:
                cog_annotation = json.load(f)
        except:
            with open(db_log, 'a') as f:
                f.write(f"Error loading {video_name} driving decision annotation json \n")
            continue

        if args.dataset == 'PSI2.0':
            cv_folder = 'PSI2.0_TrainVal/annotations/cv_annotation'
        elif args.dataset == 'PSI1.0':
            cv_folder = 'PSI1.0/annotations/cv_annotation'

        try:
            with open(os.path.join(dataroot, cv_folder, video_name, 'cv_annotation.json'), 'r') as f:
                cv_annotation = json.load(f)
        except:
            with open(db_log, 'a') as f:
                f.write(f"Error loading {video_name} cv annotation json \n")
            continue

        db[video_name] = {}

        add_case(db, video_name, cog_annotation, cv_annotation, db_log)
    return db
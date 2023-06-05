import numpy as np
import json
import os
import collections
def generate_data_sequence(set_name, database, args):
    frame_seq = []
    video_seq = []
    speed_seq = []
    gps_seq = []
    description_seq = []
    driving_speed_seq = []
    driving_direction_seq = []
    driving_speed_prob_seq = []
    driving_direction_prob_seq = []

    video_ids = sorted(database.keys())
    for video in sorted(video_ids): # video_name: e.g., 'video_0001'
        frame_seq.append(database[video]['frames'])
        n = len(database[video]['frames'])
        video_seq.append([video] * n)
        speed_seq.append(database[video]['speed'])
        gps_seq.append(database[video]['gps'])

        dr_speed, dr_dir, dr_speed_prob, dr_dir_prob, descrp = get_driving(database, video, args)
        driving_speed_seq.append(dr_speed)
        driving_direction_seq.append(dr_dir)
        driving_speed_prob_seq.append(dr_speed_prob)
        driving_direction_prob_seq.append(dr_dir_prob)
        description_seq.append(descrp)


    return {
        'frame': frame_seq,
        'video_id': video_seq,
        'speed': speed_seq,
        'gps': gps_seq,
        'driving_speed': driving_speed_seq,
        'driving_speed_prob': driving_speed_prob_seq,
        'driving_direction': driving_direction_seq,
        'driving_direction_prob': driving_direction_prob_seq,
        'description': description_seq
    }

def get_driving(database, video, args):
    # driving_speed, driving_dir, dr_speed_dsagr, dr_dir_dsagr, description
    n = len(database[video]['frames'])
    dr_speed = []
    dr_dir = []
    dr_speed_prob = []
    dr_dir_prob = []
    description = []
    nlp_vid_uid_pairs = list(database[video]['nlp_annotations'].keys())
    for i in range(n):
        speed_ann, speed_prob = get_driving_speed_to_category(database, video, i)
        dir_ann, dir_prob = get_driving_direction_to_category(database, video, i)
        des_ann = [database[video]['nlp_annotations'][vu]['description'][i] for vu in nlp_vid_uid_pairs
                   if database[video]['nlp_annotations'][vu]['description'][i] != '']
        dr_speed.append(speed_ann)
        dr_speed_prob.append(speed_prob)
        dr_dir.append(dir_ann)
        dr_dir_prob.append(dir_prob)
        description.append(des_ann) # may contains different number of descriptions for different frames

    return dr_speed, dr_dir, dr_speed_prob, dr_dir_prob, description


def get_driving_speed_to_category(database, video, i):
    nlp_vid_uid_pairs = list(database[video]['nlp_annotations'].keys())
    speed_ann_list = [database[video]['nlp_annotations'][vu]['driving_speed'][i] for vu in nlp_vid_uid_pairs
                 if database[video]['nlp_annotations'][vu]['driving_speed'][i] != '']
    counter = collections.Counter(speed_ann_list)
    most_common = counter.most_common(1)[0]
    speed_ann = str(most_common[0])
    speed_prob = int(most_common[1]) / len(speed_ann_list)
    # speed_ann = max(set(speed_ann_list), key=speed_ann_list.count)

    if speed_ann == 'maintainSpeed':
        speed_ann = 0
    elif speed_ann == 'decreaseSpeed':
        speed_ann = 1
    elif speed_ann == 'increaseSpeed':
        speed_ann = 2
    else:
        raise Exception("Unknown driving speed annotation: " + str(most_common))
    return speed_ann, speed_prob


def get_driving_direction_to_category(database, video, i):
    nlp_vid_uid_pairs = list(database[video]['nlp_annotations'].keys())
    direction_ann_list = [database[video]['nlp_annotations'][vu]['driving_direction'][i] for vu in nlp_vid_uid_pairs
                 if database[video]['nlp_annotations'][vu]['driving_direction'][i] != '']
    counter = collections.Counter(direction_ann_list)
    most_common = counter.most_common(1)[0]
    direction_ann = str(most_common[0])
    direction_prob = int(most_common[1]) / len(direction_ann_list)
    # direction_ann = max(set(direction_ann_list), key=direction_ann_list.count)

    if direction_ann == 'goStraight':
        direction_ann = 0
    elif direction_ann == 'turnLeft':
        direction_ann = 1
    elif direction_ann == 'turnRight':
        direction_ann = 2
    else:
        raise Exception("Unknown driving direction annotation: " + direction_ann)

    return direction_ann, direction_prob

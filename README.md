
# This folder contains the baseline of pedestrian trajectory prediction based on  PSI dataset.
([**PSI2.0**](https://github.com/PSI-Intention2022/PSI-Dataset) is an extension dataset based on the [**PSI1.0**](http://situated-intent.net/) dataset.)


## 1. PSI dataset Structure
Please refer to [PSI dataset]() for the details of PSI dataset and data structure.

### (1) Driving Decision Annotations
```python
db = {
	'video_name': *video_name*,
	'frames': {
		'frame_*frameId*': {
			'cognitive_annotation': {
				'*objType*_track_*trackId*': {
					*annotatorId1*: {
						'driving_decision_speed': str, # ['increaseSpeed', 'decreaseSpeed', 'maintainSpeed']
						'driving_decision_direction': str, # ['goStraight', 'turnLeft', 'turnRight']
						'explanation': str, 
						'key_frame': int # {0: not key frame, 1: key frame}
					},
					*annotatorId2*: {
						'driving_decision_speed': str,
						'driving_decision_direction': str, 
						'explanation': str,
						'key_frame': int
					},
					...
				}
			}
		}
	}
}
```

## 4. Driving Decision Prediction
(0) Arguments

```buildoutcfg
# Experimental Setting
Input: Observed video sequence 
Output: Driving decision prediction (Speed: increase/decrease/maintain speed/stop; Direction: go straight/turn left/turn right)
Observed sequence length: 15 frames (0.5s for 30 fps)
Prediction: 2 outputs - driving decision(s) (speed + direction)
Overlap rate: 0.9 for traingin/validation, 1 for test 
              (To sample tracks with stride length = len(observed_sequence_length) * overlap rate
Video Splits: 
    ('./splits/PSI200_split.json')
        - Train: Video_0001 ~ Video_0110
        - Val: Video_0111 ~ Video_0146
        - Test: Video_0147 ~ Video_0204
    ('./splits/PSI100_split.json')
        - Train: Video_0001 ~ Video_0082
        - Val: Video_0083 ~ Video_0088
        - Test: Video_0089 ~ Video_0110
```


(1) Generate database
```buildoutcfg
./database/create_database(args)
```
Organize the data into format as:
```python

db = {
    - *video_name*: { # video name
        - 'frames': [0, 1, 2, ...], # list of frames that the target pedestrian appear
        - 'speed': [],
        - 'gps': [],
        - 'nlp_annotations': {
            - *annotator_id*: { # annotator's id/name
                - 'speed': [], # list of driving decision (speed) at speific frame, extended from key-frame annotations 
                - 'direction': [], # list of driving decision (direction) at speific frame, extended from key-frame annotations 
                - 'description': [], # list of explanation of the intent estimation for every frame from the current annotator_id
                - 'key_frame': [] # if the specific frame is key-frame, directly annotated by the annotator. 0-NOT key-frame, 1-key-frame
            },
            ...
        }
    }
}
```
**Driving decision ground-truth:**

Here in this baseline, we use the major voting strategy to set the speed/direction annotation category with the most number
of agreements among all annotators as the ground-truth driving decision annotation.

(2) training / validation / test split

Our splits are provided in ```./splits```. Specifically, for PSI100, all videos are splited into train/val/test as ratio 
$0.75:0.05:0.2$. For PSI200, we take the first 110 videos (same as all PSI100 data) as training set, video_0111 ~ video_0146
as validation, and the rest 50 videos are for test. 

(3) Run training
```shell
python main.py
```

(4) Evaluation Metrics
```buildoutcfg
Acc-speed: Overall accuracy of speed driving decision prediction
mAcc-speed: Class-wise average accuracy of speed driving decision prediction
Acc-direction: Overall accuracy of wheel direction driving decision prediction
mAcc-direction: Class-wise average accuracy of wheel direction driving decision prediction
```
|Dataset|split|Acc-speed|mAcc-speed|Acc-direction|mAcc-direction|
|:---|:---|:-----|:-----|:-----|:-----|
|PSI200|val||||
|PSI200|test||||
|PSI100|val||||
|PSI100|test||||



(4). Environment
```buildoutcfg
Python 3.8
PyTorch 1.10.0 + Cuda 111
Tensorboard 2.10.1
```

(5) Notes

This baseline only take the bounding boxes sequence of the target pedestrian as input. However, PSI contains various
multi-modal annotations and information available for further exploration to contribute to the intent prediction. E.g.,
Video sequence, other road users bounding boxes, detailed text-based explanation annotations, etc.


### References 

[1] Tina Chen, Taotao Jing, Renran Tian, Yaobin Chen, Joshua Domeyer, Heishiro Toyoda, Rini Sherony, Zhengming Ding. "Psi: A pedestrian behavior dataset for socially intelligent autonomous car." arXiv preprint arXiv:2112.02604 (2021). 

[2] Chen, Tina, Renran Tian, and Zhengming Ding. "Visual reasoning using graph convolutional networks for predicting pedestrian crossing intention." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3103-3109. 2021. 


### Contact 

Please feel free to send any questions or comments to [tjing@tulane.edu](tjing@tulane.edu)
#!/usr/bin/env python

import time
import rospy
from std_msgs.msg import Float32MultiArray, String
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
from collections import Counter
from pathlib import Path

def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        # logging.info(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        # print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper

class GestureClassificationNode:
    """_summary_    
    """
    
    def __init__(self):
        self.clf = pickle.load(open(Path(__file__).parent.parent.joinpath("models/Poly2SVC.model"), 'rb'))
        self.ppl = pickle.load(open(Path(__file__).parent.parent.joinpath("models/training.ppl"), 'rb'))
        self.pub_chatter = rospy.Publisher('/chatter', String, queue_size=2)
        self.pub_raw_rec = rospy.Publisher('/test_GR_frequency', String, queue_size=2)
        self.frame_buffer = []
        self.prediction_buffer = []
        self.sliding_window = 90
        self.last_gesture = "NoGesture"
        self.last_execution_time = time.time()
        self.execution_duration = 3 # duration for action execution

        # self.recognition_frequency = 2 # Hz
        # self.slice_size = int( self.serie_size // self.recognition_frequency )
        
        self.label_decoding=[
            'fist',#'None'
            'turn_to_left',#'TurnLeft',
            'turn_to_right',#'TurnRight',
            'walk_to_forward', #'Forwards',
            'walk_to_backward',#'Backwards'
            'walk_to_left',#'GoLeft',
            'walk_to_right',#'GoRight',
        ]


    def frequency_query(self, labels: np.ndarray):
        data = Counter(labels)
        return data.most_common(1)[0]


    def check_threshold(self, labels: np.ndarray, confidences: np.ndarray):
        # only when most labels are the same and the lowest confidence level is above 0.83, return the gesture
        invalid_gesture = "NoGesture"
        vote_threshold = 0.7
        conf_threshold = 0.83
        label, frequency = self.frequency_query(labels)
        if frequency/len(labels) < vote_threshold:
            return invalid_gesture
        if confidences.mean() < conf_threshold:
            return invalid_gesture
        # mannually check the fist gesture
        if label == 0:
            return invalid_gesture
        # print(f"current convidance: {confidence.mean()}; current label {self.label_decoding[label]}")
        return self.label_decoding[label]

    # @log_execution_time
    # def recognize_gesture(self):
    #     current_frame_buffer = np.array(self.frame_buffer).reshape(-1, 78)
    #     x = self.ppl.transform(current_frame_buffer)
    #     y = self.clf.predict_proba(x)
    #     labels = np.argmax(y, axis=1)
    #     confidence = np.max(y, axis=1)
    #     res = self.check_threshold(labels, confidence)
    #     self.pub_raw_rec.publish(f"{res}")
    #     current_time = time.time()
    #     if res != self.last_gesture:
    #         if res == "NoGesture":
    #             # publish and continue
    #             # self.pub_chatter.publish(f"{res}")
    #             self.last_gesture = res
    #         else:
    #             # wait for the execution of the last valid action
    #             if current_time - self.last_execution_time >= self.execution_duration:
    #                 self.last_execution_time = current_time
    #                 self.pub_chatter.publish(f"{res}")
    #                 self.last_gesture = res

    @log_execution_time
    def recognize_gesture(self, data):
        current_frame = np.array(data).reshape(-1, 78)
        x = self.ppl.transform(current_frame)
        y = self.clf.predict_proba(x)
        label = np.argmax(y, axis=1)
        confidence = np.max(y, axis=1)
        pred_cur_frame = [label[0], confidence[0]]

        if len(self.prediction_buffer) < self.sliding_window:
            self.prediction_buffer.append(pred_cur_frame)
        else:
            self.prediction_buffer.pop(0)
            self.prediction_buffer.append(pred_cur_frame)
            buffer_array = np.asarray(self.prediction_buffer).reshape(-1, 2)
            labels, confidences = buffer_array[:,0].astype(int), buffer_array[:,1]
            res = self.check_threshold(labels, confidences)
            self.pub_raw_rec.publish(f"{res}")
            current_time = time.time()
            if res != self.last_gesture:
                if res == "NoGesture":
                    # publish and continue
                    # self.pub_chatter.publish(f"{res}")
                    self.last_gesture = res
                else:
                    # wait for the execution of the last valid action
                    if current_time - self.last_execution_time >= self.execution_duration:
                        self.last_execution_time = current_time
                        self.pub_chatter.publish(f"{res}")
                        self.last_gesture = res

    def callback_gesture(self, data):
        # if len(self.frame_buffer) < self.sliding_window:
        #     self.frame_buffer.append(data.data)
        # else:
        #     self.frame_buffer.pop(0)
        #     self.frame_buffer.append(data.data)
            # while True and not rospy.is_shutdown():
        self.recognize_gesture(data.data)
                # rospy.sleep(0.5)
    
    def callback_show_keypoints(self, data):
        array = np.array(data.data).reshape(-1,78)
        self.hand_keypoint_list.append(array)
        print(array)


    def run(self):
        rospy.init_node('gesture_recognizer', anonymous=True)
        rospy.Subscriber("/hand_keypoints", Float32MultiArray, self.callback_gesture)
        rospy.spin()

if __name__ == "__main__":

    gc = GestureClassificationNode()
    gc.run()

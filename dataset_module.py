import os
import numpy as np
import glob
import sqlite3
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from enum import Enum
from scipy.io import loadmat
import cv2

# Enum for gesture classes
class GestureClass(Enum):
    ##shrec2021 dataset
    # Static gestures
    ONE = 0
    TWO = 1
    THREE = 2
    FOUR = 3
    OK = 4
    MENU = 5

    # Dynamic gestures
    LEFT = 6
    RIGHT = 7
    CIRCLE = 8
    V = 9
    CROSS = 10

    # Fine-grained dynamic gestures
    GRAB = 11
    PINCH = 12
    TAP = 13
    DENY = 14
    KNOB = 15
    EXPAND = 16

    ##UCI_HAR gesture type
    WALKING = 17
    WALKING_UPSTAIRS = 18
    WALKING_DOWNSTAIRS = 19
    SITTING = 20
    STANDING = 21
    LAYING = 22

SHREC_NAME = ['ONE', 'TWO', 'THREE', 'FOUR', 'OK', 'MENU', 'LEFT', 'RIGHT', 'CIRCLE', 'V', 'CROSS', 'GRAB', 'PINCH', 'TAP', 'DENY', 'KNOB', 'EXPAND']

UCI_HAR_NAME = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']


DB4_NAME = [
    'REST',  # Label 0

    # Exercise A (1-17)
    'Index flexion', 'Index extension', 'Middle flexion', 'Middle extension',
    'Ring flexion', 'Ring extension', 'Little finger flexion', 'Little finger extension',
    'Thumb adduction', 'Thumb abduction', 'Thumb flexion', 'Thumb extension',
    'Wrist flexion', 'Wrist extension', 'Wrist radial deviation',
    'Wrist ulnar deviation', 'Wrist extension with closed hand',

    # Exercise B (1-17, offset by 12)
    'Thumb up', 'Extension of index and middle, flexion of the others',
    'Flexion of ring and little finger, extension of the others',
    'Thumb opposing base of little finger', 'Abduction of all fingers',
    'Fingers flexed together in fist', 'Pointing index', 'Adduction of extended fingers',
    'Wrist supination (axis: middle finger)', 'Wrist pronation (axis: middle finger)',
    'Wrist supination (axis: little finger)', 'Wrist pronation (axis: little finger)',

    # Exercise C (1-23, offset by 29)
    'Large diameter grasp', 'Small diameter grasp (power grip)', 'Fixed hook grasp',
    'Index finger extension grasp', 'Medium wrap', 'Ring grasp',
    'Prismatic four fingers grasp', 'Stick grasp', 'Writing tripod grasp',
    'Power sphere grasp', 'Three finger sphere grasp', 'Precision sphere grasp',
    'Tripod grasp', 'Prismatic pinch grasp', 'Tip pinch grasp', 'Quadpod grasp',
    'Lateral grasp', 'Parallel extension grasp', 'Extension type grasp',
    'Power disk grasp', 'Open a bottle with a tripod grasp',
    'Turn a screw (grasp the screwdriver with a stick grasp)',
    'Cut something (grasp the knife with an index finger extension grasp)'
]

MHEALTH_NAME = [
    "Rest",
    "Standing still",
    "Sitting and relaxing",
    "Lying down",
    "Walking",
    "Climbing stairs",
    "Waist bends forward",
    "Frontal elevation of arms",
    "Knees bending (crouching)",
    "Cycling",
    "Jogging",
    "Running",
    "Jump front & back"
]

OPPO_NAME = [
    "Stand",
    "Walk",
    "Sit",
    "Lie"
]

BERKELEY_NAME =[
       "Jumping in place",        # 0
    "Jumping jacks",              # 1
    "Bending - hands up all the way down",  # 2
    "Punching (boxing)",          # 3
    "Waving - two hands",         # 4
    "Waving - one hand (right)",  # 5
    "Clapping hands",             # 6
    "Throwing a ball",            # 7
    "Sit down then stand up",     # 8
    "Sit down",                   # 9
    "Stand up",                   # 10
    "T-pose"                      # 11
] 

LMDHG_NAME = [
    'Catching',                   # 0
    'Catching-hands-up',          # 1
    'C',                          # 2
    'Scroll-Finger',              # 3
    'Line',                       # 4
    'Rotating',                   # 5
    'Pointing',                   # 6
    'Pointing-With-Hand-Raised',  # 7
    'Resting',                    # 8
    'Shaking',                    # 9
    'Shaking-Low',                # 10
    'Shaking-Raised-Fist',        # 11
    'Slicing',                    # 12
    'Zoom'                        # 13
]


# Enum for sensor modalities (IMU, Joint, EMG)
class Modality(Enum):
    IMU = 'imu'
    JOINT = 'joint'
    EMG = 'emg'

# Base class for gesture datasets
class BaseGestureDataset(Dataset):
    def __init__(self, data_modality: List[Modality], window_size: int, padding_value: int, stride: int, sampling_rate: float, dataset_name: str, timestamps: List[int], split_joint_name: List[str], sensor_name: List[str]):
        self.gesture_count = len(GestureClass)
        self.data_modality = data_modality
        self.window_size = window_size
        self.padding_value = padding_value
        self.stride = stride
        self.sampling_rate = sampling_rate
        self.dataset_name = dataset_name
        self.timestamps = timestamps
        self.split_joint_name = split_joint_name
        self.sensor_name = sensor_name

    def get_gesture_count(self):
        return self.gesture_count

    def __len__(self):
        raise NotImplementedError("Subclasses should implement this!")

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses should implement this!")

    def select_modality(self, gesture_data: Dict[Modality, List[np.array]], data_modality=Modality.JOINT) -> list:
        if data_modality in gesture_data:
            return gesture_data[data_modality]
        else:
            raise ValueError(f"Modality {data_modality} is not present in the data.")

# Improved dataset class for SHREC2021
class Shrec2021Dataset(BaseGestureDataset):
    """
    This dataset represents a type where each gesture is labeled from its start time to end time, 
    indicating a continuous segment of a specific gesture. labels to specific periods for each gesture should be done in advance
    """
    def __init__(self, data_path: str, annotation_path: str, data_modality: List[Modality], window_size: int, padding_value: int, stride: int, sampling_rate: float, dataset_name: str, timestamps: List[int], split_joint_name: List[str], sensor_name: List[str]):
        super().__init__(data_modality, window_size, padding_value,stride, sampling_rate, dataset_name, timestamps, split_joint_name, sensor_name)
        self.data_path = data_path
        self.annotation_path = annotation_path
        self.stride = stride
        self.sampling_rate = sampling_rate
        self.dataset_name = dataset_name
        self.timestamps = timestamps
        self.split_joint_name = split_joint_name
        self.sensor_name = sensor_name      
        # Load and process the data
        self.gesture_data_sorted = self.read_dataset(self.data_path)
        self.gesture_data, self.label, self.gesture_num = self.read_annotation(self.annotation_path, self.gesture_data_sorted)
        self.gesture_single_modal = self.select_modality(self.gesture_data_sorted, data_modality=Modality.JOINT)
        
    def __len__(self):
        return len(self.gesture_single_modal)
    
    def __getitem__(self, idx):
        label_mapping = {gesture.name: idx for idx, gesture in enumerate(GestureClass)}
        gesture = self.gesture_single_modal[idx]
        gesture = self.pad_gesture_segment(gesture, self.window_size)
        label = label_mapping[self.label[idx].name]
        return gesture, label

    def read_dataset(self, data_path) -> Dict[Modality, list]:
        modality_data = {modality: [] for modality in Modality}
        sequence_idx_list = []
        full_path = os.path.join(os.getcwd(), data_path)

        files = sorted(glob.glob(os.path.join(full_path, "*")))
        for fname in files:
            idx = os.path.basename(fname).split('.')[0]
            sequence_idx_list.append(int(idx))

            data = np.genfromtxt(fname, delimiter=';')
            data_ = np.delete(data, -1, axis=1)

            # Here, the correct modality should be assigned to the data based on its type
            imu_data = []  # Example of how to handle different modalities
            joint_data = data_
            emg_data = []
            modality_data[Modality.IMU].append(imu_data)
            modality_data[Modality.JOINT].append(joint_data)
            modality_data[Modality.EMG].append(emg_data)

        for modality in modality_data:
            modality_data[modality] = [x for _, x in sorted(zip(sequence_idx_list, modality_data[modality]))]

        return modality_data

    def read_annotation(self, annotation_path, gesture_data_sorted):
        full_path = os.path.join(os.getcwd(), annotation_path)
        with open(full_path, 'r') as file:
            lines = file.readlines()

        annotation_idx_list = []
        gesture_name_list = []
        start_idx_list = []
        end_idx_list = []

        for line in lines:
            parts = line.strip().split(';')
            annotation_idx_list.append(int(parts[0]))
            gesture_names = parts[1:-1:3]
            starts = parts[2:-1:3]
            ends = parts[3::3]
            gesture_name_list.append(gesture_names)
            start_idx_list.append(starts)
            end_idx_list.append(ends)

        gesture_name_list = [x for _, x in sorted(zip(annotation_idx_list, gesture_name_list))]
        start_idx_list = [x for _, x in sorted(zip(annotation_idx_list, start_idx_list))]
        end_idx_list = [x for _, x in sorted(zip(annotation_idx_list, end_idx_list))]

        single_gesture = []
        gesture_class = []
        joint_data = self.select_modality(gesture_data_sorted, data_modality=Modality.JOINT)

        for i in range(len(joint_data)):
            for j in range(len(gesture_name_list[i])):
                start_idx = int(start_idx_list[i][j])
                end_idx = int(end_idx_list[i][j])
                gesture_segment = joint_data[i][start_idx:end_idx]
                padded_gesture = self.pad_gesture_segment(gesture_segment, self.window_size)
                single_gesture.append(padded_gesture)
                gesture_enum = GestureClass[gesture_name_list[i][j]]
                gesture_class.append(gesture_enum.value)

        label_to_gesture = {
            0: GestureClass.ONE.value,
            1: GestureClass.TWO.value,
            2: GestureClass.THREE.value,
            3: GestureClass.FOUR.value,
            4: GestureClass.OK.value,
            5: GestureClass.MENU.value,
            6: GestureClass.LEFT.value,
            7: GestureClass.RIGHT.value,
            8: GestureClass.CIRCLE.value,
            9: GestureClass.V.value,
            10: GestureClass.CROSS.value,
            11: GestureClass.GRAB.value,
            12: GestureClass.PINCH.value,
            13: GestureClass.TAP.value,
            14: GestureClass.DENY.value,
            15: GestureClass.KNOB.value,
            16: GestureClass.EXPAND.value
        }

        return np.array(single_gesture, dtype=np.float32), np.array(gesture_class), len(label_to_gesture)

    def pad_gesture_segment(self, gesture_segment, window_size):
        if len(gesture_segment) < window_size:
            padding_length = window_size - len(gesture_segment)
            padding_array = np.full((padding_length, gesture_segment.shape[1]), self.padding_value)
            gesture_segment = np.concatenate((gesture_segment, padding_array), axis=0)
        else:
            gesture_segment = gesture_segment[:window_size]
        return gesture_segment.astype(np.float32)
    

class UCI_HAR_dataset(BaseGestureDataset):
    """
    This dataset represents a type where each row of data corresponds to a definite label, 
    rather than a gesture from a start time to an end time corresponding to a label.
    """
    def __init__(self, data_path: str, annotation_path: str, data_modality: List[Modality], window_size: int, padding_value: int, stride: int, sampling_rate: float, dataset_name: str, timestamps: List[int], split_joint_name: List[str], sensor_name: List[str]):
        super().__init__(data_modality, window_size, padding_value, stride, sampling_rate, dataset_name, timestamps, split_joint_name, sensor_name)
        self.data_path = data_path
        self.annotation_path = annotation_path
        # Load and process the data
        self.gesture_data_sorted = self.read_dataset(self.data_path)
        self.label, self.gesture_num = self.read_annotation(self.annotation_path)
        self.gesture_IMU = self.select_modality(self.gesture_data_sorted, data_modality=Modality.IMU)

        
    def __len__(self):
        return len(self.gesture_IMU)
    

    def __getitem__(self, idx):
        gesture = self.gesture_IMU[idx]
        label = self.label[idx]
        return gesture, label

    def read_dataset(self, data_path) -> Dict[Modality, list]:
        """
    Reads the UCI_HAR dataset and organizes it under the corresponding modality.
    
    Args:
        data_path (str): Path to the UCI_HAR dataset file (e.g., train_uci.txt or test_uci.txt).
        
    Returns:
        modality_data (Dict[Modality, list]): A dictionary with the IMU modality as the key
                                               and a list of numpy arrays for each sample.
    """
        modality_data = {modality: [] for modality in Modality}
        full_path = os.path.join(os.getcwd(), data_path)
        
        # Read the data file where each row represents a sample with features
        modality_data[Modality.IMU] = np.loadtxt(full_path)
    
        # Here, the correct modality should be assigned to the data based on its type
        joint_data = []  
        emg_data = []
        modality_data[Modality.JOINT].append(joint_data)
        modality_data[Modality.EMG].append(emg_data)

        return modality_data

    
    def read_annotation(self, annotation_path):
        full_path = os.path.join(os.getcwd(), annotation_path)
        annotation_label = np.loadtxt(full_path)

        # Create a mapping from label values to GestureClass enums
        label_to_gesture = {
            1: GestureClass.WALKING.value - GestureClass.WALKING.value,
            2: GestureClass.WALKING_UPSTAIRS.value - GestureClass.WALKING.value,
            3: GestureClass.WALKING_DOWNSTAIRS.value - GestureClass.WALKING.value,
            4: GestureClass.SITTING.value - GestureClass.WALKING.value,
            5: GestureClass.STANDING.value - GestureClass.WALKING.value,
            6: GestureClass.LAYING.value - GestureClass.WALKING.value
        }

        gesture_value = [label_to_gesture[label] for label in annotation_label]
        gesture_num = len(label_to_gesture)
        

        return np.array(gesture_value), gesture_num



class DB4GestureClass:
    """
    For the original data representation, and the real data involved here is preprocessed
    before uploading to the server for easy deployment.
    """
    # Exercise A: 1-17
    A_LABELS = {i: i for i in range(1, 13)}

    # Exercise B: 1-17, offset by 12 (i.e., 13-29)
    B_LABELS = {i: i + 12 for i in range(1, 18)}

    # Exercise C: 1-23, offset by 29 (i.e., 30-52)
    C_LABELS = {i: i + 29 for i in range(1, 24)}

    # REST: Label 0
    REST = 0

    @staticmethod
    def get_label(exercise, gesture):
        """
        获取手势的全局标签值，根据Exercise和gesture编号映射到统一标签。
        """
        if exercise == "A":
            return DB4GestureClass.A_LABELS.get(gesture, 0)
        elif exercise == "B":
            return DB4GestureClass.B_LABELS.get(gesture, 0)
        elif exercise == "C":
            return DB4GestureClass.C_LABELS.get(gesture, 0)
        else:
            return "Wrong gesture type in the original data files!"


# Dataset for DB4
class DB4_dataset(BaseGestureDataset):
    """
    This dataset represents a type where each row of data corresponds to a definite label, 
    rather than a gesture from a start time to an end time corresponding to a label.
    """
    def __init__(self, data_path: str, annotation_path: str, data_modality: List[Modality], window_size: int, padding_value: int, stride: int, sampling_rate: float, dataset_name: str, timestamps: List[int], split_joint_name: List[str], sensor_name: List[str]):
        super().__init__(data_modality, window_size, padding_value, stride, sampling_rate, dataset_name, timestamps, split_joint_name, sensor_name)
        self.data_path = data_path
        self.annotation_path = annotation_path
        # Load and process the data
        self.gesture_data= self.read_dataset(self.data_path)
        self.label= self.read_annotation(self.annotation_path)
        self.gesture_EMG = self.select_modality(self.gesture_data, data_modality=Modality.EMG)
        self.gesture_num = len(DB4_NAME)

        
    def __len__(self):
        return len(self.gesture_EMG)
    

    def __getitem__(self, idx):
        gesture = self.gesture_EMG[idx]
        label = self.label[idx]
        return gesture, label

    def read_dataset(self, data_path) -> Dict[Modality, list]:
        """
    Reads the DB4 dataset and organizes it under the corresponding modality.
    
    Args:
        data_path (str): Path to the DB4 dataset file (e.g., train_db4.txt or test_db4.txt).
        
    Returns:
        modality_data (Dict[Modality, list]): A dictionary with the IMU modality as the key
                                               and a list of numpy arrays for each sample.
    """
        modality_data = {modality: [] for modality in Modality}
        full_path = os.path.join(os.getcwd(), data_path)
        
        # Read the data file where each row represents a sample with features
        modality_data[Modality.EMG] = np.loadtxt(full_path)
    
        # Here, the correct modality should be assigned to the data based on its type
        joint_data = []  
        imu_data = []
        modality_data[Modality.JOINT].append(joint_data)
        modality_data[Modality.IMU].append(imu_data)

        return modality_data

    
    def read_annotation(self, annotation_path):
        full_path = os.path.join(os.getcwd(), annotation_path)
        annotation_label = np.loadtxt(full_path)

        return np.array(annotation_label)




class mhealth_dataset(BaseGestureDataset):
    """
    This dataset represents a type where each row of data corresponds to a definite label, 
    rather than a gesture from a start time to an end time corresponding to a label.
    Label distribution in the dataset:
        Label 0: 72682 samples
        Label 1: 2560 samples
        Label 2: 2560 samples
        Label 3: 2560 samples
        Label 4: 2560 samples
        Label 5: 2562 samples
        Label 6: 2361 samples
        Label 7: 2453 samples
        Label 8: 2449 samples
        Label 9: 2560 samples
        Label 10: 2560 samples
        Label 11: 2560 samples
        Label 12: 866 samples
        原始 Label 12 样本数量: 866
        原始 Label 0 样本数量: 72682
        增强后 Label 12 样本数量: 2598
        减少后 Label 0 样本数量: 2560
        shape: (sample, 25, 21)
    """
    def __init__(self, data_path: str, annotation_path: str, data_modality: List[Modality], window_size: int, padding_value: int, stride: int, sampling_rate: float, dataset_name: str, timestamps: List[int], split_joint_name: List[str], sensor_name: List[str]):
        super().__init__(data_modality, window_size, padding_value, stride, sampling_rate, dataset_name, timestamps, split_joint_name, sensor_name)
        self.data_path = data_path
        self.annotation_path = annotation_path
        # Load and process the data
        self.gesture_data= self.read_dataset(self.data_path)
        self.label= self.read_annotation(self.annotation_path)
        self.gesture_IMU = self.select_modality(self.gesture_data, data_modality=Modality.IMU)
        self.gesture_num = len(MHEALTH_NAME)

        
    def __len__(self):
        return len(self.gesture_IMU)
    

    def __getitem__(self, idx):
        gesture = self.gesture_IMU[idx]
        label = self.label[idx]
        return gesture, label

    def read_dataset(self, data_path) -> Dict[Modality, list]:
        """
    Reads the mhealth dataset and organizes it under the corresponding modality.
    
    Args:
        data_path (str): Path to the mhealth dataset file (e.g., train_mhealth.txt or test_mhealth.txt).
        
    Returns:
        modality_data (Dict[Modality, list]): A dictionary with the IMU modality as the key
                                               and a list of numpy arrays for each sample.
    """
        modality_data = {modality: [] for modality in Modality}
        full_path = os.path.join(os.getcwd(), data_path)
        
        # Read the data file where each row represents a sample with features
        modality_data[Modality.IMU] = np.load(full_path)
    
        # Here, the correct modality should be assigned to the data based on its type
        joint_data = []  
        emg_data = []
        modality_data[Modality.JOINT].append(joint_data)
        modality_data[Modality.EMG].append(emg_data)

        return modality_data

    
    def read_annotation(self, annotation_path):
        full_path = os.path.join(os.getcwd(), annotation_path)
        annotation_label = np.loadtxt(full_path)

        return np.array(annotation_label)



class LMDHG_dataset(Dataset):
    """
    This dataset represents a type where each row of data corresponds to a definite label, 
    rather than a gesture from a start time to an end time corresponding to a label.
    Label distribution in the dataset:
    Train Image Counts:
    Label: Scroll_Finger, Count: 112
    Label: Catching, Count: 92
    Label: Resting, Count: 136
    Label: Shaking_Raised_Fist, Count: 92
    Label: Slicing, Count: 100
    Label: Shaking_Low, Count: 104
    Label: Rotating, Count: 104
    Label: Catching_hands_up, Count: 108
    Label: Line, Count: 112
    Label: Zoom, Count: 104
    Label: Shaking, Count: 92
    Label: C, Count: 100
    Label: Pointing, Count: 116
    Label: Pointing_With_Hand_Raised, Count: 36

    Test Image Counts:
    Label: C, Count: 48
    Label: Pointing, Count: 44
    Label: Resting, Count: 57
    Label: Slicing, Count: 48
    Label: Scroll_Finger, Count: 44
    Label: Pointing_With_Hand_Raised, Count: 44
    Label: Shaking_Low, Count: 56
    Label: Shaking, Count: 48
    Label: Line, Count: 40
    Label: Rotating, Count: 36
    Label: Shaking_Raised_Fist, Count: 48
    Label: Catching, Count: 44
    Label: Catching_hands_up, Count: 48
    Label: Zoom, Count: 56

    """
    def __init__(self, data_path: str, dataset_name: str):
        
        self.data_path = data_path
        # Load and process the data
        self.gesture_data, self.label = self.read_dataset(self.data_path)
        self.gesture_num = len(LMDHG_NAME)

        
    def __len__(self):
        return len(self.gesture_IMU)
    

    def __getitem__(self, idx):
        gesture = self.gesture_data[idx]
        label = self.label[idx]
        return gesture, label
    


    def read_dataset(self, data_path) -> Tuple[List[np.ndarray], List[int]]:
        """
        读取 LMDHG 数据集中的图像和标签
        
        Args:
            data_path (str): 数据集根目录路径
            
        Returns:
            Tuple[List[np.ndarray], List[int]]: 
                - 图像数据列表
                - 对应的标签列表
        """
        gesture_data = []
        labels = []
        label_mapping = {name: idx for idx, name in enumerate(LMDHG_NAME)}
        
        full_path = os.path.join(os.getcwd(), data_path)
        for root, _, files in os.walk(full_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    # 获取图像完整路径
                    img_path = os.path.join(root, file)
                    
                    # 读取图像
                    img = cv2.imread(img_path)
                    ###rgb 2 luminance
                    # img_2D = np.dot(img[..., :3], [0.2126, 0.7152, 0.0722]).astype(np.int8)    acc 0.7890
                    ##enhance luminance
                    luminance = np.dot(img[..., :3], [0.2126, 0.7152, 0.0722])  # 计算亮度
                    enhanced = np.log1p(luminance)
                    img_2D = (enhanced / np.max(enhanced) * 255).astype(np.int8)
                    ### rgb maximum channel
                    # max_channel = np.max(img, axis=2)
                    # img_2D = max_channel

                    if img_2D is not None:
                        gesture_data.append(img_2D)
                        
                        # 从路径中提取标签名称
                        
                        # gesture_name = os.path.basename(root)
                        gesture_name = file[:-9]
                        label = label_mapping[gesture_name]
                        labels.append(label)

        return np.array(gesture_data), np.array(labels)

    

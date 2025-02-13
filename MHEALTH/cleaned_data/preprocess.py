import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.utils import shuffle
from collections import Counter



def preprocess_mhealth_data(input_folder, window_size=25, overlap=0.5):
    """
    Preprocess mHealth dataset and split into train and test sets.

    Parameters:
    - input_folder (str): Folder containing `.log` files.
    - output_folder (str): Folder to save the preprocessed train and test files.
    - window_size (int): Number of timestamps per window.
    - overlap (float): Overlap ratio between consecutive windows.

    Outputs:
    - X_train.txt, y_train.txt, X_test.txt, y_test.txt
    """
    all_data = []
    all_labels = []

    # List all .log files in the folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.log'):
            file_path = os.path.join(input_folder, file_name)
            
            # Load the data
            data = pd.read_csv(file_path, header=None, delim_whitespace=True)
            labels = data.iloc[:, -1].values
            # Remove columns 4, 5, and 24
          
            data = data.drop(columns=[3, 4, 23])
         
            # Extract labels
            
            features = data.values

            # Create overlapping windows
            window_size = int(window_size)
            overlap = float(overlap)
            step_size = int(window_size * (1 - overlap))
            num_windows = (len(features) - window_size) // step_size + 1
            
            for i in range(num_windows):
                start_idx = i * step_size
                end_idx = start_idx + window_size
                
                window_features = features[start_idx:end_idx]
                window_label = labels[start_idx:end_idx]
                
                # Decide label by majority voting
                label = np.bincount(window_label.astype(int)).argmax()
                
                all_data.append(window_features)
                all_labels.append(label)
    
    # Convert to arrays
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    print(all_data.shape)
    print(all_labels.shape)

    # Count label occurrences
    label_counts = Counter(all_labels)
    print("Label distribution in the dataset:")
    for label, count in sorted(label_counts.items()):
        print(f"Label {label}: {count} samples")

    return all_data, all_labels


class DataAugmentor:
    def __init__(self, translation_range=0.1, scale_range=0.1, noise_std=0.01, input_dim=525):
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.input_dim = input_dim

    def translate_data(self, data):
        # 平移
        translation = np.random.uniform(-self.translation_range, self.translation_range, size=(1, data.shape[1]))
        return data + translation

    def scale_data(self, data):
        # 缩放
        scale = np.random.uniform(1 - self.scale_range, 1 + self.scale_range)
        return data * scale

    def add_noise(self, data):
        # 加噪声
        noise = np.random.normal(0, self.noise_std, size=data.shape)
        return data + noise

    def augment_data(self, data):
        # 数据增强：随机选择平移、缩放、加噪声中的任意组合
        augmentations = [self.translate_data, self.scale_data, self.add_noise]
        np.random.shuffle(augmentations)
        for augment in augmentations[:2]:  # 随机应用两种增强
            data = augment(data)
        return data




def balance_labels(X, y, target_label=12, reduce_label=0, reduce_to=2560):
    """
    对目标标签进行增强并对指定标签进行样本减少。

    Parameters:
    - X (ndarray): 输入特征，形状为 (num_samples, input_dim)。
    - y (ndarray): 标签，形状为 (num_samples,)。
    - target_label (int): 需要增强的目标标签。
    - reduce_label (int): 需要减少的目标标签。
    - reduce_to (int): 保留的目标标签样本数量。

    Returns:
    - X_balanced (ndarray): 增强和减少后的特征数据。
    - y_balanced (ndarray): 增强和减少后的标签数据。
    """
    # 提取目标标签的样本
    X_target = X[y == target_label]
    X_reduce = X[y == reduce_label]
    X_other = X[(y != target_label) & (y != reduce_label)]
    y_other = y[(y != target_label) & (y != reduce_label)]

    print(f"原始 Label {target_label} 样本数量: {X_target.shape[0]}")
    print(f"原始 Label {reduce_label} 样本数量: {X_reduce.shape[0]}")

    # 初始化增强器
    augmentor = DataAugmentor(input_dim=X.shape[1])

    # 对 target_label 增强两次
    augmented_samples_1 = np.array([augmentor.augment_data(sample) for sample in X_target])
    augmented_samples_2 = np.array([augmentor.augment_data(sample) for sample in X_target])
    X_target_balanced = np.vstack([X_target, augmented_samples_1, augmented_samples_2])
    y_target_balanced = np.array([target_label] * X_target_balanced.shape[0])

    # 对 reduce_label 随机选取指定数量样本
    X_reduce_balanced = shuffle(X_reduce, random_state=42)[:reduce_to]
    y_reduce_balanced = np.array([reduce_label] * X_reduce_balanced.shape[0])

    # 合并所有数据
    X_balanced = np.vstack([X_target_balanced, X_reduce_balanced, X_other])
    y_balanced = np.hstack([y_target_balanced, y_reduce_balanced, y_other])

    # 打乱数据
    X_balanced, y_balanced = shuffle(X_balanced, y_balanced, random_state=42)

    # 打印增强和减少后的分布
    print(f"增强后 Label {target_label} 样本数量: {X_target_balanced.shape[0]}")
    print(f"减少后 Label {reduce_label} 样本数量: {X_reduce_balanced.shape[0]}")
    print(f"重平衡后的标签分布: {Counter(y_balanced)}")

    return X_balanced, y_balanced


# Example Usage
if __name__ == "__main__":
    # 假设已经加载数据
    X, y = preprocess_mhealth_data("./MHealth/MHEALTHDATASET")  
   

    # 对 Label 12 增强并重平衡数据集
    X_rebalanced, y_rebalanced = balance_labels(X, y, target_label=12)
    X_train, X_test, y_train, y_test = train_test_split(X_rebalanced, y_rebalanced, test_size=0.2, random_state=42)
    # 保存重平衡后的数据
    np.save("X_train.npy", X_rebalanced)
    np.savetxt("y_train.txt", y_rebalanced, fmt="%d", delimiter="\n")
    np.save("X_test.npy", X_rebalanced)
    np.savetxt("y_test.txt", y_rebalanced, fmt="%d", delimiter="\n")
    print("Data successfully split and saved!")
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.interpolate import interp1d

class DataAugmentor:
    def __init__(self, translation_range=0.1, rotation_range=np.pi/4, scale_range=0.1, noise_std=0.01, speed_factor=1.0, max_length=250, input_dim=140):
        self.translation_range = translation_range
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.speed_factor = speed_factor
        self.max_length = max_length
        self.input_dim = input_dim

    def translate_data(self, data):
        translation = np.random.uniform(-self.translation_range, self.translation_range, size=(1, 3))
        translated_data = data[:, :, :3] + translation
        return np.concatenate((translated_data, data[:, :, 3:]), axis=2)

    def rotate_data(self, data):
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        axis = np.random.normal(size=3)
        axis /= np.linalg.norm(axis)
        r = R.from_rotvec(angle * axis)
        quats = data[:, :, 3:]
        norms = np.linalg.norm(quats, axis=2, keepdims=True)
        norms[norms == 0] = 1e-6
        normalized_quats = quats / norms
        try:
            rotated_data = np.array([r.apply(frame[:, :3]) for frame in data])
            rotated_quats = np.array([(r * R.from_quat(frame)).as_quat() for frame in normalized_quats])
            return np.concatenate((rotated_data, rotated_quats), axis=2)
        except Exception as e:
            return data

    def scale_data(self, data):
        scale = np.random.uniform(1 - self.scale_range, 1 + self.scale_range, size=(1, 1, 3))
        scaled_data = data[:, :, :3] * scale
        return np.concatenate((scaled_data, data[:, :, 3:]), axis=2)

    def add_noise(self, data):
        noisy_data = data[:, :, :3] + np.random.normal(0, self.noise_std, size=data[:, :, :3].shape)
        noisy_quats = data[:, :, 3:] + np.random.normal(0, self.noise_std, size=data[:, :, 3:].shape)
        return np.concatenate((noisy_data, noisy_quats), axis=2)

    def normalize_quaternions(self, data):
        quats = data[:, :, 3:]
        norms = np.linalg.norm(quats, axis=2, keepdims=True)
        normalized_quats = quats / norms
        return np.concatenate((data[:, :, :3], normalized_quats), axis=2)

    def time_warp(self, sequence):
        num_frames, num_features = sequence.shape
        new_num_frames = int(num_frames / self.speed_factor)
        original_indices = np.arange(num_frames)
        new_indices = np.linspace(0, num_frames - 1, new_num_frames)
        warped_sequence = np.zeros((new_num_frames, num_features))
        for i in range(num_features):
            interp_function = interp1d(original_indices, sequence[:, i], kind='linear')
            warped_sequence[:, i] = interp_function(new_indices)
        data_padded = np.zeros((self.max_length, self.input_dim))
        data_padded[:warped_sequence.shape[0]] = warped_sequence
        return data_padded.astype(np.float32)

    def augment_data(self, sequence):
        sequence = sequence.reshape(-1, 20, 7)  # Reshape to [250, 20, 7]
        sequence = self.translate_data(sequence)
        # sequence = self.rotate_data(sequence)
        sequence = self.scale_data(sequence)
        sequence = self.add_noise(sequence)
        sequence = self.normalize_quaternions(sequence)
        return sequence.reshape(-1, 140).astype(np.float32)  # Reshape back to [250, 140]
import cv2
import dlib
import torch

from thad.poser.morph_rotate_combine_poser import MorphRotateCombinePoser256Param6
from thad.puppet.head_pose_solver import HeadPoseSolver
from thad.poser.poser import Poser
from thad.puppet.util import compute_left_eye_normalized_ratio, compute_right_eye_normalized_ratio, \
    compute_mouth_normalized_ratio
from thad.tha.combiner import CombinerSpec
from thad.tha.face_morpher import FaceMorpherSpec
from thad.tha.two_algo_face_rotator import TwoAlgoFaceRotatorSpec
from thad.util import rgba_to_numpy_image_greenscreen


class PredictorLocal:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.poser = MorphRotateCombinePoser256Param6(
            morph_module_spec=FaceMorpherSpec(),
            morph_module_file_name="checkpoints/face_morpher.pt",
            rotate_module_spec=TwoAlgoFaceRotatorSpec(),
            rotate_module_file_name="checkpoints/two_algo_face_rotator.pt",
            combine_module_spec=CombinerSpec(),
            combine_module_file_name="checkpoints/combiner.pt",
            device=self.device)
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_locator = dlib.shape_predictor("checkpoints/shape_predictor_68_face_landmarks.dat")
        self.head_pose_solver = HeadPoseSolver()
        self.pose_size = len(self.poser.pose_parameters())
        self.source_image = None
        self.current_pose = None
        self.last_pose = None

    def set_source_image(self, image):
        image_size = image.shape[0]
        image = image \
                    .reshape(image_size * image_size, 4) \
                    .transpose() \
                    .reshape(4, image_size, image_size)
        torch_image = torch.from_numpy(image).float() * 2.0 - 1.0
        self.source_image = torch_image.to(self.device).unsqueeze(dim=0)

    def predict(self, driving_frame):
        assert self.source_image is not None, "call set_source_image()"

        rgb_frame = cv2.cvtColor(driving_frame, cv2.COLOR_BGR2RGB)
        faces = self.face_detector(rgb_frame)
        euler_angles = None
        face_landmarks = None
        if len(faces) > 0:
            face_rect = faces[0]
            face_landmarks = self.landmark_locator(rgb_frame, face_rect)
            face_box_points, euler_angles = self.head_pose_solver.solve_head_pose(face_landmarks)

        if euler_angles is not None and self.source_image is not None:
            self.current_pose = torch.zeros(self.pose_size, device=self.device)
            self.current_pose[0] = max(min(-euler_angles.item(0) / 15.0, 1.0), -1.0)
            self.current_pose[1] = max(min(-euler_angles.item(1) / 15.0, 1.0), -1.0)
            self.current_pose[2] = max(min(euler_angles.item(2) / 15.0, 1.0), -1.0)

            if self.last_pose is None:
                self.last_pose = self.current_pose
            else:
                self.current_pose = self.current_pose * 0.5 + self.last_pose * 0.5
                self.last_pose = self.current_pose

            eye_min_ratio = 0.15
            eye_max_ratio = 0.25
            left_eye_normalized_ratio = compute_left_eye_normalized_ratio(face_landmarks, eye_min_ratio, eye_max_ratio)
            self.current_pose[3] = 1 - left_eye_normalized_ratio
            right_eye_normalized_ratio = compute_right_eye_normalized_ratio(face_landmarks,
                                                                            eye_min_ratio,
                                                                            eye_max_ratio)
            self.current_pose[4] = 1 - right_eye_normalized_ratio

            min_mouth_ratio = 0.02
            max_mouth_ratio = 0.3
            mouth_normalized_ratio = compute_mouth_normalized_ratio(face_landmarks, min_mouth_ratio, max_mouth_ratio)
            self.current_pose[5] = mouth_normalized_ratio

            self.current_pose = self.current_pose.unsqueeze(dim=0)

            posed_image = self.poser.pose(self.source_image, self.current_pose).detach().cpu()
            out = rgba_to_numpy_image_greenscreen(posed_image[0])
            return out

from isaacgymenvs.tasks import isaacgym_task_map
from .shadow_hand_grasp import ShadowHandGrasp
from .leap_hand_grasp import LeapHandGrasp
from .bi_leap_hand_grasp_v0 import BiLeapHandGraspV0
from .bi_leap_hand_grasp_v1 import BiLeapHandGraspV1
from .bi_leap_hand_grasp_v2 import BiLeapHandGraspV2
# from .bi_leap_hand_grasp_pcd import BiLeapHandGraspPCD
from .bi_leap_hand_grasp_vision import BiLeapHandGraspVision

isaacgym_task_map["ShadowHandGrasp"] = ShadowHandGrasp
isaacgym_task_map["LeapHandGrasp"] = LeapHandGrasp
isaacgym_task_map["BiLeapHandGraspV0"] = BiLeapHandGraspV0
isaacgym_task_map["BiLeapHandGraspV1"] = BiLeapHandGraspV1
isaacgym_task_map["BiLeapHandGraspV2"] = BiLeapHandGraspV2
# isaacgym_task_map["BiLeapHandGraspPCD"] = BiLeapHandGraspPCD
isaacgym_task_map["BiLeapHandGraspVision"] = BiLeapHandGraspVision
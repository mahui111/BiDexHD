from isaacgymenvs.tasks import isaacgym_task_map
from .shadow_hand_grasp import ShadowHandGrasp
from .leap_hand_grasp import LeapHandGrasp
from .bi_leap_hand_grasp import BiLeapHandGrasp

isaacgym_task_map["ShadowHandGrasp"] = ShadowHandGrasp
isaacgym_task_map["LeapHandGrasp"] = LeapHandGrasp
isaacgym_task_map["BiLeapHandGrasp"] = BiLeapHandGrasp
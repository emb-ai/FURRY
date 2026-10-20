"""Custom G1/HOMIE command embodiment, not the NVIDIA SONIC controller."""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["commands"]),
    "action": ModalityConfig(delta_indices=list(range(40)), modality_keys=["commands"], action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE,type=ActionType.NON_EEF,format=ActionFormat.DEFAULT)]),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}
register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

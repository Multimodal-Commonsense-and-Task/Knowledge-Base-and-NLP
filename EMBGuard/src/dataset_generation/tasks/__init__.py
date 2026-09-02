"""
Task implementations for dataset generation pipeline.
Each task is defined in its own module.
"""
from src.dataset_generation.tasks.taxonomy_to_scenario import TaxonomyToScenarioTask
from src.dataset_generation.tasks.scenario_to_graph import ScenarioToGraphTask
from src.dataset_generation.tasks.graph_to_text import GraphToTextTask
from src.dataset_generation.tasks.text_to_image import TextToImageTask
from src.dataset_generation.tasks.scene_normalization import SceneNormalizationTask
from src.dataset_generation.tasks.scene_augmentation import SceneAugmentationTask
from src.dataset_generation.tasks.hazard_removal import HazardRemovalTask
from src.dataset_generation.tasks.hazard_augmentation import HazardAugmentationTask
from src.dataset_generation.tasks.action_augmentation import ActionAugmentationTask
from src.dataset_generation.tasks.embguard_train_data_construction import EMBGuardTrainDataConstructionTask

__all__ = [
    "TaxonomyToScenarioTask",
    "ScenarioToGraphTask",
    "GraphToTextTask",
    "TextToImageTask",
    "SceneNormalizationTask",
    "SceneAugmentationTask",
    "HazardRemovalTask",
    "HazardAugmentationTask",
    "ActionAugmentationTask",
    "EMBGuardTrainDataConstructionTask",
]


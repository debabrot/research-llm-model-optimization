"""Tree Surgery: can an LLM improve a trained LightGBM model by editing its leaf values?"""

from .arms import ARMS, ArmContext, ArmOutput, BudgetError, build_context, get_arm
from .datasets import SplitData, anon_mapping, load_dataset
from .design_matrix import LeafIndex, delta_vector, leaf_design_matrix, leaf_values
from .inspector import extract_rules, leaf_table, render_leaf_line, render_leaf_table
from .llm_client import CachedClient, LLMResponse, make_client
from .optimizers import fit_ridge_select_lambda, lasso_supports, omp_path, ridge_on_subset
from .patcher import Candidate, EditError, LeafEdit, ModelPatcher, load_candidate
from .train import fit_base_model, load_model, train_base_model

__all__ = [
    "CachedClient",
    "LLMResponse",
    "make_client",
    "ARMS",
    "ArmContext",
    "ArmOutput",
    "BudgetError",
    "build_context",
    "get_arm",
    "fit_ridge_select_lambda",
    "lasso_supports",
    "omp_path",
    "ridge_on_subset",
    "SplitData",
    "anon_mapping",
    "load_dataset",
    "LeafIndex",
    "delta_vector",
    "leaf_design_matrix",
    "leaf_values",
    "extract_rules",
    "leaf_table",
    "render_leaf_line",
    "render_leaf_table",
    "Candidate",
    "EditError",
    "LeafEdit",
    "ModelPatcher",
    "load_candidate",
    "fit_base_model",
    "load_model",
    "train_base_model",
]

__version__ = "0.3.0"

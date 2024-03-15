from __future__ import annotations

from torch.nn import LayerNorm, SiLU

from matsciml.datasets.utils import element_types
from matsciml.models import (
    FAENet,
    M3GNet,
    MEGNet,
    PLEGNNBackbone,
    TensorNet,
    GalaPotential,
)

available_models = {
    "egnn": {
        "encoder_class": PLEGNNBackbone,
        "encoder_kwargs": {
            "embed_in_dim": 1,
            "embed_hidden_dim": 32,
            "embed_out_dim": 128,
            "embed_depth": 5,
            "embed_feat_dims": [128, 128, 128],
            "embed_message_dims": [128, 128, 128],
            "embed_position_dims": [64, 64],
            "embed_edge_attributes_dim": 0,
            "embed_activation": "relu",
            "embed_residual": True,
            "embed_normalize": True,
            "embed_tanh": True,
            "embed_activate_last": False,
            "embed_k_linears": 1,
            "embed_use_attention": False,
            "embed_attention_norm": "sigmoid",
            "readout": "sum",
            "node_projection_depth": 3,
            "node_projection_hidden_dim": 128,
            "node_projection_activation": "relu",
            "prediction_out_dim": 1,
            "prediction_depth": 3,
            "prediction_hidden_dim": 128,
            "prediction_activation": "relu",
        },
        "output_kwargs": {
            "norm": LayerNorm(128),
            "hidden_dim": 128,
            "activation": SiLU,
            "lazy": False,
            "input_dim": 128,
        },
    },
    "faenet": {
        "encoder_class": FAENet,
        "encoder_kwargs": {
            "average_frame_embeddings": True,
            "pred_as_dict": False,
            "hidden_dim": 128,
            "out_dim": 128,
            "tag_hidden_channels": 0,
        },
        "output_kwargs": {"lazy": False, "input_dim": 128, "hidden_dim": 128},
    },
    "gala": {
        "encoder_class": GalaPotential,
        "encoder_kwargs": {
            "D_in": 100,
            "depth": 2,
            "hidden_dim": 64,
            "merge_fun": "concat",
            "join_fun": "concat",
            "invariant_mode": "full",
            "covariant_mode": "full",
            "include_normalized_products": True,
            "invar_value_normalization": "momentum",
            "eqvar_value_normalization": "momentum_layer",
            "value_normalization": "layer",
            "score_normalization": "layer",
            "block_normalization": "layer",
            "equivariant_attention": False,
            "tied_attention": True,
            "encoder_only": True,
        },
        "output_kwargs": {"lazy": False, "input_dim": 64, "hidden_dim": 64},
    },
    "m3gnet": {
        "encoder_class": M3GNet,
        "encoder_kwargs": {
            "element_types": element_types(),
            "return_all_layer_output": True,
        },
        "output_kwargs": {"lazy": False, "input_dim": 64, "hidden_dim": 64},
    },
    "megnet": {
        "encoder_class": MEGNet,
        "encoder_kwargs": {
            "edge_feat_dim": 2,
            "node_feat_dim": 128,
            "graph_feat_dim": 9,
            "num_blocks": 4,
            "hiddens": [256, 256, 128],
            "conv_hiddens": [128, 128, 128],
            "s2s_num_layers": 5,
            "s2s_num_iters": 4,
            "output_hiddens": [64, 64],
            "is_classification": False,
            "encoder_only": True,
        },
        "output_kwargs": {"lazy": False, "input_dim": 640, "hidden_dim": 640},
    },
    "tensornet": {
        "encoder_class": TensorNet,
        "encoder_kwargs": {
            "element_types": element_types(),
            "num_rbf": 32,
            "max_n": 3,
            "max_l": 3,
            # "units": 64,
            
        },
        # element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        # units: int = 64,
        # ntypes_state: int | None = None,
        # dim_state_embedding: int = 0,
        # dim_state_feats: int | None = None,
        # include_state: bool = False,
        # nblocks: int = 2,
        # num_rbf: int = 32,
        # max_n: int = 3,
        # max_l: int = 3,
        # rbf_type: Literal["Gaussian", "SphericalBessel"] = "Gaussian",
        # use_smooth: bool = False,
        # activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",
        # cutoff: float = 5.0,
        # equivariance_invariance_group: str = "O(3)",
        # dtype: torch.dtype = matgl.float_th,
        # width: float = 0.5,
        # readout_type: Literal["set2set", "weighted_atom", "reduce_atom"] = "weighted_atom",
        # task_type: Literal["classification", "regression"] = "regression",
        # niters_set2set: int = 3,
        # nlayers_set2set: int = 3,
        # field: Literal["node_feat", "edge_feat"] = "node_feat",
        # is_intensive: bool = True,
        # ntargets: int = 1,
        "output_kwargs": {"lazy": False, "input_dim": 64, "hidden_dim": 64*3},
        "lr": 0.0001,
    },
    "generic": {
        "output_kwargs": {
            "norm": LayerNorm(128),
            "hidden_dim": 128,
            "activation": "SiLU",
            "lazy": False,
            "input_dim": 128,
        },
        "lr": 0.0001,
    },
}
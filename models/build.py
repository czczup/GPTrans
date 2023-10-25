from .gptrans import GPTrans


def build_model(config):
    model_type = config.MODEL.TYPE
    
    if model_type == 'GPTrans':
        model = GPTrans(
            num_layers=config.MODEL.GPTRANS.NUM_LAYERS,
            num_heads=config.MODEL.GPTRANS.NUM_HEADS,
            node_dim=config.MODEL.GPTRANS.NODE_DIM,
            edge_dim=config.MODEL.GPTRANS.EDGE_DIM,
            layer_scale=config.MODEL.GPTRANS.LAYER_SCALE,
            mlp_ratio=config.MODEL.GPTRANS.MLP_RATIO,
            num_classes=config.MODEL.GPTRANS.NUM_CLASSES,
            drop_rate=config.MODEL.DROP_RATE,
            attn_drop_rate=config.MODEL.ATTN_DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            num_atoms=config.DATA.NUM_ATOMS,
            num_edges=config.DATA.NUM_EDGES,
            num_in_degree=config.DATA.NUM_IN_DEGREE,
            num_out_degree=config.DATA.NUM_OUT_DEGREE,
            num_spatial=config.DATA.NUM_SPATIAL,
            num_edge_dist=config.DATA.NUM_EDGE_DIST,
            multi_hop_max_dist=config.DATA.MULTI_HOP_MAX_DIST,
            edge_type=config.DATA.EDGE_TYPE,
            task_type=config.DATA.TASK_TYPE,
            with_cp=config.TRAIN.USE_CHECKPOINT,
            random_feature=config.AUG.RANDOM_FEATURE,
        )
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")

    return model

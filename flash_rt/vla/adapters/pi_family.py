"""Checkpoint action decoding and native graph ownership shared by Pi adapters."""


def decoder(config):
    from ..processing import load_action_decoder
    return load_action_decoder(config.checkpoint)


def release(frontend):
    from ..resources import release_pi_frontend_graphs
    release_pi_frontend_graphs(frontend)


def release_retired(pipelines):
    from ..resources import release_pi_pipeline_graphs
    release_pi_pipeline_graphs(pipelines)

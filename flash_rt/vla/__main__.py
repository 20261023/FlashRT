"""Describe an execution candidate. This command never loads a model."""

import argparse
import json

from flash_rt.sampling import ProbeFlowConfig
from .config import BackendConfig
from .registry import describe_backend


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True,
                        help='pi0, pi05, fastwam or hyvla')
    parser.add_argument('--checkpoint', required=True, help='Local checkpoint path')
    parser.add_argument('--sampler', choices=('euler', 'probeflow'), default='euler')
    parser.add_argument('--hardware', default='auto')
    parser.add_argument('--framework', choices=('torch', 'jax'), default='torch')
    parser.add_argument('--num-steps', type=int)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--probe-horizon', type=float)
    parser.add_argument('--max-steps', type=int)
    args = parser.parse_args(argv)
    policy = None
    explicit_policy = {name: value for name, value in (
        ('epsilon', args.epsilon), ('probe_horizon', args.probe_horizon),
        ('max_steps', args.max_steps)) if value is not None}
    if args.sampler == 'euler' and explicit_policy:
        parser.error('--epsilon/--probe-horizon/--max-steps require --sampler probeflow')
    if args.sampler == 'probeflow':
        policy = ProbeFlowConfig(**explicit_policy)
    config = BackendConfig(model=args.model, checkpoint=args.checkpoint,
                           sampler=args.sampler, hardware=args.hardware,
                           framework=args.framework, num_steps=args.num_steps,
                           probeflow=policy)
    print(json.dumps(describe_backend(config), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

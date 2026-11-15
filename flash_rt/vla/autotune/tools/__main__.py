"""Tool registry CLI: list, show, propose and price levers.

  python -m flash_rt.vla.autotune.tools list [--model M] [--hardware H] [--stage S] [--lane L] [--all] [--markdown]
  python -m flash_rt.vla.autotune.tools show TOOL_ID
  python -m flash_rt.vla.autotune.tools propose --baseline B.json --tool T [--param k=JSON ...] --out P.json
  python -m flash_rt.vla.autotune.tools levers --baseline B.json --lever name=JSON ... [--stage S] --out P.json

``propose``/``levers`` write a JSON list of Candidate records, the format of
``controller init --proposals``. Parameter and lever values are JSON
(strings need quotes: --param 'modes=["fixed"]'). Invalid input exits 2 with
``error: ...`` on stderr.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from ..schema import Candidate
from . import get_tool, list_tools, propose
from .external import ADAPTERS, external_specs
from .levers import price_levers


def _key_value(text, flag):
    name, separator, value = text.partition('=')
    if not separator or not name:
        raise ValueError(f'{flag} expects name=JSON, got {text!r}')
    try:
        return name, json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f'{flag} {name}: value is not JSON ({error.msg}); quote strings') from None


def _pairs(items, flag):
    result = {}
    for item in items or ():
        name, value = _key_value(item, flag)
        if name in result:
            raise ValueError(f'{flag} {name} given twice')
        result[name] = value
    return result


def _baseline(path):
    return Candidate.from_dict(json.loads(Path(path).read_text()))


def _write_candidates(path, candidates):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps([candidate.to_dict() for candidate in candidates], indent=2, allow_nan=False) + '\n'
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return {'out': str(path), 'count': len(candidates),
            'candidate_ids': [candidate.candidate_id for candidate in candidates]}


def markdown_table(tools, model=None) -> str:
    lines = ['| id | lane | stages | models | exactness | status | reason |',
             '|---|---|---|---|---|---|---|']
    for tool in tools:
        reason = tool.reason_for(model).replace('|', '/').replace('\n', ' ')
        lines.append(f'| {tool.tool_id} | {tool.lane} | {", ".join(tool.stages)} | '
                     f'{", ".join(tool.models)} | {tool.exactness} | {tool.status_for(model)} | {reason} |')
    return '\n'.join(lines)


def _show(tool_id):
    tool = get_tool(tool_id)
    record = tool.to_dict()
    if tool_id in ADAPTERS:
        spec = external_specs()[tool_id]
        record.update(setup=list(spec['setup']), risks=list(spec['risks']))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m flash_rt.vla.autotune.tools', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('list')
    listing.add_argument('--model')
    listing.add_argument('--hardware')
    listing.add_argument('--stage')
    listing.add_argument('--lane')
    listing.add_argument('--all', action='store_true', help='Include tools that are not ready')
    listing.add_argument('--markdown', action='store_true')
    show = commands.add_parser('show')
    show.add_argument('tool_id')
    proposing = commands.add_parser('propose')
    proposing.add_argument('--baseline', required=True)
    proposing.add_argument('--tool', required=True)
    proposing.add_argument('--param', action='append', default=[])
    proposing.add_argument('--out', required=True)
    levers = commands.add_parser('levers')
    levers.add_argument('--baseline', required=True)
    levers.add_argument('--lever', action='append', default=[], required=True)
    levers.add_argument('--stage')
    levers.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'list':
            tools = list_tools(model=args.model, hardware=args.hardware, stage=args.stage,
                               lane=args.lane, include_unavailable=args.all)
            if args.markdown:
                print(markdown_table(tools, args.model))
                return 0
            output = [tool.to_dict() for tool in tools]
        elif args.command == 'show':
            output = _show(args.tool_id)
        elif args.command == 'propose':
            output = _write_candidates(args.out, propose(args.tool, _baseline(args.baseline),
                                                         **_pairs(args.param, '--param')))
        else:
            output = _write_candidates(args.out, price_levers(_baseline(args.baseline),
                                                              _pairs(args.lever, '--lever'), stage=args.stage))
    except (OSError, TypeError, ValueError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

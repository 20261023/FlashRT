"""Full conditional Expert replay with a LIVEinput-derived RMS source model."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path


def _sibling(name):
    s=importlib.util.spec_from_file_location('hy_expert_tree_'+name,Path(__file__).with_name(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


mt=_sibling('mean_tree');rms=mt.rms;order=_sibling('expert_order_replay')


def tree_rms(x,w,epsilon=1e-5):
    import torch
    if (epsilon!=1e-5 or x.ndim<2 or x.dtype not in (torch.bfloat16,torch.float32)
            or w.dtype!=torch.bfloat16 or w.shape!=(x.shape[-1],) or x.device!=w.device
            or not torch.isfinite(x).all() or not torch.isfinite(w).all()):
        raise ValueError('Source-tree RMS requires admitted dtype/weight/finite input and fixed epsilon')
    xf=x.float();variance=mt.tree_mean(xf.pow(2),vectorized=x.shape[-1]==1024,descending=False)
    return w*(xf*torch.rsqrt(variance+epsilon)).to(x.dtype)


@contextmanager
def norm_scope(owner,*,use_tree,expected_calls):
    import torch
    if type(use_tree) is not bool or type(expected_calls) is not int or expected_calls<1:raise ValueError('RMS scope flags/call count invalid')
    original=owner._rms_norm;state={'calls':0,'by_role':{},'original_norm_calls':0,'complete':False,'restored':False,'use_tree':use_tree}
    def call(x,w):
        if state['calls']>=expected_calls:raise ValueError('extra RMS calls')
        if x.ndim<2 or x.dtype not in (torch.bfloat16,torch.float32) or x.shape[-1] not in (128,1024):raise ValueError('Unsupported RMS input role')
        role=str(x.dtype)+':'+str(x.shape[-1]);state['by_role'][role]=state['by_role'].get(role,0)+1;state['calls']+=1
        if use_tree:return tree_rms(x,w)
        state['original_norm_calls']+=1
        return original(x,w)
    try:
        owner._rms_norm=call;yield state
        if state['calls']!=expected_calls:raise ValueError('RMS scope call count incomplete')
        state['complete']=True
    finally:owner._rms_norm=original;state['restored']=True


def admit_support(report,raw,old,capture,*,factory):
    if (report.get('kind')!='hy_source_explicit_mean_tree_diagnostic' or report.get('status')!='diagnostic_complete'
            or report.get('job_id')!='316521' or report.get('source_revision')!='19e87bcccf875449e4d52fc4fa67566de8b19257'
            or report.get('golden_variance_used_as_input') is not False or report.get('model_constructed') is not False
            or report.get('checkpoint_tensors_read')!=0
            or report.get('raw_proof')!={'tensors':96,'native_baselines_exact':16,'squares_exact':16}
            or report.get('summary',{}).get('source_old_exact')!=16 or report.get('summary',{}).get('source_native_exact')!=16):
        raise ValueError('R31 input-derived mean support identity/header failed')
    mt.validate_raw(raw,capture)
    for probe in rms.PROBES:
        for n in range(4):
            name=f'{probe}.n{n}';arm='scalar_ascending' if capture[name+'.square'].shape[-1]==128 else 'vector_ascending'
            if (not mt.handoff._exact(raw[name+'.'+arm],old['own.'+name+'.stage.variance'])
                    or not mt.handoff._exact(raw[name+'.vector_descending'],capture[name+'.unprofiled_variance'])):
                raise ValueError('R31 actual selected mean support differs before owner creation')
    return factory()


def baseline_gate(current,reference):
    order.validate_raw(current);order.validate_raw(reference)
    return rms.layers.verify_replay_arrays(current,reference)


def compare_outputs(current,reference):
    order.validate_raw(current);order.validate_raw(reference)
    metrics={k:mt.handoff._difference(v,reference[k]) for k,v in current.items()}
    velocities=[metrics[f'propagated.step{s}.velocity'] for s in range(10)]
    boundary={mode+'.'+field:sum(metrics[f'{mode}.step{s}.layer{layer}.{field}']['exact'] for s in range(10) for layer in range(32))
        for mode in ('teacher','propagated') for field in ('residual','output')}
    return metrics,{'exact_tensors':sum(v['exact'] for v in metrics.values()),'tensors':len(metrics),
        'boundary_exact':boundary,'velocity_exact_steps':sum(v['exact'] for v in velocities),
        'gold_head_exact_steps':sum(metrics[f'gold_head.step{s}.velocity']['exact'] for s in range(10)),
        'velocity_relative_l2_min':min(v['relative_l2'] for v in velocities),'velocity_relative_l2_max':max(v['relative_l2'] for v in velocities)}


def _oracle_header(hdr,*,engine,variant,checkpoint,runtime):
    if (hdr.get('kind')!='hy_split_projection_order_replay' or hdr.get('status')!='original_order_observed'
            or hdr.get('job_id')!='315024' or hdr.get('source_revision')!='5248e14b544aa8edc17b58f1b61186b5c08ca492'
            or hdr.get('engine')!=engine or hdr.get('variant')!=variant or hdr.get('projection')!='separate'
            or hdr.get('attention_order')!='original' or hdr.get('checkpoint')!=checkpoint or hdr.get('runtime')!=runtime
            or hdr.get('weights_loaded')!=363 or set(hdr.get('selected_checkpoint_keys',[]))!=set(rms.layers.all_selected_keys())
            or set(hdr.get('source_records',{}))!={'driver','core','native','adapter','suffix'}):
        raise ValueError('R25 oracle role/runtime/checkpoint/source admission failed')


def load_cases(ro,rn,*,task):
    from safetensors.torch import load_file
    import torch
    block=_sibling('expert_rms_block_handoff');results=task/'results'
    m,paths,_=block.admitted_input_owner(rn,results_root=results,factory=lambda:None)
    expected_root=results/'hy-split-order-315024'
    order.validate_inputs(m,job_id='315024',source_revision='5248e14b544aa8edc17b58f1b61186b5c08ca492',engine='native',arm='original',output=expected_root/'native-original')
    if ro['historical_block_reference']['manifest']!=rn['historical_block_reference']['manifest'] or Path(m['checkpoint']).resolve()!=Path(ro['checkpoint']['path']).parent:raise ValueError('Historical manifest/checkpoint pairing differs')
    cases={};oracles={'native':{},'official':{}};records={}
    frozen=_sibling('frozen_eager_attention');capture=_sibling('expert_layers_capture')
    layer_source=Path(m['official_attention_source']).parent/'hunyuan_vl_mot/modeling_hunyuan_vl_mot.py'
    for variant in ('baseline','state'):
        headers={};records[variant]={}
        for engine,parent in (('native',rn),('official',ro)):
            bound=parent['historical_block_reference']['records'][variant]
            for role,filename in (('report','result.json'),('raw','operator_outputs.safetensors')):
                expected=expected_root/(engine+'-original')/variant/filename
                if Path(bound[role]['path']).resolve()!=expected:raise ValueError('R25 oracle report/raw path binding differs')
                rms._record_files_unchanged({role:bound[role]},results)
            hdr=json.loads(Path(bound['report']['path']).read_text());headers[engine]=hdr
            _oracle_header(hdr,engine=engine,variant=variant,checkpoint=ro['checkpoint'],runtime=parent['runtime'])
            rms._record_files_unchanged(hdr['source_records'],task/'snapshots')
            oracle=load_file(bound['raw']['path']);order.validate_raw(oracle);oracles[engine][variant]=oracle
            records[variant][engine]=bound
        actual_paths={'manifest':Path(rn['historical_block_reference']['manifest']['path']),
            'gold_report':Path(m['gold']['root'])/'official'/variant/'result.json','gold_trace':paths[variant]['gold_trace'],
            'native_report':Path(m['native']['root'])/variant/'result.json','native_trace':paths[variant]['native_trace'],
            'native_frozen':Path(m['native']['root'])/variant/'frozen_velocities.safetensors'}
        for hdr in headers.values():
            input_records=hdr.get('input_records',{})
            if not set(actual_paths)<=set(input_records) or any(Path(input_records[k]['path']).resolve()!=p.resolve() for k,p in actual_paths.items()):raise ValueError('Actual replay input role/path binding differs')
            rms._record_files_unchanged(input_records,results)
        gp=json.loads(actual_paths['gold_report'].read_text());nr=json.loads(actual_paths['native_report'].read_text())
        frozen.admit_capture_report(gp,job_id=m['gold']['job_id'],source_revision=m['gold']['source_revision'],variant=variant,attention_source=m['official_attention_source'])
        rms.layers.admit_layer_capture(gp,checkpoint=ro['checkpoint'],layer_source=layer_source)
        rms.layers.admit_baseline_header(nr,job_id=m['native']['job_id'],source_revision=m['native']['source_revision'],checkpoint=ro['checkpoint'])
        if nr['runtime']!=rn['runtime']:raise ValueError('Actual native input runtime differs')
        gold=load_file(str(actual_paths['gold_trace']));trace=load_file(str(actual_paths['native_trace']));prior=load_file(str(actual_paths['native_frozen']))
        capture.verify_boundary_chain(gold,first_suffix=gold['suffix.embeddings'])
        for key in ('prepared.images','prepared.state','prepared.tokens','prepared.token_mask','prefix.pad_mask','prefix.modality_mask','suffix.state_embedding'):
            if not mt.handoff._exact(gold[key],trace[key]):raise ValueError('Prepared Gold/native boundary pairing differs')
        mask=gold['prefix.modality_mask'][0]
        if mask.dtype!=torch.bool or mask.numel()!=240:raise ValueError('Expected240prefix modality rows')
        perm=torch.cat((mask.nonzero().flatten(),(~mask).nonzero().flatten())).numpy()
        _sibling('suffix_key_order').admit_original_mask(trace['suffix.mask_sorted'],gold['suffix.attention_mask'],perm)
        cases[variant]={'gold':gold,'trace':trace,'frozen':prior,'perm':perm}
    return cases,oracles,records


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--history-root',type=Path,required=True);p.add_argument('--support-root',type=Path,required=True)
    p.add_argument('--source-revision',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);rms._sibling('diagnose').require_step()
    import platform,sys
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    if tuple(torch.cuda.get_device_capability())!=(8,0) or 'A800' not in torch.cuda.get_device_name():raise RuntimeError('Expert tree replay requires recorded A800')
    if not torch.__version__.startswith('2.11.0'):raise RuntimeError('Expert tree replay requires native2.11')
    if len(args.source_revision)!=40 or any(c not in '0123456789abcdef' for c in args.source_revision):raise ValueError('Exact current source revision required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=args.history_root.resolve();task=root.parent.parent;support=args.support_root.resolve()
    if root!=task/'results/hy-rms-stages-315256' or support!=task/'results/hy-mean-tree-316521':raise ValueError('Expert tree reference roots differ')
    if (root/'SOURCE_GIT_COMMIT').read_text().strip()!='b522e8fd04616c22b680a32a3c183a566c4088a4' or (support/'SOURCE_GIT_COMMIT').read_text().strip()!='19e87bcccf875449e4d52fc4fa67566de8b19257':raise ValueError('Expert tree reference markers differ')
    ro=json.loads((root/'official/result.json').read_text());rn=json.loads((root/'native/result.json').read_text())
    old=load_file(str(root/'official/operator_outputs.safetensors'));native=load_file(str(root/'native/operator_outputs.safetensors'))
    rms_proof=mt.handoff.admit_history(ro,old,rn,native,job_id='315256',source_revision='b522e8fd04616c22b680a32a3c183a566c4088a4')
    rms._record_files_unchanged(ro['source_records'],task/'snapshots')
    if rms.layers._file_record(ro['checkpoint']['path'])!=ro['checkpoint']:raise ValueError('Checkpoint stat changed')
    sr=json.loads((support/'native/result.json').read_text());sa=load_file(str(support/'native/operator_outputs.safetensors'))
    if set(sr.get('source_records',{}))!={'driver','launch','stages','history'}:raise ValueError('R31 source roles differ')
    rms._record_files_unchanged(sr['source_records'],task/'snapshots/round031/code')
    capture_root=task/'results/hy-mean-launch-315622';cap_bound=sr['capture_reference']
    if cap_bound['job_id']!='315622' or cap_bound['source_revision']!='071ea36442cefb0b394f676cd562eb5a52f8e8eb' or Path(cap_bound['root']).resolve()!=capture_root:raise ValueError('R30 capture identity differs')
    for role,filename in (('report','result.json'),('raw','operator_outputs.safetensors')):
        if Path(cap_bound['records'][role]['path']).resolve()!=capture_root/'native'/filename:raise ValueError('R30 capture file role binding differs')
    rms._record_files_unchanged(cap_bound['records'],task/'results')
    cr=json.loads((capture_root/'native/result.json').read_text());mt.admit_capture(cr)
    ca=load_file(str(capture_root/'native/operator_outputs.safetensors'));mt.launch.validate_raw(ca,old,native,engine='native')
    runtime={'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}
    if any(r['runtime']!=runtime or r['torch_git_version']!=torch.version.git_version or r['bf16_reduced_precision_reduction']!=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction for r in (rn,sr,cr)):raise ValueError('Expert tree runtime/Git/flags differ')
    cases,oracles,input_records=load_cases(ro,rn,task=task)
    here=Path(__file__).resolve();native_source=here.parents[3]/'flash_rt/models/hyvla/pipeline_thor.py'
    for current,record in ((Path(mt.__file__),sr['source_records']['driver']),(native_source,rn['source_records']['native']),(Path(rms.layers.__file__),rn['source_records']['layers'])):
        if current.read_bytes()!=Path(record['path']).read_bytes():raise ValueError('Tree/native/core source changed from admitted reference')
    owner=admit_support(sr,sa,old,ca,factory=lambda:rms.layers.load_native_owner(native_source,projection='separate'))
    arrays={};arms={};keys=set(rms.layers.all_selected_keys())
    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),safe_open(ro['checkpoint']['path'],framework='pt',device='cpu') as weights:
        for arm,use_tree in (('control',False),('tree',True)):
            with norm_scope(owner,use_tree=use_tree,expected_calls=5160) as state:
                result=rms.layers.replay_core(owner,cases,weights.get_tensor,attention_order='original')
            if state['by_role']!={'torch.bfloat16:1024':2600,'torch.float32:128':2560} or state['original_norm_calls']!=(0 if use_tree else 5160):raise ValueError('Actual full Expert norm role counts differ')
            if len(result['selected_checkpoint_keys'])!=363 or set(result['selected_checkpoint_keys'])!=keys:raise ValueError('Actual Expert selected keys differ')
            summaries={};metrics={};proofs={};case_proofs={}
            for variant,data in result['cases'].items():
                if data['h0_exact']!=10 or data['attention_calls']!=640 or len(data['order_contexts'])!=32 or any(c['cleanup']['passed'] is not True for c in data['order_contexts']):raise ValueError('Actual replay H0/order/cleanup proof differs')
                outputs=data['outputs'];order.validate_raw(outputs)
                if arm=='control':proofs[variant]=baseline_gate(outputs,oracles['native'][variant])
                metrics[variant],summaries[variant]=compare_outputs(outputs,oracles['official'][variant])
                arrays.update({arm+'.'+variant+'.'+k:v for k,v in outputs.items()})
                case_proofs[variant]={k:data[k] for k in ('h0_exact','attention_calls','order_contexts')}
            arms[arm]={'norm_scope':state,'case_proofs':case_proofs,'native_baseline_proofs':proofs,'old_comparison':metrics,'summary':summaries,
                'selected_checkpoint_keys':result['selected_checkpoint_keys']}
    if len(arrays)!=5200:raise ValueError('Full Expert raw role count differs')
    report={'kind':'hy_live_source_tree_RMS_full_Expert_replay','status':'diagnostic_complete','job_id':os.environ['SLURM_JOB_ID'],'source_revision':args.source_revision,
        'historical_rms_reference':{'job_id':'315256','source_revision':'b522e8fd04616c22b680a32a3c183a566c4088a4','root':str(root),'proof':rms_proof},
        'source_tree_support':{'job_id':'316521','source_revision':'19e87bcccf875449e4d52fc4fa67566de8b19257','root':str(support)},'input_records':input_records,
        'runtime':runtime,'torch_git_version':torch.version.git_version,'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'checkpoint':ro['checkpoint'],'arms':arms,'raw_tensors':len(arrays),'weight_read_events':726,'model_constructed':False,'golden_norm_data_used':False,
        'production_norm_replaced':False,'performance_measurement':False,'source_records':{n:rms.layers._file_record(f) for n,f in (('driver',here),('tree',Path(mt.__file__)),('core',Path(rms.layers.__file__)),('native',native_source))},
        'scope':'Full32layer Expert/heads under GoldKV and fixedexternaltrajectory only; finalnorm participates, Goldhead notunchangednegative. No nativevision/free-run/sevenfixture/robot qualification. Historical6/7FAILED unchanged.'}
    args.output.mkdir(parents=True,exist_ok=False);rms.publish(args.output,arrays,report)
    print(json.dumps({'raw_tensors':len(arrays),'old_comparison':{a:v['summary'] for a,v in arms.items()},'golden_norm_data_used':False}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

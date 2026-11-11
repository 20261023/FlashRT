"""Separate-QKV order interaction with typed historical and current controls."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re


def _sibling(name):
    s=importlib.util.spec_from_file_location('hy_split_order_'+name,Path(__file__).with_name(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


layers=_sibling('expert_layers')
LEGACY_JOB='314589'
LEGACY_SOURCE='eddf597d58d29653bf78233b92c8540a2d623d9e'


def expected_shapes():
    return {**{f'{mode}.step{s}.layer{i}.{field}':(1,41,1024) for mode in ('teacher','propagated')
               for s in range(10) for i in range(32) for field in ('residual','output')},
            **{f'{mode}.step{s}.velocity':(1,40,32) for mode in ('propagated','gold_head') for s in range(10)}}


def validate_raw(arrays):
    import torch
    expected=expected_shapes()
    if set(arrays)!=set(expected) or any(not isinstance(arrays[k],torch.Tensor) or arrays[k].dtype!=torch.bfloat16
            or tuple(arrays[k].shape)!=shape or not torch.isfinite(arrays[k]).all() for k,shape in expected.items()):
        raise ValueError('Order replay raw tensor set/shape/finite BF16 contract differs')
    return {'exact':True,'tensors':len(arrays)}


def _keys_exact(report):
    keys=report.get('selected_checkpoint_keys',[])
    return len(keys)==363 and set(keys)==set(layers.all_selected_keys())


def admit_historical_split(report,*,job_id,source_revision,engine,variant,checkpoint,runtime,gold_job,gold_source,fused_path):
    order=report.get('attention_order')
    if order is None and (job_id,source_revision)==(LEGACY_JOB,LEGACY_SOURCE):order='sorted'
    ref=report.get('reference_producer',{});control=report.get('projection_control') or {};gate=report.get('historical_native_baseline') or {}
    if (report.get('kind')!='recorded_Expert_layer_local_and_propagated_replay_not_policy_gate'
            or report.get('status')!='diagnostic_complete' or report.get('job_id')!=job_id or report.get('source_revision')!=source_revision
            or report.get('engine')!=engine or report.get('variant')!=variant or report.get('runtime')!=runtime
            or report.get('checkpoint')!=checkpoint or report.get('projection')!='separate' or order!='sorted'
            or report.get('model_constructed') is not False or report.get('weights_loaded')!=363 or not _keys_exact(report)
            or report.get('gold_h0_exact_steps')!=10 or ref.get('job_id')!=gold_job or ref.get('source_revision')!=gold_source
            or control.get('job_id')!=job_id or control.get('source_revision')!=source_revision or control.get('fused1300exact') is not True
            or Path(control.get('path','')).resolve()!=Path(fused_path).resolve()
            or gate.get('required') is not False or gate.get('passed') is not True):
        raise ValueError('Historical split-sorted role/source/runtime/case admission failed')


def admit_current_sorted(report,*,job_id,source_revision,engine,variant,checkpoint,runtime,sources,input_records,history_job,history_source):
    proof=report.get('historical_split_sorted') or {};preflight=report.get('adapter_preflight') or {}
    if (report.get('kind')!='hy_split_projection_order_replay' or report.get('status')!='sorted_split_reproduced'
            or report.get('artifact_complete') is not True or report.get('job_id')!=job_id or report.get('source_revision')!=source_revision
            or report.get('engine')!=engine or report.get('variant')!=variant or report.get('runtime')!=runtime
            or report.get('checkpoint')!=checkpoint or report.get('projection')!='separate' or report.get('attention_order')!='sorted'
            or report.get('model_constructed') is not False or report.get('weights_loaded')!=363 or not _keys_exact(report)
            or report.get('gold_h0_exact_steps')!=10 or report.get('attention_calls')!=640 or report.get('source_records')!=sources
            or report.get('input_records')!=input_records
            or proof.get('exact') is not True or proof.get('tensors')!=1300 or proof.get('job_id')!=history_job
            or proof.get('source_revision')!=history_source or preflight.get('exact') is not True or preflight.get('steps')!=10):
        raise ValueError('Current split-sorted control identity/source/raw-proof admission failed')


def sorted_gate(current,historical):
    validate_raw(current);validate_raw(historical)
    try:return layers.verify_replay_arrays(current,historical)
    except ValueError as error:return {'exact':False,'tensors':len(current),'reason':str(error)}


def admitted_owner(current,historical,factory):
    if set(current)!=set(historical):raise ValueError('Current control case set differs')
    for case in current:
        if not sorted_gate(current[case],historical[case])['exact']:
            raise ValueError('Current sorted control tensor replay differs before owner compilation')
    return factory()


def validate_inputs(m,*,job_id,source_revision,engine,arm,output):
    expected={'schema_version','job_id','source_revision','current_root','snapshots_root','checkpoint','official_attention_source','gold','qkv','native'}
    if set(m)!=expected or m['schema_version']!=1 or m['job_id']!=job_id or m['source_revision']!=source_revision:
        raise ValueError('Fixed order manifest schema/job/source differs')
    current=Path(m['current_root']).resolve();results=current.parent;snapshots=Path(m['snapshots_root']).resolve()
    if (engine not in ('official','native') or arm not in ('sorted','original') or Path(output).resolve()!=current/(engine+'-'+arm)
            or snapshots!=results.parent/'snapshots' or not re.fullmatch('[0-9a-f]{40}',source_revision)):
        raise ValueError('Order manifest output/snapshot path or source differs')
    if not Path(m['official_attention_source']).resolve().is_relative_to(snapshots):raise ValueError('Official source escapes snapshot root')
    for name in ('gold','qkv','native'):
        r=m[name]
        if (set(r)!={'root','job_id','source_revision'} or not str(r['job_id']).isdigit()
                or not re.fullmatch('[0-9a-f]{40}',r['source_revision']) or not Path(r['root']).resolve().is_relative_to(results)):
            raise ValueError('Historical manifest path/root/job/source differs')


def adapter_preflight(owner,gold,metadata,perm,*,num_steps=10,device='cuda'):
    import torch
    frozen=_sibling('frozen_eager_attention');adapter=_sibling('suffix_key_order');outputs={}
    for step in range(num_steps):
        base=f'attention.step{step}.layer0.'
        q=frozen.restore_layout(gold[base+'q'],metadata[base+'q'],device=device).transpose(1,2)
        prefix=len(perm);rows=q.shape[2]
        columns=torch.cat((torch.as_tensor(perm,device=device,dtype=torch.long),torch.arange(prefix,prefix+rows,device=device)))
        pairs=[]
        for field in ('k','v'):
            value=gold[base+field].to(device).transpose(1,2).repeat_interleave(owner.n_heads//owner.n_kv,dim=1).contiguous()
            pairs.append(value.index_select(2,columns))
        mask=gold[base+'mask'].to(device)[:,None].index_select(-1,columns)
        inverse=adapter.inverse_columns(perm,suffix_rows=rows,device=device)
        result=adapter.attention_in_original_order(owner._attn,q,*pairs,mask,columns=inverse).transpose(1,2).reshape(1,rows,owner.q_dim)
        expected=gold[base+'output']
        if result.dtype!=expected.dtype or not torch.equal(result.detach().cpu(),expected):
            raise ValueError('Current order adapter preflight context differs from actual Gold')
        outputs[f'step{step}.context']=result.detach().cpu().contiguous().clone()
    return outputs,{'exact':True,'steps':num_steps,'weights_loaded':0,'model_constructed':False}


def publish_case(folder,outputs,preflight,report):
    from safetensors.torch import save_file
    save_file({k:v.contiguous().clone() for k,v in preflight.items()},str(folder/'adapter_preflight.safetensors'))
    layers.publish(folder,outputs,report)


def source_records():
    root=Path(__file__).resolve().parent
    return {name:layers._file_record(path) for name,path in (
        ('driver',__file__),('core',layers.__file__),('native',root.parents[2]/'flash_rt/models/hyvla/pipeline_thor.py'),
        ('adapter',root/'suffix_key_order.py'),('suffix',root/'frozen_suffix.py'))}


def check_old_sources(report,snapshot_root):
    for name in ('native_method_source','prototype_source'):
        record=report[name];path=Path(record['path']).resolve()
        if not path.is_relative_to(Path(snapshot_root).resolve()) or layers._file_record(path)!=record:
            raise ValueError('Historical source record changed or escapes snapshots')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--engine',choices=('official','native'),required=True);p.add_argument('--arm',choices=('sorted','original'),required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--source-revision',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);common=_sibling('diagnose');common.require_step()
    import platform,sys
    import torch
    import numpy as np
    from safetensors.torch import load_file
    from safetensors import safe_open
    m=json.loads(args.inputs.read_text());job=os.environ['SLURM_JOB_ID']
    validate_inputs(m,job_id=job,source_revision=args.source_revision,engine=args.engine,arm=args.arm,output=args.output)
    if tuple(torch.cuda.get_device_capability())!=(8,0) or 'A800' not in torch.cuda.get_device_name():raise RuntimeError('Order replay requires recorded A800')
    if not torch.__version__.startswith('2.7.1' if args.engine=='official' else '2.11.0'):raise RuntimeError('Order replay runtime differs')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    runtime={'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}
    checkpoint=_sibling('proj1_camera_ablation')._checkpoint_stat(m['checkpoint'])
    frozen=_sibling('frozen_eager_attention');capture=_sibling('expert_layers_capture');adapter=_sibling('suffix_key_order')
    sources=source_records();cases={};histories={};fused_headers={};metadata={};controls={};input_records={}
    gold_root=Path(m['gold']['root']);qkv_root=Path(m['qkv']['root']);native_root=Path(m['native']['root'])
    layer_source=Path(m['official_attention_source']).parent/'hunyuan_vl_mot/modeling_hunyuan_vl_mot.py'
    for variant in ('baseline','state'):
        gp=json.loads((gold_root/'official'/variant/'result.json').read_text())
        frozen.admit_capture_report(gp,job_id=m['gold']['job_id'],source_revision=m['gold']['source_revision'],variant=variant,attention_source=m['official_attention_source'])
        layers.admit_layer_capture(gp,checkpoint=checkpoint,layer_source=layer_source)
        gold=load_file(str(gold_root/'official'/variant/'trace.safetensors'));capture.verify_boundary_chain(gold,first_suffix=gold['suffix.embeddings'])
        nr=json.loads((native_root/variant/'result.json').read_text())
        layers.admit_baseline_header(nr,job_id=m['native']['job_id'],source_revision=m['native']['source_revision'],checkpoint=checkpoint)
        if args.engine=='native' and nr['runtime']!=runtime:raise ValueError('Native runtime differs from admitted history')
        trace=load_file(str(native_root/variant/'trace.safetensors'));prior=load_file(str(native_root/variant/'frozen_velocities.safetensors'))
        for key in ('prepared.images','prepared.state','prepared.tokens','prepared.token_mask','prefix.pad_mask','prefix.modality_mask','suffix.state_embedding'):
            if gold[key].dtype!=trace[key].dtype or not torch.equal(gold[key],trace[key]):raise ValueError('Historical prepared boundary differs')
        mm=gold['prefix.modality_mask'][0]
        if mm.dtype!=torch.bool or mm.numel()!=240:raise ValueError('Unexpected prefix modality geometry')
        perm=torch.cat((mm.nonzero().flatten(),(~mm).nonzero().flatten())).numpy()
        adapter.admit_original_mask(trace['suffix.mask_sorted'],gold['suffix.attention_mask'],perm)
        fused_folder=qkv_root/(args.engine+'-fused')/variant;split_folder=qkv_root/(args.engine+'-separate')/variant
        fr=json.loads((fused_folder/'result.json').read_text());sr=json.loads((split_folder/'result.json').read_text())
        lr=json.loads((gold_root/(args.engine+'-ops')/variant/'result.json').read_text())
        for r,j,s,proof in ((fr,m['qkv']['job_id'],m['qkv']['source_revision'],True),(lr,m['gold']['job_id'],m['gold']['source_revision'],False)):
            layers.admit_layer_replay_report(r,job_id=j,source_revision=s,engine=args.engine,variant=variant,checkpoint=checkpoint,runtime=runtime,require_layer_proof=proof)
            if set(r['selected_checkpoint_keys'])!=set(layers.all_selected_keys()):raise ValueError('Historical fused selected-key set differs')
            check_old_sources(r,m['snapshots_root'])
        admit_historical_split(sr,job_id=m['qkv']['job_id'],source_revision=m['qkv']['source_revision'],engine=args.engine,variant=variant,
            checkpoint=checkpoint,runtime=runtime,gold_job=m['gold']['job_id'],gold_source=m['gold']['source_revision'],fused_path=fused_folder)
        check_old_sources(sr,m['snapshots_root'])
        fa=load_file(str(fused_folder/'operator_outputs.safetensors'));sa=load_file(str(split_folder/'operator_outputs.safetensors'))
        la=load_file(str(gold_root/(args.engine+'-ops')/variant/'operator_outputs.safetensors'))
        for arrays in (fa,sa,la):validate_raw(arrays)
        layers.verify_replay_arrays(fa,la);layers.projection_effects(sa,fa,common=common)
        if args.engine=='native' and not layers.baseline_gate([fa[f'propagated.step{s}.velocity'] for s in range(10)],
                [prior[f'official_kv_sorted.step{s}.velocity'] for s in range(10)],engine='native')['passed']:
            raise ValueError('Historical fused velocity baseline differs')
        cases[variant]={'gold':gold,'trace':trace,'frozen':prior,'perm':perm};histories[variant]=sa;metadata[variant]=gp['tensor_metadata']
        paths={'manifest':args.inputs,'gold_report':gold_root/'official'/variant/'result.json',
            'gold_trace':gold_root/'official'/variant/'trace.safetensors','native_report':native_root/variant/'result.json',
            'native_trace':native_root/variant/'trace.safetensors','native_frozen':native_root/variant/'frozen_velocities.safetensors',
            'fused_report':fused_folder/'result.json','fused_raw':fused_folder/'operator_outputs.safetensors',
            'split_report':split_folder/'result.json','split_raw':split_folder/'operator_outputs.safetensors',
            'layer_report':gold_root/(args.engine+'-ops')/variant/'result.json',
            'layer_raw':gold_root/(args.engine+'-ops')/variant/'operator_outputs.safetensors'}
        input_records[variant]={name:layers._file_record(path) for name,path in paths.items()}
        fused_headers[variant]={'job_id':m['qkv']['job_id'],'source_revision':m['qkv']['source_revision'],'path':str(fused_folder),'raw1300exact':True}
        if args.arm=='original':
            folder=Path(m['current_root'])/(args.engine+'-sorted')/variant
            cr=json.loads((folder/'result.json').read_text())
            admit_current_sorted(cr,job_id=job,source_revision=args.source_revision,engine=args.engine,variant=variant,
                checkpoint=checkpoint,runtime=runtime,sources=sources,input_records=input_records[variant],
                history_job=m['qkv']['job_id'],history_source=m['qkv']['source_revision'])
            controls[variant]=load_file(str(folder/'operator_outputs.safetensors'));validate_raw(controls[variant])
    factory=lambda:layers.load_native_owner(sources['native']['path'],projection='separate')
    owner=admitted_owner(controls,histories,factory) if args.arm=='original' else factory()
    preflight={};proofs={}
    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
        for variant,case in cases.items():preflight[variant],proofs[variant]=adapter_preflight(owner,case['gold'],metadata[variant],case['perm'])
        with safe_open(checkpoint['path'],framework='pt',device='cpu') as weights:
            result=layers.replay_core(owner,cases,weights.get_tensor,attention_order=args.arm)
    if set(result['selected_checkpoint_keys'])!=set(layers.all_selected_keys()):raise ValueError('Actual selected weights differ')
    args.output.mkdir(parents=True,exist_ok=False);passed=True
    for variant,data in result['cases'].items():
        validate_raw(data['outputs'])
        proof=sorted_gate(data['outputs'],histories[variant]) if args.arm=='sorted' else {'exact':True,'tensors':1300}
        passed=passed and proof['exact'];effects=None
        if args.arm=='original':effects=layers.projection_effects(data['outputs'],controls[variant],common=common)
        report={'kind':'hy_split_projection_order_replay','status':('sorted_split_reproduced' if proof['exact'] else 'sorted_split_reproduction_failed') if args.arm=='sorted' else 'original_order_observed',
            'artifact_complete':True,'engine':args.engine,'variant':variant,'job_id':job,'source_revision':args.source_revision,
            'projection':'separate','attention_order':args.arm,'runtime':runtime,'checkpoint':checkpoint,'source_records':sources,
            'input_records':input_records[variant],
            'selected_checkpoint_keys':result['selected_checkpoint_keys'],'weights_loaded':363,'model_constructed':False,
            'gold_h0_exact_steps':data['h0_exact'],'attention_calls':data['attention_calls'],'attention_contexts':data['order_contexts'],
            'adapter_preflight':proofs[variant],'historical_split_sorted':{'job_id':m['qkv']['job_id'],'source_revision':m['qkv']['source_revision'],
                'path':str(qkv_root/(args.engine+'-separate')/variant),
                'proof_scope':'this_sorted_replay_vs_history' if args.arm=='sorted' else 'current_sorted_control_vs_history',**proof},
            'current_split_sorted_control':None if args.arm=='sorted' else {'job_id':job,'source_revision':args.source_revision,
                'path':str(Path(m['current_root'])/(args.engine+'-sorted')/variant),'raw1300exact':True},
            'historical_fused_reference':fused_headers[variant],'order_effect_metrics':effects,'metrics':data['metrics'],
            'performance_measurement':False,'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'scope':'SeparateQKV fixed; prefixKV/mask order only. Query layout observed, not changed. Gold externaltrajectory/KV, no samplerfree-run/new6of7gate. NativeH0notseparatelyexported.'}
        folder=args.output/variant;folder.mkdir();publish_case(folder,data['outputs'],preflight[variant],report)
        print(json.dumps({'engine':args.engine,'variant':variant,'arm':args.arm,'status':report['status'],'current_sorted_control_exact':proof['exact']}),flush=True)
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())

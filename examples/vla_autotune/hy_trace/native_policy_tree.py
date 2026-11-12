"""Seven-fixture native-prefix/free-running quality diagnostic, never a default."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def _module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def _sibling(name):return _module(Path(__file__).with_name(name+'.py'),'hy_native_policy_'+name)


policy=_sibling('expert_policy_scope');expert=policy.expert;order=expert.order
ablation=_sibling('proj1_camera_ablation');controls=_sibling('native_controls');common=policy.common
validator=_module(Path(__file__).parents[1]/'validate_hyvla.py','hy_native_policy_validator')
VARIANTS=ablation.VARIANTS
CONDITIONS=('prefix.embeddings','prefix.pad_mask','prefix.block_mask','prefix.modality_mask','prefix.permutation',
    'prefix.sorted_embeddings','prefix.mask_sorted','prefix.cos_sorted','prefix.sin_sorted','prefix.k','prefix.v',
    'prepared.state','prepared.tokens','prepared.token_mask','input.noise','suffix.mask_sorted','suffix.cos','suffix.sin','suffix.state_embedding')


def admit_replay(first,second):
    import torch
    a,b=first['tensors'],second['tensors']
    if (not a or set(a)!=set(b) or first['sampler_metadata']!=second['sampler_metadata']
            or any(not policy._same(v,b[k]) or not torch.isfinite(v).all() for k,v in a.items())):
        raise ValueError('Current request replay raw/metadata mismatch')
    return {'all_current_raw_exact':True,'tensors':len(a),'sampler_metadata_equal':True}


def run_matrix(execute,qualify_control,qualify_candidate):
    matrix={'control':{},'candidate':{}}
    for arm in ('control','candidate'):
        for variant in VARIANTS:
            pair=[execute(arm,variant,index) for index in range(2)]
            replay=admit_replay(*pair)
            qualification=(qualify_control(variant,pair) if arm=='control' else
                qualify_candidate(variant,pair,matrix['control'][variant]['requests'][0]))
            matrix[arm][variant]={'requests':pair,'replay':replay,'qualification':qualification}
        if arm=='control' and (any(row['qualification'].get('historical_actions_exact') is not True for row in matrix[arm].values())
                or sum(row['qualification']['official_oracle']['same_action_gate_passed'] for row in matrix[arm].values())!=6):
            raise ValueError('All historical controls and unchanged6of7 gate must pass before candidate')
    return matrix


def gate_counts(matrix):
    counts={'control':sum(r['qualification']['official_oracle']['same_action_gate_passed'] for r in matrix['control'].values()),
        'candidate':sum(r['qualification']['same_action_gate_passed'] for r in matrix['candidate'].values())}
    if counts['control']!=6:raise ValueError('Reproduced control no longer yields the unchanged historical6of7 gate')
    return counts


def legacy_step_check(variant,tensors,legacy):
    if variant not in ('baseline','state'):
        if legacy is not None:raise ValueError('No historical legacy step trace exists for this fixture')
        return {'historical_step_trace_available':False,'scope':'current_control_new_capture_two_replay_only'}
    if legacy is None:raise ValueError('Required historical baseline/state legacy step trace missing')
    names={'actions',*(f'step{s}.{f}' for s in range(10) for f in ('latent_before','time_embedding','velocity'))}
    names.update(name for name in CONDITIONS if name in legacy)
    for name in names:
        if name not in tensors or name not in legacy or not policy._same(tensors[name],legacy[name]):
            raise ValueError('Current control differs from historical legacy step field: '+name)
    return {'historical_step_trace_available':True,'job_id':'313766','fields_exact':len(names),
        'scope':'observed_native_steps_and_available_conditions_not_forged_other_fixtures'}


def qualify_candidate(candidate,control,arrays):
    a,b=candidate['tensors'],control['tensors']
    names=(*CONDITIONS,*(f'step{s}.time_embedding' for s in range(10)))
    if any(name not in a or name not in b or not policy._same(a[name],b[name]) for name in names):
        raise ValueError('Candidate changed native prefix/prefill or fixed conditions')
    actions=a['actions'][0].numpy();validator.validate_actions(actions)
    result=validator.oracle_comparison(actions,arrays,sampler='euler',num_steps=10)
    result['native_prefix_and_fixed_conditions_exact_control']=True
    return result


def admit_full_support(report,raw,references,*,factory):
    if (report.get('kind')!='hy_live_source_tree_RMS_full_Expert_replay' or report.get('status')!='diagnostic_complete'
            or report.get('job_id')!='316722' or report.get('source_revision')!='d8cdfec96aac64e31454c4c65cac5f05cd931706'
            or report.get('raw_tensors')!=5200 or report.get('golden_norm_data_used') is not False
            or report.get('model_constructed') is not False or report.get('production_norm_replaced') is not False):
        raise ValueError('R32 full Expert support identity/header differs')
    expected=set()
    for case in ('baseline','state'):
        for arm,target in (('control','native'),('tree','old')):
            prefix=arm+'.'+case+'.';values={k.removeprefix(prefix):v for k,v in raw.items() if k.startswith(prefix)}
            expert.baseline_gate(values,references[case][target]);expected.update(prefix+k for k in values)
    if set(raw)!=expected or len(raw)!=5200:raise ValueError('R32 full support raw role set differs')
    return factory()


def publish_completed(path,report,backend_state,requests):
    if backend_state.get('execution')!='diagnostic_eager' or backend_state.get('cleanup',{}).get('passed') is not True:
        raise ValueError('Cannot publish before backend cleanup/close')
    if len(requests)!=28 or any(r['scope'].get('complete') is not True or r['scope'].get('nfe')!=10
            or r['scope'].get('cleanup',{}).get('passed') is not True for r in requests):raise ValueError('Incomplete request capture/cleanup')
    text=json.dumps(report,indent=2,allow_nan=False)+'\n';tmp=path.with_name(path.name+'.tmp');tmp.write_text(text);tmp.replace(path)


def _load_support(folder,task):
    from safetensors.torch import load_file
    if folder!=task/'results/hy-expert-mean-tree-316722' or (folder/'SOURCE_GIT_COMMIT').read_text().strip()!='d8cdfec96aac64e31454c4c65cac5f05cd931706':raise ValueError('R32 support root/source differs')
    r=json.loads((folder/'native/result.json').read_text())
    if set(r.get('source_records',{}))!={'driver','tree','core','native'}:raise ValueError('R32 support source roles differ')
    expert.rms._record_files_unchanged(r['source_records'],task/'snapshots/round032/code')
    refs={}
    for case in ('baseline','state'):
        refs[case]={}
        for engine,target in (('native','native'),('official','old')):
            rec=r['input_records'][case][engine]
            for role,name in (('report','result.json'),('raw','operator_outputs.safetensors')):
                if Path(rec[role]['path']).resolve()!=task/'results/hy-split-order-315024'/(engine+'-original')/case/name:raise ValueError('R32 oracle file binding differs')
            expert.rms._record_files_unchanged(rec,task/'results')
            refs[case][target]=load_file(rec['raw']['path'])
    return r,load_file(str(folder/'native/operator_outputs.safetensors')),refs


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('checkpoint','fixtures','prior','support-root','preflight-report','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-revision',required=True);args=p.parse_args(argv);common.require_step()
    import platform,sys
    import numpy as np
    import torch
    from safetensors.torch import load_file,save_file
    from flash_rt.vla import BackendConfig,load_backend
    from flash_rt.frontends.torch.hyvla_probeflow import HyVLAFlowFrontend
    if len(args.source_revision)!=40 or any(c not in '0123456789abcdef' for c in args.source_revision):raise ValueError('Exact current source revision required')
    validator.require_cuda(torch);torch.set_num_threads(2);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    support=args.support_root.resolve();task=support.parent.parent
    if args.prior.resolve()!=task/'results/hy-proj1-fold-309681/camera_proj1':raise ValueError('Historical seven-fixture prior root differs')
    runtime={'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}
    sr,sa,refs=_load_support(support,task);_,checkpoint=validator.checkpoint_record(args.checkpoint)
    if (runtime!=sr['runtime'] or torch.version.git_version!=sr['torch_git_version']
            or torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction!=sr['bf16_reduced_precision_reduction']
            or ablation._checkpoint_stat(args.checkpoint)!=sr['checkpoint']):raise ValueError('Support/current runtime/checkpoint/flags differ')
    native_source=Path(__file__).resolve().parents[3]/'flash_rt/models/hyvla/pipeline_thor.py'
    for path,record in ((Path(expert.__file__),sr['source_records']['driver']),(Path(expert.mt.__file__),sr['source_records']['tree']),(native_source,sr['source_records']['native'])):
        if path.read_bytes()!=Path(record['path']).read_bytes():raise ValueError('R32 RMS/tree/native implementation source changed')
    proof=json.loads(args.preflight_report.read_text());ablation.validate_precondition(proof,job_id=os.environ['SLURM_JOB_ID'],source_revision=args.source_revision,checkpoint=args.checkpoint,fold_spatial=True)
    data={};priors={};inputs={};legacy={};legacy_reports={}
    for variant in VARIANTS:
        old_folder=args.prior/variant;pr=json.loads((old_folder/'result.json').read_text())
        ablation.classify_exit(json.loads((old_folder/'child_exit.json').read_text())['returncode'],pr)
        with np.load(old_folder/'actions.npz',allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
        fixture=args.fixtures/(variant+'.npz')
        with np.load(fixture,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        admitted=controls.admit_candidate_prior(pr,old,arrays,variant=variant,expected_job='309681',checkpoint=checkpoint,runtime=runtime,fixture=validator.fixture_record(fixture))
        if not np.array_equal(old['oracle_actions20'],arrays['expected_actions20']) or not np.array_equal(old['actions20'],old['actions32'][:,:20]):raise ValueError('Prior public slice/oracle pairing differs')
        data[variant]=arrays;priors[variant]=old
        inputs[variant]={'admission':admitted,'fixture':validator.fixture_record(fixture),'prior_files':{n:expert.rms.layers._file_record(old_folder/n) for n in ('result.json','actions.npz','child_exit.json')}}
        if variant in ('baseline','state'):
            folder=task/'results/hy-folded-state-trace-313766/native'/variant;lr=json.loads((folder/'result.json').read_text())
            expert.rms.layers.admit_baseline_header(lr,job_id='313766',source_revision='2959bd8710038c7dcdebdd17ac645b7feed4333f',checkpoint=checkpoint)
            if lr['runtime']!=runtime or lr['variant']!=variant:raise ValueError('Legacy native step trace runtime/case differs')
            legacy[variant]=load_file(str(folder/'trace.safetensors'));legacy_reports[variant]={'job_id':'313766','files':{n:expert.rms.layers._file_record(folder/n) for n in ('result.json','trace.safetensors')}}
    args.output.mkdir(parents=True,exist_ok=False)
    config=BackendConfig(model='hyvla',checkpoint=str(args.checkpoint),hardware='cuda_sm80',precision='bf16',sampler='euler',num_steps=10,
        options={'use_cuda_graph':False,'numerical_profile':'official_local_fp32'})
    backend_state={};requests=[];sources=None
    context=admit_full_support(sr,sa,refs,factory=lambda:ablation.scoped_backend(load_backend,config,arm='camera_proj1',preflight=proof,state=backend_state,
        synchronize=torch.cuda.synchronize,expected_frontend_type=HyVLAFlowFrontend,fold_spatial=True,execution='diagnostic_eager'))
    del sa,refs
    try:
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),context as backend:
            frontend=backend.frontend;sources=validator.model_sources(backend)
            separate=expert.rms.layers.load_native_owner(native_source,projection='separate',n_heads=frontend.n_heads,n_kv=frontend.n_kv,head_dim=frontend.pipe.head_dim)._block.__func__
            def execute(arm,variant,index):
                arrays=data[variant];folder=args.output/arm/variant/('replay'+str(index));folder.mkdir(parents=True,exist_ok=False)
                tensors={};state={};request_id=arm+'.'+variant+'.r'+str(index)
                obs={'images':{'observation.images.'+k:arrays[k] for k in ('top_head','hand_left','hand_right')},'state':arrays['state'],'prompt':str(arrays['prompt'].item())}
                try:
                    with policy.expert_policy_request(frontend,enabled=arm=='candidate',request_id=request_id,separate_block=separate,tensors=tensors,state=state,
                            expected_nfe=10,synchronize=torch.cuda.synchronize):result=backend.infer(obs,noise=arrays['noise'].copy())
                    validator.validate_actions(result['actions'])
                    if not policy._same(tensors['actions'],torch.from_numpy(result['actions'])[None]):raise ValueError('API action domain differs from raw native graph boundary')
                    sampler=validator.jsonable(result.get('sampler_metadata'));meta=validator.jsonable(result.get('backend_metadata'))
                    validator.validate_sampler_metadata(sampler,sampler='euler',num_steps=10,max_steps=None)
                    if (sampler.get('numerical_profile')!='official_local_fp32' or meta.get('model')!='hyvla' or meta.get('precision')!='bf16'
                            or meta.get('action_domain')!='normalized-padded-model-actions'):raise ValueError('Actual native sampler/backend metadata differs')
                    for s in range(10):
                        expected=tensors[f'step{s+1}.latent_before'] if s<9 else tensors['actions']
                        if not policy._same(controls.reference_euler_step(tensors[f'step{s}.latent_before'],tensors[f'step{s}.velocity']),expected):raise ValueError('Observed native Euler increment chain differs')
                    record={'status':'request_captured_pending_backend_cleanup','request_id':request_id,'scope':state,'sampler_metadata':sampler,'backend_metadata':meta,
                        'euler_chain_characterization_exact':True,'time_scalar_source':'frontend_schedule_reconstruction_not_observed_scalar','path':str(folder.relative_to(args.output))}
                    common._save(folder,tensors,record,atomic_report=True)
                except BaseException as error:
                    if tensors:save_file({k:v.contiguous() for k,v in tensors.items()},str(folder/'unqualified_partial.safetensors'))
                    (folder/'failure.json').write_text(json.dumps({'status':'unqualified_request_failure','error_type':type(error).__name__,'error':str(error)},indent=2)+'\n')
                    raise
                item={**record,'tensors':tensors};requests.append(item);return item
            def qualify_control(variant,pair):
                if not np.array_equal(pair[0]['tensors']['actions'][0].numpy(),priors[variant]['actions32']):raise ValueError('Current control actions differ from historical HWD prior')
                evidence=legacy_step_check(variant,pair[0]['tensors'],legacy.get(variant))
                return {'historical_actions_exact':True,'historical_job_id':'309681','step_evidence':evidence,
                    'official_oracle':validator.oracle_comparison(pair[0]['tensors']['actions'][0].numpy(),data[variant],sampler='euler',num_steps=10)}
            matrix=run_matrix(execute,qualify_control,lambda variant,pair,control:qualify_candidate(pair[0],control,data[variant]))
        summarized={arm:{case:{'replay':row['replay'],'qualification':row['qualification'],'request_paths':[r['path'] for r in row['requests']]}
            for case,row in rows.items()} for arm,rows in matrix.items()}
        counts=gate_counts(matrix)
        report={'kind':'hy_native_policy_expert_phase_source_tree_diagnostic','status':'diagnostic_complete','job_id':os.environ['SLURM_JOB_ID'],'source_revision':args.source_revision,
            'runtime':runtime,'torch_git_version':torch.version.git_version,'checkpoint':checkpoint,'source_support':{'job_id':'316722',
                'source_revision':'d8cdfec96aac64e31454c4c65cac5f05cd931706','root':str(support),
                'files':{n:expert.rms.layers._file_record(support/'native'/n) for n in ('result.json','operator_outputs.safetensors')}},
            'precondition':proof,'inputs':inputs,'legacy_step_traces':legacy_reports,'model_sources':sources,'backend_cleanup':backend_state['cleanup'],
            'matrix':summarized,'original_padded_gate_passes':counts,'requests':28,'actual_nfe':sum(r['sampler_metadata']['nfe'] for r in requests),
            'common_vision_projection':'per_camera_hwd','candidate_expert_changes':['separate_qkv','prefix_original_attention_columns','live_source_old_rms_including_final_head'],
            'historical_policy_result_unchanged':{'job_id':'309681','padded_passes':6,'cases':7},
            'model_constructed':True,'weight_instances':1,'graphs_enabled':False,'performance_measurement':False,'production_default_changed':False,
            'source_records':{n:expert.rms.layers._file_record(f) for n,f in (('driver',__file__),('scope',policy.__file__),('RMS',expert.__file__),('tree',expert.mt.__file__))},
            'scope':'Only existingsevenfixture numericalproxy/nativeKV/free-running latents; no robot/generalization/speed/default claim. Historical6/7FAILED record unchanged.'}
        publish_completed(args.output/'result.json',report,backend_state,requests)
        print(json.dumps({'original_padded_gate_passes':counts,'requests':28,'nfe':report['actual_nfe'],'production_default_changed':False}),flush=True)
    except BaseException as error:
        (args.output/'failure.json').write_text(json.dumps({'status':'unqualified_diagnostic_failure','completed_request_paths':[r['path'] for r in requests],
            'error_type':type(error).__name__,'error':str(error),'backend_cleanup':backend_state.get('cleanup')},indent=2)+'\n')
        raise
    return 0


if __name__=='__main__':raise SystemExit(main())

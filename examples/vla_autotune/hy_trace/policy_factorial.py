"""One bounded E×P interaction matrix; not case-specific policy routing."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def _sibling(name):
    spec=importlib.util.spec_from_file_location('hy_factorial_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


p2=_sibling('proj2_policy');legacy=p2.legacy;ablation=p2.ablation
GROUPS={'base':(False,False),'Eonly':(True,False),'Ponly':(False,True),'both':(True,True)}
KNOWN={'base':('317665','a507813727816c1bf201bbcf759c0df87d57d905','control'),
    'Eonly':('316975','54a661d52a7b3a9d4619eb4f3639ff5ae7f29e91','candidate'),
    'Ponly':('317665','a507813727816c1bf201bbcf759c0df87d57d905','candidate')}


def group_flags(group):
    if group not in GROUPS:raise ValueError('Unknown fixed factorial group')
    return GROUPS[group]


def execute_group_request(backend,arrays,*,group,output,variant,index,separate_template,synchronize):
    E,P=group_flags(group);global_id=group+'.'+variant+'.r'+str(index)
    def context(tensors):
        state={'factorial_group':group,'E':E,'P':P,'global_request_id':global_id}
        return p2.merger_request(backend.frontend.pipe,enabled=P,tensors=tensors,state=state,synchronize=synchronize)
    record=legacy.execute_request(backend,arrays,output=Path(output)/'groups'/group,arm='control' if group=='base' else 'candidate',
        variant=variant,index=index,expert_enabled=E,separate_block=separate_template if E else None,synchronize=synchronize,request_context=context)
    return {**record,'group':group,'global_path':'groups/'+group+'/'+record['path']}


def run_factorial(execute,admit_known,qualify):
    matrix={}
    for group in GROUPS:
        matrix[group]={}
        for case in legacy.VARIANTS:
            pair=[execute(group,case,i) for i in (0,1)];replay=legacy.admit_replay(*pair)
            known=None
            if group!='both':
                known=admit_known(group,case,pair,matrix)
                if known.get('raw_reference_exact') is not True:raise ValueError('Known cell raw/reference admission did not pass')
            qualification=qualify(group,case,pair[0],matrix)
            matrix[group][case]={'requests':pair,'replay':replay,'known_reference':known,'qualification':qualification}
    return matrix


def raw_roles(*,vision=True):
    roles={*legacy.CONDITIONS,'actions',*(f'step{i}.{f}' for i in range(10) for f in ('latent_before','time','time_embedding','velocity'))}
    if vision:roles.update(p2.VISION_FIELDS)
    return roles


def admit_known_cell(group,case,pair,matrix,references):
    import torch
    if group not in KNOWN or case not in legacy.VARIANTS or len(pair)!=2:raise ValueError('Invalid known cell identity/coverage')
    ref=references[group][case];job,source,_=KNOWN[group];old=ref['tensors'];roles=raw_roles(vision=group!='Eonly')
    if (ref.get('job_id')!=job or ref.get('source_revision')!=source or ref.get('variant')!=case or set(old)!=roles):
        raise ValueError('Known cell reference source/case/raw roles differ')
    for record in pair:
        raw=record['tensors']
        if (set(raw)!=raw_roles() or record['sampler_metadata']!=ref['sampler_metadata'] or record['backend_metadata']!=ref['backend_metadata']
                or any(not torch.isfinite(v).all() or not p2.merger._exact(v,old[k]) for k,v in raw.items() if k in roles)):
            raise ValueError('Known cell raw reference values/dtype/metadata differ')
        if group=='Eonly':
            base=matrix['base'][case]['requests'][0]['tensors']
            if any(not p2.merger._exact(raw[k],base[k]) for k in p2.VISION_FIELDS):raise ValueError('Known Eonly new vision fields differ from current base')
    return {'raw_reference_exact':True,'reference_job_id':job,'reference_source_revision':source,'matched_historical_fields':len(old),
        'current_replays_matched':2,'new_vision_fields_exact_current_base':4 if group=='Eonly' else 0,
        'vision_provenance':'new_current_capture_compared_to_current_base_not_R33' if group=='Eonly' else 'actual_R36_reference_capture'}


def admit_known_header(report,group,*,inputs,runtime,checkpoint,torch_git):
    if group not in KNOWN:raise ValueError('Unknown historical reference group')
    job,source,_=KNOWN[group]
    kind='hy_native_policy_expert_phase_source_tree_diagnostic' if group=='Eonly' else 'hy_native_policy_single_PROJ2_diagnostic'
    if (report.get('kind')!=kind or report.get('status')!='diagnostic_complete' or report.get('job_id')!=job or report.get('source_revision')!=source
            or report.get('runtime')!=runtime or report.get('checkpoint')!=checkpoint or report.get('torch_git_version')!=torch_git
            or report.get('inputs')!=inputs or report.get('requests')!=28 or report.get('actual_nfe')!=280 or report.get('weight_instances')!=1
            or report.get('backend_cleanup',{}).get('passed') is not True or report.get('graphs_enabled') is not False
            or report.get('performance_measurement') is not False or report.get('production_default_changed') is not False):
        raise ValueError('Known parent reference identity/input/checkpoint/runtime/cleanup differs')
    matrix=report.get('matrix',{})
    if set(matrix)!={'control','candidate'}:raise ValueError('Known reference parent matrix roles differ')
    for arm in matrix:
        if set(matrix[arm])!=set(legacy.VARIANTS):raise ValueError('Known reference parent case coverage differs')
        for case,row in matrix[arm].items():
            if row.get('request_paths')!=[f'{arm}/{case}/replay{i}' for i in (0,1)]:raise ValueError('Known reference raw paths/case binding differ')


def load_known_pair(root,*,group,case):
    import torch
    from safetensors.torch import load_file
    job,source,arm=KNOWN[group];pair=[];files={}
    for index in (0,1):
        relative=f'{arm}/{case}/replay{index}';folder=Path(root)/relative
        record=json.loads((folder/'result.json').read_text());raw=load_file(str(folder/'trace.safetensors'))
        if (record.get('status')!='request_captured_pending_backend_cleanup' or record.get('request_id')!=f'{arm}.{case}.r{index}'
                or record.get('path')!=relative or set(raw)!=raw_roles(vision=group!='Eonly') or any(not torch.isfinite(v).all() for v in raw.values())
                or record.get('scope',{}).get('complete') is not True or record['scope'].get('cleanup',{}).get('passed') is not True):
            raise ValueError('Known reference sidecar identity/raw/cleanup differs')
        try:validate_schema({**record,'tensors':raw},vision=group!='Eonly')
        except ValueError as error:raise ValueError('Known reference typed raw/sampler admission failed: '+str(error)) from error
        pair.append({**record,'tensors':raw})
        for name in ('result.json','trace.safetensors'):files[relative+'/'+name]=p2.merger.file_record(folder/name)
    legacy.admit_replay(*pair)
    if pair[0]['backend_metadata']!=pair[1]['backend_metadata']:raise ValueError('Known reference backend metadata replay differs')
    return {**pair[0],'job_id':job,'source_revision':source,'variant':case,'reference_files':files}


def _visual_bridge(raw):
    import torch
    pe=raw['prefix.embeddings'];mm=raw['prefix.modality_mask'];perm=raw['prefix.permutation']
    if mm.dtype!=torch.bool or mm.shape!=pe.shape[:2] or int(mm.sum())!=147 or perm.dtype!=torch.long:raise ValueError('Factorial actual prefix modality geometry differs')
    expected=torch.cat((mm[0].nonzero().flatten(),(~mm[0]).nonzero().flatten()))
    pairs=((perm,expected),(raw['prefix.sorted_embeddings'],pe[:,perm]),
        (pe[:,mm[0]].reshape(3,49,2048),raw['vision.merged']),
        (raw['vision.proj2_output'].reshape(3,49,2048),raw['vision.merged']))
    if any(not p2.merger._exact(a,b) for a,b in pairs):raise ValueError('Factorial own visual/sorted prefix bridge differs')


def qualify_cell(group,record,cells,arrays):
    E,P=group_flags(group);state=record['operator_scope'];raw=record['tensors']
    if record['scope'].get('enabled') is not E or state.get('enabled') is not P or state.get('factorial_group')!=group or state.get('E') is not E or state.get('P') is not P:
        raise ValueError('Factorial factor/phase binding differs')
    if 'base' not in cells:raise ValueError('Factorial fixed base conditions missing')
    base=cells['base'];b=base['tensors'];changed={'prefix.embeddings','prefix.sorted_embeddings','prefix.k','prefix.v'}
    strict=tuple(k for k in legacy.CONDITIONS if k not in changed)+('vision.merger_input','vision.proj2_activation','step0.latent_before')
    strict+=tuple(f'step{i}.time_embedding' for i in range(10))
    if any(k not in raw or k not in b or not p2.merger._exact(raw[k],b[k]) for k in strict):raise ValueError('Factorial fixed input/condition invariant differs')
    if any(state[k]!=base['operator_scope'][k] for k in ('merger_input_layout','proj2_activation_layout')):raise ValueError('Factorial input/activation layout invariant differs')
    if record['scope'].get('prefix_segments')!=base['scope'].get('prefix_segments'):raise ValueError('Factorial prefix segment invariant differs')
    _visual_bridge(raw);_visual_bridge(b);mm=raw['prefix.modality_mask'][0]
    if not p2.merger._exact(raw['prefix.embeddings'][:,~mm],b['prefix.embeddings'][:,~mm]):raise ValueError('Factorial non-image prefix invariant differs')
    fixed=None
    if group=='Eonly':fixed=base
    elif group=='both':
        if 'Ponly' not in cells:raise ValueError('Factorial fixed-P cell missing')
        fixed=cells['Ponly']
    if fixed is not None:
        keys=(*legacy.CONDITIONS,*p2.VISION_FIELDS)
        if any(not p2.merger._exact(raw[k],fixed['tensors'][k]) for k in keys):raise ValueError('Factorial Expert leaked into fixed-P Vision/prefix invariant')
    actions=raw['actions'][0].numpy();legacy.validator.validate_actions(actions)
    q=legacy.validator.oracle_comparison(actions,arrays,sampler='euler',num_steps=10)
    q.update(all_factorial_fixed_conditions_exact=True,fixed_P_prefix_invariant_checked=fixed is not None)
    return q


def interactions(cells):
    import torch
    if set(cells)!=set(GROUPS):raise ValueError('Interaction requires all four actual cells')
    for key in ('step0.latent_before','step0.time_embedding','suffix.state_embedding','input.noise'):
        if any(not p2.merger._exact(cells[g][key],cells['base'][key]) for g in GROUPS):raise ValueError('Interaction initial conditions differ')
    arrays={};stats={}
    for name,key,dtype in (('first_velocity','step0.velocity',torch.bfloat16),('actions','actions',torch.float32)):
        if any(cells[g][key].shape!=(1,40,32) or cells[g][key].dtype!=dtype or not torch.isfinite(cells[g][key]).all() for g in GROUPS):raise ValueError('Interaction actual output shape/dtype/finite differs')
        y={g:cells[g][key].detach().cpu().double() for g in GROUPS};value=(y['both']-y['Ponly'])-(y['Eonly']-y['base'])
        arrays[name]=value
        stats[name]={'shape':list(value.shape),'dtype':str(value.dtype),'nonzero_elements':int(torch.count_nonzero(value)),
            'max_abs':float(value.abs().max()),'l2_norm':float(value.norm()),'exactly_zero':bool(torch.count_nonzero(value)==0)}
    return arrays,stats


def validate_schema(record,*,vision=True):
    import torch
    raw=record['tensors'];s=record['scope']
    if set(raw)!=raw_roles(vision=vision) or any(not torch.isfinite(v).all() for v in raw.values()):raise ValueError('Factorial raw role/finite coverage differs')
    shapes={'vision.merger_input':(3,196,1152),'vision.proj2_activation':(3,7,7,2048),'vision.proj2_output':(3,7,7,2048),'vision.merged':(3,49,2048)}
    if vision and any(raw[k].shape!=shape or raw[k].dtype!=torch.bfloat16 for k,shape in shapes.items()):raise ValueError('Factorial actual Vision geometry differs')
    if raw['prefix.embeddings'].ndim!=3:raise ValueError('Factorial prefix geometry differs')
    P=raw['prefix.embeddings'].shape[1]
    specification={'actions':((1,40,32),torch.float32),'input.noise':((1,40,32),torch.float32),
        **{k:((1,P,2048),torch.bfloat16) for k in ('prefix.embeddings','prefix.sorted_embeddings')},
        **{k:((1,P),torch.bool) for k in ('prefix.pad_mask','prefix.block_mask','prefix.modality_mask')},
        'prefix.permutation':((P,),torch.long),'prefix.mask_sorted':((1,1,P,P),torch.bool),
        **{k:((1,1,P,128),torch.float32) for k in ('prefix.cos_sorted','prefix.sin_sorted')},
        'prefix.k':((32,1,P,4,128),torch.float32),'prefix.v':((32,1,P,4,128),torch.bfloat16),
        'prepared.state':((1,32),torch.bfloat16),'prepared.tokens':((1,64),torch.long),'prepared.token_mask':((1,64),torch.bool),
        'suffix.mask_sorted':((1,1,41,P+41),torch.bool),'suffix.cos':((1,1,41,128),torch.float32),'suffix.sin':((1,1,41,128),torch.float32),
        'suffix.state_embedding':((1,1,1024),torch.bfloat16)}
    for i in range(10):
        specification.update({f'step{i}.latent_before':((1,40,32),torch.float32),f'step{i}.time':((1,),torch.float32),
            f'step{i}.time_embedding':((1,1024),torch.bfloat16),f'step{i}.velocity':((1,40,32),torch.bfloat16)})
    if any(raw[k].shape!=shape or raw[k].dtype!=dtype for k,(shape,dtype) in specification.items()):raise ValueError('Factorial raw dtype/geometry differs')
    sampler=record['sampler_metadata'];meta=record['backend_metadata']
    legacy.validator.validate_sampler_metadata(sampler,sampler='euler',num_steps=10,max_steps=None)
    if (sampler.get('numerical_profile')!='official_local_fp32' or meta.get('model')!='hyvla' or meta.get('precision')!='bf16'
            or meta.get('action_domain')!='normalized-padded-model-actions'):raise ValueError('Factorial actual sampler/backend metadata differs')
    if s.get('generation')!=1 or s.get('prefix_rows')!=P or s.get('n_vis')!=147 or set(s.get('prefix_segments',{}))!={'idx_ranges','full_ranges'}:
        raise ValueError('Factorial actual prefix geometry/generation/segment evidence differs')


def validate_capture(record,group,case,index):
    validate_schema(record)
    E,projection=group_flags(group);raw=record['tensors'];s=record['scope'];op=record['operator_scope']
    local='control' if group=='base' else 'candidate';identity=f'{local}.{case}.r{index}';path=f'{local}/{case}/replay{index}'
    if (case not in legacy.VARIANTS or type(index) is not int or index not in (0,1) or record.get('request_id')!=identity
            or s.get('request_id')!=identity or record.get('path')!=path or record.get('group')!=group
            or record.get('global_path')!=f'groups/{group}/'+path):raise ValueError('Factorial local/global request identity/path differs')
    if (s.get('enabled') is not E or s.get('nfe')!=10 or s.get('prefill_calls')!=1 or s.get('complete') is not True
            or s.get('cleanup',{}).get('passed') is not True or len(s.get('steps',[]))!=10
            or op.get('enabled') is not projection or op.get('E') is not E or op.get('P') is not projection or op.get('factorial_group')!=group
            or op.get('global_request_id')!=f'{group}.{case}.r{index}' or op.get('complete') is not True or op.get('cleanup',{}).get('passed') is not True
            or op.get('merger_calls')!=1 or op.get('proj2_calls')!=1 or len(op.get('proj2_views',[]))!=(3 if projection else 1)
            or len(op.get('proj1_views',[]))!=3 or any(v.get('shares_input_storage') is not True for v in op['proj2_views'])):
        raise ValueError('Factorial actual phase/identity/cleanup coverage differs')
    for step in s['steps']:
        norm=step['norm']
        if (step['block_calls']!=32 or step['attention_calls']!=32 or step['prefix_readonly'] is not True
                or norm['calls']!=129 or norm['original_norm_calls']!=(0 if E else 129) or norm['use_tree'] is not E
                or norm['complete'] is not True or norm['restored'] is not True or norm['by_role']!={'torch.bfloat16:1024':65,'torch.float32:128':64}):
            raise ValueError('Factorial actual Expert call/role counters differ')
        if E and (step.get('order',{}).get('attention_calls')!=32 or step['order'].get('cleanup',{}).get('passed') is not True):raise ValueError('Factorial Expert order proof incomplete')
        if not E and 'order' in step:raise ValueError('Factorial unexpectedly changed attention order')
    _visual_bridge(raw)


def publish_completed(path,report,backend_state,matrix,interaction_arrays):
    import torch
    from safetensors.torch import save_file
    if backend_state.get('execution')!='diagnostic_eager' or backend_state.get('cleanup',{}).get('passed') is not True:raise ValueError('Factorial backend close/cleanup incomplete')
    if set(matrix)!=set(GROUPS) or any(set(matrix[g])!=set(legacy.VARIANTS) for g in GROUPS):raise ValueError('Factorial four-cell/seven-case coverage incomplete')
    rows=[r for cases in matrix.values() for row in cases.values() for r in row['requests']]
    if len(rows)!=56 or any(len(row['requests'])!=2 for cases in matrix.values() for row in cases.values()):raise ValueError('Factorial requires exactly56 requests and complete replay coverage')
    identities=set();count=0
    for group in GROUPS:
        for case,row in matrix[group].items():
            if group!='both' and row.get('known_reference',{}).get('raw_reference_exact') is not True:raise ValueError('Factorial known reference barrier proof incomplete')
            for i,r in enumerate(row['requests']):
                validate_capture(r,group,case,i);ident=r['operator_scope']['global_request_id']
                if ident in identities:raise ValueError('Factorial global request identity duplicated')
                identities.add(ident);count+=len(r['tensors'])
    expected={case+'.'+key for case in legacy.VARIANTS for key in ('first_velocity','actions')}
    if (count!=3584 or set(interaction_arrays)!=expected or any(v.dtype!=torch.float64 or v.shape!=(1,40,32) or not torch.isfinite(v).all() for v in interaction_arrays.values())):
        raise ValueError('Factorial full raw/14interaction-array coverage differs')
    save_file({k:v.contiguous() for k,v in interaction_arrays.items()},str(path.parent/'interaction_outputs.safetensors'))
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def load_known_references(task,*,inputs,runtime,checkpoint,torch_git,reduction_flag):
    task=Path(task);references={};provenance={}
    for group in KNOWN:
        job,source,_=KNOWN[group];folder=task/'results'/('hy-native-policy-tree-'+job if group=='Eonly' else 'hy-proj2-policy-'+job)
        if (folder/'SOURCE_GIT_COMMIT').read_text().strip()!=source:raise ValueError('Known reference frozen source marker differs')
        report=json.loads((folder/'native/result.json').read_text())
        admit_known_header(report,group,inputs=inputs,runtime=runtime,checkpoint=checkpoint,torch_git=torch_git)
        round_name='round033' if group=='Eonly' else 'round036'
        mapping=({'driver':legacy.__file__,'scope':legacy.policy.__file__,'RMS':legacy.expert.__file__,'tree':legacy.expert.mt.__file__}
            if group=='Eonly' else p2._source_records())
        if set(report.get('source_records',{}))!=set(mapping):raise ValueError('Known reference source roles differ')
        for role,record in report['source_records'].items():
            current=Path(mapping[role] if group=='Eonly' else mapping[role]['path'])
            relative=current.resolve().relative_to(Path(__file__).resolve().parents[3])
            frozen=task/'snapshots'/round_name/'code'/relative
            if Path(record['path']).resolve()!=frozen or p2.merger.file_record(frozen)!=record:raise ValueError('Known reference source actual path/stat differs')
            # R33's driver was deliberately extracted in R36; its fixed old
            # source is retained, while the scope/RMS/tree math must be unchanged.
            if not (group=='Eonly' and role=='driver') and current.read_bytes()!=frozen.read_bytes():raise ValueError('Known reference computation source changed')
        if group!='Eonly' and report.get('bf16_reduced_precision_reduction')!=reduction_flag:raise ValueError('Known reference reduction flags differ')
        references[group]={};files={}
        for case in legacy.VARIANTS:
            reference=load_known_pair(folder/'native',group=group,case=case)
            files[case]=reference['reference_files'];reference['parent_qualification']=report['matrix'][KNOWN[group][2]][case]['qualification']
            references[group][case]=reference
        provenance[group]={'job_id':job,'source_revision':source,'parent':p2.merger.file_record(folder/'native/result.json'),
            'source_records':report['source_records'],'requests':files}
    return references,provenance


def run(args,runtime,torch_git,reduction_flag):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from flash_rt.vla import BackendConfig,load_backend
    from flash_rt.frontends.torch.hyvla_probeflow import HyVLAFlowFrontend
    task,history,refs,support=p2.load_references(args.reference_root,runtime=runtime,torch_git=torch_git,reduction_flag=reduction_flag)
    checkpoint=support['input_manifest']['checkpoint'];checkpoint_dir=Path(checkpoint['path']).parent
    root=task/'results'/('hy-policy-factorial-'+os.environ['SLURM_JOB_ID'])
    pre_path=args.precondition_report.resolve()
    if pre_path!=root/'precondition/result.json' or args.output.resolve()!=root/'native':raise ValueError('R37 current precondition/output/job paths differ')
    pre=json.loads(pre_path.read_text());pre_raw=load_file(str(pre_path.with_name('operator_outputs.safetensors')))
    if pre.get('source_support')!=support or pre.get('source_records')!=p2._source_records() or pre.get('bf16_reduced_precision_reduction')!=reduction_flag:
        raise ValueError('R37 fresh complete precondition source/support/flags changed')
    p2.admit_precondition(pre,pre_raw,references=refs,job_id=os.environ['SLURM_JOB_ID'],source_revision=args.source_revision,
        checkpoint=checkpoint,runtime=runtime,torch_git=torch_git,factory=lambda:None)
    # Admit the complete R32 source-and-array support before any E intervention
    # or model construction, not merely the prior summary of exact endpoints.
    sr,sa,expert_refs=legacy._load_support(task/'results/hy-expert-mean-tree-316722',task)
    if (sr['runtime']!=runtime or sr['torch_git_version']!=torch_git or sr['checkpoint']!=checkpoint
            or sr['bf16_reduced_precision_reduction']!=reduction_flag):raise ValueError('R37 Expert support runtime/checkpoint/flags differ')
    native_source=Path(__file__).resolve().parents[3]/'flash_rt/models/hyvla/pipeline_thor.py'
    for path,record in ((Path(legacy.expert.__file__),sr['source_records']['driver']),
            (Path(legacy.expert.mt.__file__),sr['source_records']['tree']),(native_source,sr['source_records']['native'])):
        if path.read_bytes()!=Path(record['path']).read_bytes():raise ValueError('R37 Expert RMS/tree/native implementation changed')
    legacy.admit_full_support(sr,sa,expert_refs,factory=lambda:None)
    expert_support={'job_id':'316722','source_revision':sr['source_revision'],'source_records':sr['source_records'],
        'files':{n:p2.merger.file_record(task/'results/hy-expert-mean-tree-316722/native'/n) for n in ('result.json','operator_outputs.safetensors')}}
    del sa,expert_refs,pre_raw,refs,history
    _,directory=legacy.validator.checkpoint_record(checkpoint_dir)
    data,priors,inputs,old,old_records=legacy.load_policy_inputs(args.fixtures,args.prior,task=task,checkpoint=directory,weight_stat=checkpoint,runtime=runtime)
    known,known_records=load_known_references(task,inputs=inputs,runtime=runtime,checkpoint=directory,torch_git=torch_git,reduction_flag=reduction_flag)
    config=BackendConfig(model='hyvla',checkpoint=str(checkpoint_dir),hardware='cuda_sm80',precision='bf16',sampler='euler',num_steps=10,
        options={'use_cuda_graph':False,'numerical_profile':'official_local_fp32'})
    backend_state={};completed=[]
    context=ablation.scoped_backend(load_backend,config,arm='camera_proj1',preflight=pre['proj1_child'],state=backend_state,
        synchronize=torch.cuda.synchronize,expected_frontend_type=HyVLAFlowFrontend,fold_spatial=True,execution='diagnostic_eager')
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),context as backend:
            frontend=backend.frontend;sources=legacy.validator.model_sources(backend)
            separate=legacy.expert.rms.layers.load_native_owner(native_source,projection='separate',n_heads=frontend.n_heads,n_kv=frontend.n_kv,
                head_dim=frontend.pipe.head_dim)._block.__func__
            def execute(group,case,index):
                r=execute_group_request(backend,data[case],group=group,output=args.output,variant=case,index=index,separate_template=separate,synchronize=torch.cuda.synchronize)
                validate_capture(r,group,case,index);completed.append(r['global_path']);return r
            def admit(group,case,pair,matrix):
                proof=admit_known_cell(group,case,pair,matrix,known)
                if group=='base':
                    if not np.array_equal(pair[0]['tensors']['actions'][0].numpy(),priors[case]['actions32']):raise ValueError('Factorial base historical309681 actions differ')
                    proof['historical_actions_exact']=True;proof['step_evidence']=legacy.legacy_step_check(case,pair[0]['tensors'],old.get(case))
                return proof
            def qualify(group,case,r,matrix):
                cells={g:rows[case]['requests'][0] for g,rows in matrix.items() if case in rows};cells[group]=r
                q=qualify_cell(group,r,cells,data[case])
                if group in KNOWN:
                    old_q=known[group][case]['parent_qualification']
                    if group=='base':old_q=old_q['official_oracle']
                    for key in ('gate','same_action_gate_passed','public20_gate_passed'):
                        if q[key]!=old_q[key]:raise ValueError('Known raw cell original gate disposition differs')
                    for domain in ('actions32','actions20'):
                        if q[domain]['outside_tolerance']!=old_q[domain]['outside_tolerance']:raise ValueError('Known raw cell original elementwise failures differ')
                return q
            matrix=run_factorial(execute,admit,qualify)
        # CPU-only arithmetic on captured arrays after all GPU ownership closes.
        interaction_arrays={};interaction_stats={}
        for case in legacy.VARIANTS:
            values,stats=interactions({g:matrix[g][case]['requests'][0]['tensors'] for g in GROUPS})
            interaction_arrays.update({case+'.'+k:v for k,v in values.items()});interaction_stats[case]=stats
        counts={g:sum(row['qualification']['same_action_gate_passed'] for row in rows.values()) for g,rows in matrix.items()}
        summary={g:{case:{k:row[k] for k in ('replay','known_reference','qualification')} | {'request_paths':[r['global_path'] for r in row['requests']]}
            for case,row in rows.items()} for g,rows in matrix.items()}
        report={'kind':'hy_native_policy_Expert_PROJ2_factorial_diagnostic','status':'diagnostic_complete','job_id':os.environ['SLURM_JOB_ID'],
            'source_revision':args.source_revision,'runtime':runtime,'torch_git_version':torch_git,'checkpoint':directory,'bf16_reduced_precision_reduction':reduction_flag,
            'source_records':{'factorial':p2.merger.file_record(__file__),'components':p2._source_records()},'source_support':support,
            'Expert_support':expert_support,'known_references':known_records,'inputs':inputs,'legacy_step_traces':old_records,'model_sources':sources,
            'complete_precondition':{'report':pre,'files':{n:p2.merger.file_record(pre_path.parent/n) for n in ('result.json','operator_outputs.safetensors')}},
            'matrix':summary,'factor_mapping':{g:{'E':e,'P':p} for g,(e,p) in GROUPS.items()},'original_padded_gate_passes':counts,
            'requests':56,'actual_nfe':sum(r['sampler_metadata']['nfe'] for rows in matrix.values() for row in rows.values() for r in row['requests']),
            'raw_tensors':3584,'interaction_tensors':14,'interaction_formula':'(both-Ponly)-(Eonly-base)','interaction_stats':interaction_stats,
            'backend_cleanup':backend_state['cleanup'],'model_constructed':True,'weight_instances':1,'graphs_enabled':False,
            'performance_measurement':False,'production_default_changed':False,'known_cells_all_admitted_before_both':True,
            'next_decision':'existing_seven_fixture_proxy_only_no_default_promotion' if counts['both']==7 else 'stop_combination_search_next_frozen_upstream_ViT',
            'scope':'Post-selectedmechanisticinteraction onexisting7cases,NOTindependentgeneralization/robot/speed/defaultqualification. Ehas3changesnotseparatelyidentified; upstreamvisiongapremains.'}
        publish_completed(args.output/'result.json',report,backend_state,matrix,interaction_arrays)
        print(json.dumps({'original_padded_gate_passes':counts,'requests':56,'actual_nfe':report['actual_nfe'],'next_decision':report['next_decision']}),flush=True)
    except BaseException as error:
        (args.output/'failure.json').write_text(json.dumps({'status':'unqualified_factorial_diagnostic_failure','completed_request_paths':completed,
            'error_type':type(error).__name__,'error':str(error),'backend_cleanup':backend_state.get('cleanup')},indent=2)+'\n')
        raise
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('reference-root','precondition-report','fixtures','prior','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-revision',required=True);args=p.parse_args(argv);legacy.common.require_step()
    import platform,sys
    import numpy as np
    import torch
    if len(args.source_revision)!=40 or any(c not in '0123456789abcdef' for c in args.source_revision):raise ValueError('Exact R37 source commit required')
    legacy.validator.require_cuda(torch);torch.set_num_threads(2);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    runtime={'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}
    return run(args,runtime,torch.version.git_version,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)


if __name__=='__main__':raise SystemExit(main())

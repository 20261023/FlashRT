"""Recorded Expert layer replay using native bodies, never a full model or policy."""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import MethodType,SimpleNamespace


def _sibling(name):
    spec=importlib.util.spec_from_file_location('hy_layers_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _file_record(path):
    path=Path(path).resolve();stat=path.stat()
    return {'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def load_native_owner(path,*,n_heads=16,n_kv=4,head_dim=128):
    import torch
    tree=ast.parse(Path(path).read_text())
    cls=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HyVLAThorBF16Pipeline']
    rotate=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_rot_half']
    names=('_rms_norm','_attn','_block')
    methods=[n for c in cls for n in c.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(cls)!=1 or len(rotate)!=1 or len(methods)!=3 or any(n.decorator_list for n in (*rotate,*methods)):
        raise ValueError('Native layer source must contain four undecorated selected definitions')
    if any(type(n) is not int or n<1 for n in (n_heads,n_kv,head_dim)) or n_heads%n_kv or head_dim%2:
        raise ValueError('Invalid native head geometry')
    scope={'torch':torch,'F':torch.nn.functional}
    exec(compile(ast.Module(body=copy.deepcopy([*rotate,*methods]),type_ignores=[]),str(path),'exec'),scope)
    owner=SimpleNamespace(numerical_profile='official_local_fp32',n_heads=n_heads,n_kv=n_kv,head_dim=head_dim,
        q_dim=n_heads*head_dim,kv_dim=n_kv*head_dim,rms_eps=1e-5,
        _fp8=False,_fp4=False,_ffn_mega=False,_fused_attn=False)
    for name in names:setattr(owner,name,MethodType(scope[name],owner))
    return owner


def layer_keys(index):
    e=f'model.dual_tower.expert.model.layers.{index}'
    v=f'model.dual_tower.vlm.model.language_model.model.layers.{index}.self_attn'
    return {**{name:e+f'.self_attn.{name}_proj_v.weight' for name in ('q','k','v','o')},
        **{name:e+f'.mlp_v.{field}_proj.weight' for name,field in (('g','gate'),('u','up'),('d','down'))},
        'ln':e+'.input_layernorm_v.weight','post':e+'.post_attention_layernorm_v.weight',
        'qn':v+'.query_layernorm.weight','kn':v+'.key_layernorm.weight'}


def fuse_layer_weights(raw):
    import torch
    if set(raw)!=set(layer_keys(0)) or any(value.dtype!=torch.bfloat16 for value in raw.values()):
        raise ValueError('Native layer weights must be the admitted eleven BF16 tensors')
    return (torch.cat((raw['q'],raw['k'],raw['v']),0),raw['o'],torch.cat((raw['g'],raw['u']),0),
            raw['d'],raw['ln'],raw['post']),(raw['qn'],raw['kn'])


def replay_block(owner,hs,weights,qk,k_template,v_template,mask,cos,sin,*,prefix_rows):
    import torch
    if (owner.numerical_profile!='official_local_fp32' or any(getattr(owner,n) is not False for n in ('_fp8','_fp4','_ffn_mega','_fused_attn'))
            or hs.ndim!=3 or hs.shape[0]!=1 or hs.dtype!=torch.bfloat16
            or type(prefix_rows) is not int or prefix_rows<1):
        raise ValueError('Native block replay requires admitted unquantized local-FP32 suffix geometry')
    rows=hs.shape[1];shape=(1,owner.n_heads,prefix_rows+rows,owner.head_dim)
    if (k_template.shape!=shape or v_template.shape!=shape or k_template.dtype!=torch.float32 or v_template.dtype!=torch.bfloat16
            or mask.shape!=(1,1,rows,prefix_rows+rows) or mask.dtype!=torch.bool
            or cos.shape!=(1,1,rows,owner.head_dim) or sin.shape!=cos.shape or cos.dtype!=torch.float32 or sin.dtype!=torch.float32):
        raise ValueError('Native replay cache/mask/RoPE shape or dtype differs')
    readonly=(hs,mask,cos,sin)
    if any(not torch.isfinite(t).all() for t in (*readonly,k_template,v_template)):
        raise ValueError('Native replay input must be finite')
    before=[t.clone() for t in readonly]
    kbuf,vbuf=torch.zeros_like(k_template),torch.zeros_like(v_template)
    kbuf[:,:,:prefix_rows].copy_(k_template[:,:,:prefix_rows]);vbuf[:,:,:prefix_rows].copy_(v_template[:,:,:prefix_rows])
    kp,vp=kbuf[:,:,:prefix_rows].clone(),vbuf[:,:,:prefix_rows].clone()
    original=owner._rms_norm;calls=[];residual=[]
    def observe(value,weight):
        index=len(calls);calls.append(index)
        if index>=4:raise ValueError('Unexpected extra native block norm call')
        if index==3:
            if value.shape!=hs.shape or value.dtype!=torch.bfloat16:raise ValueError('Native residual norm input differs')
            residual.append(value.detach().clone())
        return original(value,weight)
    try:
        owner._rms_norm=observe
        result=owner._block(hs,rows,None,weights,qk,mask,cos,sin,kbuf,vbuf,prefix_rows)
    finally:
        owner._rms_norm=original
    if len(calls)!=4 or len(residual)!=1:raise ValueError('Native block norm count differs')
    if any(not torch.equal(t,b) for t,b in zip(readonly,before)):raise ValueError('Native block mutated a read-only input')
    if not torch.equal(kp,kbuf[:,:,:prefix_rows]) or not torch.equal(vp,vbuf[:,:,:prefix_rows]):
        raise ValueError('Native block mutated frozen prefix KV')
    if result.shape!=hs.shape or result.dtype!=torch.bfloat16 or not torch.isfinite(result).all():
        raise ValueError('Native block output shape/dtype/finite contract differs')
    return residual[0],result


def layer_inputs(gold,propagated):
    return {'teacher':gold.clone(),'propagated':propagated.clone()}


def admit_h0(rebuilt,gold,expected_state):
    import torch
    state,suffix=rebuilt['state_embedding'],rebuilt['suffix']
    if (state.dtype!=expected_state.dtype or not torch.equal(state,expected_state)
            or suffix.dtype!=gold.dtype or suffix.shape!=(1,*gold.shape) or not torch.equal(suffix[0],gold)):
        raise ValueError('Rebuilt native state/suffix H0 differs from the recorded Gold boundary')


def final_velocity(owner,hs,norm,weight,bias,*,action_rows=40):
    import torch.nn.functional as F
    return F.linear(owner._rms_norm(hs,norm)[:,-action_rows:],weight,bias)


def baseline_gate(current,historical,*,engine):
    import torch
    if engine not in ('official','native'):raise ValueError('Unknown native-body runtime label')
    count=sum(a.dtype==b.dtype==torch.bfloat16 and a.shape==b.shape and torch.isfinite(a).all().item()
              and torch.equal(a,b) for a,b in zip(current,historical))
    exact=len(current)==len(historical) and count==len(current) and len(current)>0
    return {'required':engine=='native','passed':bool(exact or engine!='native'),
            'compared_exact':bool(exact),'exact_steps':int(count),'steps':len(current),
            'scope':'Only native2.11 must reproduce historical313766; old2.7 measures same-body runtime effect'}


def admit_layer_capture(report,*,checkpoint,layer_source):
    capture=report.get('expert_layer_capture',{});chain=capture.get('chain',{})
    if (report.get('checkpoint')!=checkpoint or capture.get('complete') is not True
            or capture.get('num_layers')!=32 or capture.get('num_steps')!=10 or capture.get('steps')!=10
            or capture.get('events')!=970 or capture.get('rows')!=41 or capture.get('hidden')!=1024
            or capture.get('module')!='expert.model' or chain.get('exact') is not True
            or chain.get('layer_boundaries')!=320 or chain.get('first_suffix_checked') is not True
            or capture.get('source')!=_file_record(layer_source)):
        raise ValueError('Expert layer capture/checkpoint/chain/source admission failed')


def admit_baseline_header(report,*,job_id,source_revision,checkpoint):
    extra=report.get('experimental_projection',{});proof=extra.get('precondition',{})
    checks=report.get('frozen_control_evidence',{})
    if (report.get('engine')!='native' or report.get('job_id')!=job_id or report.get('artifact_complete') is not True
            or report.get('execution_mode')!='diagnostic_eager' or report.get('diagnostic_cleanup',{}).get('passed') is not True
            or report.get('config',{}).get('use_cuda_graph') is not False
            or report.get('config',{}).get('numerical_profile')!='official_local_fp32'
            or extra.get('fold_spatial') is not True or extra.get('effective_projection')!='per_camera_hwd'
            or extra.get('source_revision')!=source_revision or proof.get('source_revision')!=source_revision
            or proof.get('job_id')!=job_id or proof.get('checkpoint')!=checkpoint or proof.get('operator_exact') is not True
            or report.get('historical_official_reference',{}).get('job_id')!='308690'
            or report.get('prior_graph_output',{}).get('exact') is not True or report.get('suffix_order_control') is not None
            or checks.get('native_replay_exact') is not True or checks.get('prefix_preserved') is not True):
        raise ValueError('Historical native baseline source/order/checkpoint admission failed')


def publish(folder,tensors,report):
    from safetensors.torch import save_file
    save_file({name:v.detach().cpu().contiguous().clone() for name,v in tensors.items()},str(folder/'operator_outputs.safetensors'))
    temporary=folder/'result.json.tmp';temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(folder/'result.json')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--engine',choices=('official','native'),required=True)
    for name in ('reference','checkpoint','baseline','official-attention-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--baseline-job',required=True)
    parser.add_argument('--baseline-source-revision',default='2959bd8710038c7dcdebdd17ac645b7feed4333f')
    parser.add_argument('--source-revision',required=True)
    args=parser.parse_args(argv)
    common=_sibling('diagnose');common.require_step()
    import platform
    import torch
    from safetensors.torch import load_file
    from safetensors import safe_open
    if tuple(torch.cuda.get_device_capability())!=(8,0) or 'A800' not in torch.cuda.get_device_name():
        raise RuntimeError('Expert layer replay requires the recorded A800 target')
    if not torch.__version__.startswith('2.7.1' if args.engine=='official' else '2.11.0'):
        raise RuntimeError('Use the recorded engine-specific Torch runtime')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    runtime={'python':platform.python_version(),'executable':__import__('sys').executable,
             'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':__import__('numpy').__version__}
    checkpoint=_sibling('proj1_camera_ablation')._checkpoint_stat(args.checkpoint)
    frozen=_sibling('frozen_eager_attention');capture=_sibling('expert_layers_capture')
    suffix=_sibling('frozen_suffix');order=_sibling('suffix_key_order')
    cases={};layer_source=args.official_attention_source.resolve().parent/'hunyuan_vl_mot/modeling_hunyuan_vl_mot.py'
    for variant in ('baseline','state'):
        report=json.loads((args.reference/variant/'result.json').read_text())
        frozen.admit_capture_report(report,job_id=os.environ['SLURM_JOB_ID'],source_revision=args.source_revision,
            variant=variant,attention_source=args.official_attention_source)
        admit_layer_capture(report,checkpoint=checkpoint,layer_source=layer_source)
        prior=json.loads((args.baseline/variant/'result.json').read_text())
        admit_baseline_header(prior,job_id=args.baseline_job,source_revision=args.baseline_source_revision,checkpoint=checkpoint)
        if args.engine=='native' and prior['runtime']!=runtime:raise ValueError('Native runtime differs from historical baseline')
        gold=load_file(str(args.reference/variant/'trace.safetensors'))
        old_trace=load_file(str(args.baseline/variant/'trace.safetensors'))
        old_frozen=load_file(str(args.baseline/variant/'frozen_velocities.safetensors'))
        capture.verify_boundary_chain(gold,first_suffix=gold['suffix.embeddings'])
        for name in ('prepared.images','prepared.state','prepared.tokens','prepared.token_mask','prefix.pad_mask','prefix.modality_mask','suffix.state_embedding'):
            if gold[name].dtype!=old_trace[name].dtype or not torch.equal(gold[name],old_trace[name]):
                raise ValueError('Recorded layer inputs differ from historical native prepared inputs')
        mm=gold['prefix.modality_mask'][0]
        if mm.dtype!=torch.bool or mm.numel()!=240:raise ValueError('Unexpected recorded prefix modality geometry')
        perm=torch.cat((mm.nonzero().flatten(),(~mm).nonzero().flatten())).numpy()
        order.admit_original_mask(old_trace['suffix.mask_sorted'],gold['suffix.attention_mask'],perm)
        cases[variant]={'gold':gold,'trace':old_trace,'frozen':old_frozen,'perm':perm,
                        'outputs':{},'metrics':{},'propagated':{},'h0_exact':0}
    native_source=Path(__file__).resolve().parents[3]/'flash_rt/models/hyvla/pipeline_thor.py'
    owner=load_native_owner(native_source)
    suffix_keys={name:'model.'+field for name,field in (
        ('_state_w','state_proj.weight'),('_state_b','state_proj.bias'),('_ain_w','action_in_proj.weight'),('_ain_b','action_in_proj.bias'),
        ('_atmlp_in_w','action_time_mlp_in.weight'),('_atmlp_in_b','action_time_mlp_in.bias'),
        ('_atmlp_out_w','action_time_mlp_out.weight'),('_atmlp_out_b','action_time_mlp_out.bias'))}
    head_keys=('model.dual_tower.expert.model.norm.weight','model.action_out_proj.weight','model.action_out_proj.bias')
    args.output.mkdir(parents=True,exist_ok=False)
    selected=[]
    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),safe_open(checkpoint['path'],framework='pt',device='cpu') as saved:
        def load(key):
            value=saved.get_tensor(key)
            if value.dtype!=torch.bfloat16 or not torch.isfinite(value).all():raise ValueError('Selected checkpoint tensor is not finite BF16')
            selected.append(key);return value.to('cuda')
        sw=SimpleNamespace(**{name:load(key) for name,key in suffix_keys.items()})
        for case in cases.values():
            gold,trace,prior=case['gold'],case['trace'],case['frozen']
            for step in range(10):
                latent=prior[f'frozen_inputs.step{step}.latent'];embedding=prior[f'frozen_inputs.step{step}.time_embedding']
                if latent.dtype!=gold[f'step{step}.latent_before'].dtype or not torch.equal(latent,gold[f'step{step}.latent_before']):
                    raise ValueError('Frozen official latent differs from fresh recorded trajectory')
                if embedding.dtype!=trace[f'step{step}.time_embedding'].dtype or not torch.equal(embedding,trace[f'step{step}.time_embedding']):
                    raise ValueError('Frozen time embedding differs from admitted native schedule')
                rebuilt=suffix.build_suffix(trace['prepared.state'].to('cuda'),latent.to('cuda'),embedding.to('cuda'),sw)
                h0=gold[f'expert.step{step}.layer0.input'].to('cuda')
                admit_h0(rebuilt,h0,trace['suffix.state_embedding'].to('cuda'))
                case['propagated'][step]=rebuilt['suffix'].clone();case['h0_exact']+=1
            case['mask']=trace['suffix.mask_sorted'].to('cuda');case['cos']=trace['suffix.cos'].to('cuda');case['sin']=trace['suffix.sin'].to('cuda')
        del sw
        for layer in range(32):
            raw={name:load(key) for name,key in layer_keys(layer).items()}
            weights,qk=fuse_layer_weights(raw)
            shapes=((3072,1024),(1024,2048),(4096,1024),(1024,2048),(1024,),(1024,))
            if tuple(tuple(v.shape) for v in weights)!=shapes or any(q.shape!=(128,) for q in qk):
                raise ValueError('Native Expert selected layer geometry differs from recorded contract')
            for case in cases.values():
                gold=case['gold'];prefix=240
                cache=[]
                for field,dtype in (('k',torch.float32),('v',torch.bfloat16)):
                    source=gold['prefix.'+field][layer:layer+1]
                    expanded=common.expand_official_kv(common._array(source),case['perm'],16,41)[0]
                    cache.append(torch.from_numpy(expanded).to(device='cuda',dtype=dtype))
                for step in range(10):
                    base=f'expert.step{step}.layer{layer}.'
                    target_r=gold[base+'residual'][None]
                    target_h=gold[f'expert.step{step}.layer{layer+1}.input'][None] if layer<31 else gold[f'expert.step{step}.final_input']
                    inputs=layer_inputs(gold[base+'input'][None].to('cuda'),case['propagated'][step])
                    for mode,hidden in inputs.items():
                        r,h=replay_block(owner,hidden,weights,qk,*cache,case['mask'],case['cos'],case['sin'],prefix_rows=prefix)
                        if mode=='propagated':case['propagated'][step]=h
                        for field,value,target in (('residual',r,target_r),('output',h,target_h)):
                            key=f'{mode}.step{step}.layer{layer}.{field}'
                            cpu=value.detach().cpu().contiguous().clone();case['outputs'][key]=cpu
                            metric=common.difference(common._array(cpu),common._array(target));metric['unequal_elements']=int((cpu!=target).sum())
                            case['metrics'][key]=metric
            del raw,weights,qk
        head=[load(key) for key in head_keys]
        all_passed=True
        for variant,case in cases.items():
            current,historical=[],[]
            for step in range(10):
                velocity=final_velocity(owner,case['propagated'][step],*head)
                cpu=velocity.detach().cpu().contiguous().clone()
                old=case['frozen'][f'official_kv_sorted.step{step}.velocity']
                current.append(cpu);historical.append(old);case['outputs'][f'propagated.step{step}.velocity']=cpu
                case['metrics'][f'propagated.step{step}.velocity_vs_historical']=common.difference(common._array(cpu),common._array(old))
                gold_head=final_velocity(owner,case['gold'][f'expert.step{step}.final_input'].to('cuda'),*head)
                gold_cpu=gold_head.detach().cpu().contiguous().clone();case['outputs'][f'gold_head.step{step}.velocity']=gold_cpu
                case['metrics'][f'gold_head.step{step}.velocity_vs_recorded']=common.difference(common._array(gold_cpu),common._array(case['gold'][f'step{step}.velocity']))
            gate=baseline_gate(current,historical,engine=args.engine);all_passed=all_passed and gate['passed']
            folder=args.output/variant;folder.mkdir()
            result={'kind':'recorded_Expert_layer_local_and_propagated_replay_not_policy_gate',
                'status':'diagnostic_complete' if gate['passed'] else 'historical_native_baseline_reproduction_failed',
                'engine':args.engine,'variant':variant,'job_id':os.environ['SLURM_JOB_ID'],'source_revision':args.source_revision,
                'runtime':runtime,'checkpoint':checkpoint,'native_method_source':_file_record(native_source),'prototype_source':_file_record(__file__),
                'selected_checkpoint_keys':selected,'weights_loaded':len(selected),'model_constructed':False,'performance_measurement':False,
                'gold_h0_exact_steps':case['h0_exact'],'historical_native_baseline':{'job_id':args.baseline_job,
                    'source_revision':args.baseline_source_revision,'path':str((args.baseline/variant).resolve()),**gate},
                'metrics':case['metrics'],'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                'scope':'Gold prefixKV/sortedorder/officialexternaltrajectory. Teacher Gold H_i vs layer-propagated native H; no sampler free-run or6of7gate. Errors not additive causal shares.'}
            publish(folder,case['outputs'],result)
            print(json.dumps({'engine':args.engine,'variant':variant,'status':result['status'],'baseline':gate,'h0_exact':case['h0_exact']}),flush=True)
    return 0 if all_passed else 1


if __name__=='__main__':
    raise SystemExit(main())

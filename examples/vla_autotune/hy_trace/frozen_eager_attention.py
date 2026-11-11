"""Frozen recorded eager-attention layout/order controls; no policy or weights."""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace


def load_method(path, class_name, method_name):
    import torch
    tree = ast.parse(Path(path).read_text())
    classes = [node for node in tree.body if isinstance(node,ast.ClassDef) and node.name == class_name]
    methods = [node for cls in classes for node in cls.body if isinstance(node,ast.FunctionDef) and node.name == method_name]
    if len(classes) != 1 or len(methods) != 1 or methods[0].decorator_list:
        raise ValueError('Expected one undecorated revision-bound attention method')
    scope = {'torch':torch,'nn':torch.nn,'F':torch.nn.functional}
    exec(compile(ast.Module(body=[copy.deepcopy(methods[0])],type_ignores=[]),str(path),'exec'),scope)
    return scope[method_name]


def restore_layout(values, info, *, device):
    import torch
    shape, strides, offset = info.get('shape'), info.get('live_stride'), info.get('storage_offset')
    if (shape != list(values.shape) or info.get('dtype') != str(values.dtype)
            or not isinstance(strides,list) or len(strides) != values.ndim
            or any(type(x) is not int or x < 0 for x in strides)
            or type(offset) is not int or offset < 0 or any(size < 1 for size in values.shape)):
        raise ValueError('Recorded attention layout shape/dtype/stride/offset is invalid')
    span = 1
    for stride, size in sorted((stride,size) for stride,size in zip(strides,shape) if size > 1):
        if stride < span:
            raise ValueError('Overlapping recorded attention stride layout is unsupported')
        span += stride*(size-1)
    storage = torch.empty(offset+span,dtype=values.dtype,device=device)
    view = storage.as_strided(shape,strides,offset)
    view.copy_(values.to(device))
    return view


def _layout(value):
    return {'shape':list(value.shape),'live_stride':list(value.stride()),
            'storage_offset':value.storage_offset(),'dtype':str(value.dtype)}


def attention_controls(q,k,v,mask,*,prefix_rows,prefix_perm,official_fn,native_fn):
    import torch
    if (q.ndim != 4 or k.ndim != 4 or k.shape != v.shape or q.dtype != torch.float32
            or k.dtype != torch.float32 or v.dtype != torch.bfloat16 or mask.dtype != torch.bool
            or min(q.shape) < 1 or min(k.shape) < 1 or q.shape[0] != k.shape[0]
            or q.shape[-1] != k.shape[-1] or q.shape[2] % k.shape[2]
            or tuple(mask.shape) != (q.shape[0],q.shape[1],k.shape[1])
            or any(not torch.isfinite(value).all() for value in (q,k,v))):
        raise ValueError('Frozen attention input shape/dtype/finite contract failed')
    if (type(prefix_rows) is not int or prefix_rows < 0 or k.shape[1] != prefix_rows+q.shape[1]
            or not isinstance(prefix_perm,torch.Tensor) or prefix_perm.dtype != torch.long
            or prefix_perm.ndim != 1 or prefix_perm.numel() != prefix_rows
            or not torch.equal(prefix_perm.sort().values,torch.arange(prefix_rows,device=prefix_perm.device))):
        raise ValueError('Invalid complete prefix permutation')
    outputs, layouts = {}, {}
    def call(fn,*args):
        tensors = tuple(value for value in args if isinstance(value,torch.Tensor))
        before = tuple(value.clone() for value in tensors)
        result = fn(*args)
        if any(not torch.equal(value,old) for value,old in zip(tensors,before)):
            raise ValueError('Frozen attention operator mutated a read-only input')
        return result
    batch,length,heads,width = q.shape
    output,probabilities = call(official_fn,mask,batch,width,q,k,v)
    if (tuple(output.shape) != (batch,length,heads*width) or output.dtype != torch.bfloat16
            or tuple(probabilities.shape) != (batch,heads,length,k.shape[1]) or probabilities.dtype != torch.bfloat16):
        raise ValueError('Original attention method returned unexpected shape/dtype')
    outputs['official_api.output'],outputs['official_api.probabilities'] = output,probabilities
    layouts['official_api'] = {name:_layout(value) for name,value in (('q',q),('k',k),('v',v),('mask',mask))}
    groups = heads//k.shape[2]
    expanded = []
    for value in (k,v):
        # The original method expands BN(KV)D before transposing; keep that view.
        item = value[:,:,:,None,:].expand(batch,value.shape[1],value.shape[2],groups,width)
        expanded.append(item.reshape(batch,value.shape[1],heads,width).transpose(1,2))
    query,mask4 = q.transpose(1,2),mask[:,None]
    matching_k,matching_v = expanded
    contiguous_k,contiguous_v = matching_k.contiguous(),matching_v.contiguous()
    columns = torch.cat((prefix_perm.to(q.device),torch.arange(prefix_rows,k.shape[1],device=q.device)))
    arms = {
        'native_matching_official_layout':(matching_k,matching_v,mask4),
        'native_contiguous_original_order':(contiguous_k,contiguous_v,mask4),
        'native_contiguous_sorted_order':(contiguous_k.index_select(2,columns),contiguous_v.index_select(2,columns),mask4.index_select(-1,columns)),
    }
    for name,(key,value,att_mask) in arms.items():
        result = call(native_fn,query,key,value,att_mask)
        if tuple(result.shape) != (batch,heads,length,width) or result.dtype != torch.bfloat16 or not torch.isfinite(result).all():
            raise ValueError('Native attention returned unexpected shape/dtype/finite output')
        outputs[name+'.output'] = result.transpose(1,2).reshape(batch,length,heads*width)
        layouts[name] = {label:_layout(tensor) for label,tensor in (('q',query),('k',key),('v',value),('mask',att_mask))}
    return outputs,{'layouts':layouts,'readonly_inputs_preserved':True,
        'interpretation':'Observed official Q held fixed; layout and prefix-order interventions on recorded official K/V, not native whole-policy replay'}


def _file_record(path):
    path = Path(path).resolve();stat = path.stat()
    return {'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def admit_capture_report(report,*,job_id,source_revision,variant,attention_source):
    capture = report.get('eager_attention_capture',{})
    phases = capture.get('phases',[])
    if (report.get('engine') != 'official' or report.get('job_id') != job_id
            or report.get('source_revision') != source_revision or report.get('variant') != variant
            or report.get('artifact_complete') is not True or report.get('status') != 'recorded_oracle_reproduced'
            or report.get('oracle',{}).get('exact') is not True or not report.get('prefix_invariance')
            or not all(report['prefix_invariance'].values()) or capture.get('complete') is not True
            or capture.get('num_layers') != 32 or capture.get('num_steps') != 10 or len(phases) != 11
            or any(phase.get('attention_calls') != 32 for phase in phases)
            or [phase.get('phase') for phase in phases] != ['prefix',*(f'step{i}' for i in range(10))]
            or report.get('historical_attention_admission',{}).get('exact') is not True
            or report.get('official_attention_source') != _file_record(attention_source)):
        raise ValueError('Fresh official attention capture job/source/oracle admission failed')


def load_admitted_operators(report,*,job_id,source_revision,variant,attention_source,native_source):
    admit_capture_report(report,job_id=job_id,source_revision=source_revision,variant=variant,attention_source=attention_source)
    original=load_method(attention_source,'HyDualTower','eager_attention_forward')
    native=load_method(native_source,'HyVLAThorBF16Pipeline','_attn')
    return original,native


def publish(folder,tensors,report):
    from safetensors.torch import save_file
    save_file({name:value.detach().cpu().contiguous().clone() for name,value in tensors.items()},str(folder/'operator_outputs.safetensors'))
    temporary = folder/'result.json.tmp'
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(folder/'result.json')


def _sibling(name):
    spec=importlib.util.spec_from_file_location('hy_frozen_eager_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--engine',choices=('official','native'),required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--official-attention-source',type=Path,required=True)
    parser.add_argument('--source-revision',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    common=_sibling('diagnose');common.require_step()
    import torch
    from safetensors import safe_open
    if tuple(torch.cuda.get_device_capability())!=(8,0) or 'A800' not in torch.cuda.get_device_name():
        raise RuntimeError('Frozen eager attention controls require the recorded A800 target')
    if not torch.__version__.startswith('2.7.1' if args.engine=='official' else '2.11.0'):
        raise RuntimeError('Use the recorded engine-specific Torch runtime')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    native_source=Path(__file__).resolve().parents[3]/'flash_rt/models/hyvla/pipeline_thor.py'
    args.output.mkdir(parents=True,exist_ok=False)
    reproduced=True
    for variant in ('baseline','state'):
        report=json.loads((args.reference/variant/'result.json').read_text())
        original,native=load_admitted_operators(report,job_id=os.environ['SLURM_JOB_ID'],source_revision=args.source_revision,
            variant=variant,attention_source=args.official_attention_source,native_source=native_source)
        folder=args.output/variant;folder.mkdir()
        tensors,metrics,metadata={},{},{}
        def compare(name,actual,expected):
            metrics[name]=common.difference(common._array(actual),common._array(expected))
            metrics[name]['unequal_elements']=int((actual!=expected).sum().item())
        with torch.inference_mode(),safe_open(str(args.reference/variant/'trace.safetensors'),framework='pt',device='cpu') as saved:
            modality=saved.get_tensor('prefix.modality_mask')[0].to('cuda')
            if modality.dtype!=torch.bool:raise ValueError('Recorded prefix modality must remain bool')
            prefix=modality.numel()
            perm=torch.cat((modality.nonzero().flatten(),(~modality).nonzero().flatten()))
            tensors['prefix_permutation']=perm
            for step in range(10):
                base=f'attention.step{step}.layer0.'
                values={name:restore_layout(saved.get_tensor(base+name),report['tensor_metadata'][base+name],device='cuda') for name in ('q','k','v','mask')}
                q,k,v,mask=(values[name] for name in ('q','k','v','mask'))
                if q.shape!=(1,41,16,128) or k.shape!=(1,prefix+41,4,128):
                    raise ValueError('Captured suffix attention differs from the recorded checkpoint geometry')
                owner=SimpleNamespace(config=SimpleNamespace(vlm_config=SimpleNamespace(num_attention_heads=16,num_key_value_heads=4)))
                native_owner=SimpleNamespace(numerical_profile='official_local_fp32',head_dim=128)
                outputs,detail=attention_controls(q,k,v,mask,prefix_rows=prefix,prefix_perm=perm,
                    official_fn=lambda *a:original(owner,*a),native_fn=lambda *a:native(native_owner,*a))
                expected=saved.get_tensor(base+'output').to('cuda')
                expected_probs=saved.get_tensor(base+'probabilities').to('cuda')
                compare(f'step{step}.official_probabilities_vs_recorded',outputs['official_api.probabilities'],expected_probs)
                for name,value in outputs.items():
                    tensors[f'step{step}.{name}']=value
                    if name.endswith('.output'):compare(f'step{step}.{name}_vs_recorded',value,expected)
                compare(f'step{step}.native_matching_vs_official_api',outputs['native_matching_official_layout.output'],outputs['official_api.output'])
                compare(f'step{step}.contiguous_vs_matching_layout',outputs['native_contiguous_original_order.output'],outputs['native_matching_official_layout.output'])
                compare(f'step{step}.sorted_vs_original_order',outputs['native_contiguous_sorted_order.output'],outputs['native_contiguous_original_order.output'])
                metadata[f'step{step}']=detail
                same=metrics[f'step{step}.official_api.output_vs_recorded']['exact'] and metrics[f'step{step}.official_probabilities_vs_recorded']['exact']
                reproduced=reproduced and (same or args.engine!='official')
        result={'kind':'recorded_eager_attention_layout_and_order_controls_not_policy_gate',
            'status':'diagnostic_complete' if reproduced else 'recorded_operator_reproduction_failed',
            'engine':args.engine,'variant':variant,'job_id':os.environ['SLURM_JOB_ID'],'source_revision':args.source_revision,
            'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),
            'gpu':torch.cuda.get_device_name(),'model_constructed':False,'weights_loaded':0,'performance_measurement':False,
            'reference':str((args.reference/variant).resolve()),'official_attention_source':_file_record(args.official_attention_source),
            'native_attention_source':_file_record(native_source),'metrics':metrics,'control_metadata':metadata,
            'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'caveat':'Frozen observed official Q/K/V; excludes native projection and Qnorm differences. KV contiguous controls, not a claim about actual native Q layout or full-policy accuracy.'}
        publish(folder,tensors,result)
        print(json.dumps({'engine':args.engine,'variant':variant,'status':result['status'],
            'old_runtime_recorded_reproduction_required':args.engine=='official'}),flush=True)
    return 0 if reproduced else 1


if __name__=='__main__':
    raise SystemExit(main())

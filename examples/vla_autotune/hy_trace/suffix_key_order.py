"""Experimental eager-only suffix attention order intervention, not a policy option."""
from __future__ import annotations

from contextlib import contextmanager
import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace


def _file_record(path):
    path=Path(path).resolve();stat=path.stat()
    return {'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def inverse_columns(prefix_perm,*,suffix_rows,device):
    import torch
    perm=torch.as_tensor(prefix_perm)
    if (perm.dtype!=torch.long or perm.ndim!=1 or type(suffix_rows) is not int or suffix_rows<1
            or not torch.equal(perm.sort().values,torch.arange(perm.numel(),device=perm.device))):
        raise ValueError('Invalid complete integer prefix permutation')
    perm=perm.to(device)
    return torch.cat((perm.argsort(),torch.arange(perm.numel(),perm.numel()+suffix_rows,device=device)))


def attention_in_original_order(original,q,k,v,mask,*,columns):
    import torch
    if (q.ndim!=4 or k.ndim!=4 or k.shape!=v.shape or q.shape[:2]!=k.shape[:2]
            or q.shape[-1]!=k.shape[-1] or q.dtype!=torch.float32 or k.dtype!=torch.float32
            or v.dtype!=torch.bfloat16 or mask.dtype!=torch.bool
            or tuple(mask.shape)!=(q.shape[0],1,q.shape[2],k.shape[2])):
        raise ValueError('Original-order suffix attention requires admitted expanded FP32/BF16 geometry')
    prefix=k.shape[2]-q.shape[2]
    if (prefix<1 or columns.dtype!=torch.long or columns.device!=k.device or columns.ndim!=1
            or columns.numel()!=k.shape[2]
            or not torch.equal(columns[:prefix].sort().values,torch.arange(prefix,device=k.device))
            or not torch.equal(columns[prefix:],torch.arange(prefix,k.shape[2],device=k.device))):
        raise ValueError('Original-order suffix columns must invert only the prefix permutation')
    return original(q,k.index_select(2,columns),v.index_select(2,columns),mask.index_select(-1,columns))


@contextmanager
def suffix_original_order(frontend,prefix_perm,*,state,synchronize=None):
    import torch
    pipe=frontend.pipe
    if (frontend.use_cuda_graph is not False or frontend._graph_cache or frontend._vit_graph_cache
            or pipe.numerical_profile!='official_local_fp32' or getattr(pipe,'_fused_attn',False)):
        raise ValueError('Suffix order intervention is explicit eager-only with empty graph caches and unfused official-local math')
    suffix=1+frontend.chunk
    columns=inverse_columns(prefix_perm,suffix_rows=suffix,device=frontend.device)
    prefix=columns.numel()-suffix
    if prefix<1:raise ValueError('Suffix order intervention requires a nonempty prefix')
    synchronize=synchronize or (lambda:torch.cuda.synchronize(frontend.device))
    missing=object();own=pipe.__dict__.get('_attn',missing);original=pipe._attn
    state.update(attention_calls=0,prefix_rows=prefix,suffix_rows=suffix,
        cleanup={'passed':False,'status':'active'},counter_semantics='Eager Python attention calls only, not NFE or GPU kernel counts')
    def call(q,k,v,mask):
        if q.shape[-2]!=suffix or k.shape[-2]!=prefix+suffix:
            raise ValueError('Suffix order intervention geometry changed or a prefix call reached the scope')
        state['attention_calls']+=1
        return attention_in_original_order(original,q,k,v,mask,columns=columns)
    primary=None
    try:
        pipe._attn=call
        yield state
    except BaseException as error:
        primary=error
        raise
    finally:
        errors=[]
        try:synchronize()
        except BaseException as error:errors.append(f'sync failed: {type(error).__name__}: {error}')
        try:
            if own is missing:pipe.__dict__.pop('_attn',None)
            else:pipe._attn=own
        except BaseException as error:errors.append(f'binding restore failed: {type(error).__name__}: {error}')
        state['cleanup']={'passed':not errors,'status':'passed' if not errors else 'failed','errors':errors}
        if errors:
            state['_failed_owners']=(columns,pipe,frontend,original)
            if primary is not None:primary.add_note('; '.join(errors))
            else:raise RuntimeError('; '.join(errors))


def validate_order_proof(proof,*,job_id,source_revision,native_source):
    if (proof.get('kind')!='hy_suffix_original_order_exact_precondition' or proof.get('operator_exact') is not True
            or proof.get('job_id')!=job_id or proof.get('source_revision')!=source_revision
            or proof.get('native_method_source')!=_file_record(native_source)
            or set(proof.get('cases',{}))!={'baseline','state'}
            or any(row.get('steps')!=10 or row.get('exact') is not True for row in proof['cases'].values())):
        raise ValueError('Suffix key-order proof job/source/exact admission failed')


def admit_sorted_baseline(current,historical):
    import torch
    if set(current)!=set(historical) or any(current[name].dtype!=historical[name].dtype
            or not torch.equal(current[name].cpu(),historical[name].cpu()) for name in current):
        raise ValueError('Sorted frozen baseline tensor set/values differ from the admitted previous diagnostic')
    return {'exact':True,'tensors':len(current)}


def admit_sorted_baseline_report(report,*,job_id,reference_job,checkpoint,runtime):
    extra=report.get('experimental_projection',{})
    if (report.get('engine')!='native' or report.get('job_id')!=job_id or report.get('artifact_complete') is not True
            or report.get('execution_mode')!='diagnostic_eager' or report.get('diagnostic_cleanup',{}).get('passed') is not True
            or report.get('config',{}).get('use_cuda_graph') is not False or extra.get('fold_spatial') is not True
            or extra.get('effective_projection')!='per_camera_hwd' or report.get('checkpoint')!=checkpoint
            or report.get('runtime')!=runtime or report.get('historical_official_reference',{}).get('job_id')!=reference_job
            or report.get('suffix_order_control') is not None):
        raise ValueError('Historical sorted baseline identity/fold/order admission failed')
    return {'admitted':True,'job_id':job_id,'source_revision':extra.get('source_revision')}


def admit_original_mask(smask,expected,prefix_perm):
    import torch
    if smask.ndim!=4 or smask.shape[1]!=1 or smask.dtype!=torch.bool or expected.dtype!=torch.bool:
        raise ValueError('Original-order mask eligibility requires the recorded bool geometry')
    columns=inverse_columns(prefix_perm,suffix_rows=smask.shape[-2],device=smask.device)
    if smask.shape[-1]!=columns.numel() or not torch.equal(smask.index_select(-1,columns)[:,0].cpu(),expected.cpu()):
        raise ValueError('Original-order mask eligibility differs from recorded official mask')
    return {'exact':True,'shape':list(expected.shape)}


def publish(folder,tensors,report):
    from safetensors.torch import save_file
    save_file({name:value.detach().cpu().contiguous().clone() for name,value in tensors.items()},str(folder/'operator_outputs.safetensors'))
    temporary=folder/'result.json.tmp'
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(folder/'result.json')


def _sibling(name):
    spec=importlib.util.spec_from_file_location('hy_order_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('reference','reference-ops','official-attention-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference-job',required=True)
    parser.add_argument('--reference-source-revision',required=True)
    parser.add_argument('--source-revision',required=True)
    args=parser.parse_args(argv)
    common=_sibling('diagnose');common.require_step()
    import torch
    from safetensors import safe_open
    if tuple(torch.cuda.get_device_capability())!=(8,0) or 'A800' not in torch.cuda.get_device_name() or not torch.__version__.startswith('2.11.0'):
        raise RuntimeError('Original-order adapter preflight requires the recorded native2.11 A800 runtime')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    frozen=_sibling('frozen_eager_attention')
    native_source=Path(__file__).resolve().parents[3]/'flash_rt/models/hyvla/pipeline_thor.py'
    function=frozen.load_method(native_source,'HyVLAThorBF16Pipeline','_attn')
    owner=SimpleNamespace(numerical_profile='official_local_fp32',head_dim=128)
    original=lambda *values:function(owner,*values)
    args.output.mkdir(parents=True,exist_ok=False)
    tensors,metrics,cases={},{},{}
    all_exact=True
    for variant in ('baseline','state'):
        report=json.loads((args.reference/variant/'result.json').read_text())
        frozen.admit_capture_report(report,job_id=args.reference_job,source_revision=args.reference_source_revision,
            variant=variant,attention_source=args.official_attention_source)
        old_ops=json.loads((args.reference_ops/variant/'result.json').read_text())
        if (old_ops.get('job_id')!=args.reference_job or old_ops.get('source_revision')!=args.reference_source_revision
                or old_ops.get('engine')!='native' or old_ops.get('variant')!=variant
                or old_ops.get('status')!='diagnostic_complete'):
            raise ValueError('Historical sorted attention operator identity differs')
        exact=True
        with torch.inference_mode(),safe_open(str(args.reference/variant/'trace.safetensors'),framework='pt',device='cpu') as saved,safe_open(str(args.reference_ops/variant/'operator_outputs.safetensors'),framework='pt',device='cpu') as historical:
            modality=saved.get_tensor('prefix.modality_mask')[0].to('cuda')
            if modality.dtype!=torch.bool:raise ValueError('Original-order precondition requires observed bool modality')
            perm=torch.cat((modality.nonzero().flatten(),(~modality).nonzero().flatten()))
            columns=inverse_columns(perm,suffix_rows=41,device='cuda')
            sorted_columns=torch.cat((perm,torch.arange(perm.numel(),perm.numel()+41,device='cuda')))
            for step in range(10):
                base=f'attention.step{step}.layer0.'
                values={name:frozen.restore_layout(saved.get_tensor(base+name),report['tensor_metadata'][base+name],device='cuda') for name in ('q','k','v','mask')}
                q=values['q'].transpose(1,2)
                k=values['k'].transpose(1,2).repeat_interleave(4,dim=1).contiguous().index_select(2,sorted_columns)
                v=values['v'].transpose(1,2).repeat_interleave(4,dim=1).contiguous().index_select(2,sorted_columns)
                mask=values['mask'][:,None].index_select(-1,sorted_columns)
                restored=attention_in_original_order(original,q,k,v,mask,columns=columns).transpose(1,2).reshape(1,41,2048)
                sorted_control=original(q,k,v,mask).transpose(1,2).reshape(1,41,2048)
                expected=saved.get_tensor(base+'output').to('cuda')
                expected_sorted=historical.get_tensor(f'step{step}.native_contiguous_sorted_order.output').to('cuda')
                same=torch.equal(restored,expected) and torch.equal(sorted_control,expected_sorted)
                exact=exact and same
                for name,actual,target in (('original_order_vs_recorded',restored,expected),('sorted_vs_historical',sorted_control,expected_sorted),('sorted_vs_recorded',sorted_control,expected)):
                    key=f'{variant}.step{step}.{name}'
                    metrics[key]=common.difference(common._array(actual),common._array(target))
                    metrics[key]['unequal_elements']=int((actual!=target).sum().item())
                tensors[f'{variant}.step{step}.original_order']=restored
                tensors[f'{variant}.step{step}.sorted_control']=sorted_control
        cases[variant]={'steps':10,'exact':bool(exact),'prefix_rows':int(perm.numel()),'suffix_rows':41}
        all_exact=all_exact and exact
    record={'kind':'hy_suffix_original_order_exact_precondition','operator_exact':bool(all_exact),
        'job_id':os.environ['SLURM_JOB_ID'],'source_revision':args.source_revision,'torch':torch.__version__,
        'native_method_source':_file_record(native_source),'prototype_source':_file_record(__file__),
        'historical_reference':{'job_id':args.reference_job,'source_revision':args.reference_source_revision,
            'path':str(args.reference.resolve()),'operator_path':str(args.reference_ops.resolve())},
        'cases':cases,'metrics':metrics,'model_constructed':False,'weights_loaded':0,'performance_measurement':False}
    publish(args.output,tensors,record)
    print(json.dumps({'kind':record['kind'],'operator_exact':record['operator_exact'],'cases':cases}),flush=True)
    return 0 if all_exact else 1


if __name__=='__main__':
    raise SystemExit(main())

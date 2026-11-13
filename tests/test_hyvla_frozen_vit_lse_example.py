import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'examples/vla_autotune/hy_trace/frozen_vit_lse.py'


@pytest.fixture
def lse():
    if not SCRIPT.exists():pytest.fail('Weightless LSE implementation not created yet')
    spec=importlib.util.spec_from_file_location('hy_vit_lse_test',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.mark.parametrize('engine',['official','native'])
def test_actual_QKV_axis_mapping_roundtrip_does_not_clone_input_storage(lse,engine):
    x=torch.zeros(6,196,3456,dtype=torch.bfloat16)
    # Literal camera0/frame4/token7/head3/channel11, distinct Q/K/V values.
    x[4,7,227]=1.;x[4,7,1379]=2.;x[4,7,2531]=3.
    before=x.clone();q,k,v=lse.split_qkv(x,engine=engine)
    assert q.shape==((6,196,16,72) if engine=='official' else (6,16,196,72))
    for t,want in ((q,1.),(k,2.),(v,3.)):
        assert t[4,7,3,11] == want if engine=='official' else t[4,3,7,11] == want
        assert t.untyped_storage().data_ptr()==x.untyped_storage().data_ptr()
    assert torch.equal(lse.reassemble_qkv((q,k,v),engine=engine),before) and torch.equal(x,before)


@pytest.mark.parametrize('bad',['dtype','width','overlap','finite'])
def test_invalid_or_overlapping_QKV_is_rejected_before_API(lse,bad):
    x=torch.zeros(6,196,3456,dtype=torch.bfloat16)
    if bad=='dtype':x=x.float()
    if bad=='width':x=x[:,:,:3455]
    if bad=='overlap':x=x[:1].expand(6,-1,-1)
    if bad=='finite':x[0,0,0]=float('nan')
    with pytest.raises(ValueError,match='QKV|BF16|layout|finite|overlap'):lse.split_qkv(x,engine='native')


def api_result(engine,B=6):
    out=torch.zeros((B,196,16,72) if engine=='official' else (B,16,196,72),dtype=torch.bfloat16)
    normalizer=torch.full((B,16,196),math.log(196),dtype=torch.float32)
    if engine=='official':return (out,normalizer,torch.empty(0,dtype=torch.bfloat16))
    return (out,normalizer,None,None,196,196,torch.zeros(2,dtype=torch.uint64),torch.empty((),dtype=torch.uint64),torch.empty(0,dtype=torch.bfloat16))


@pytest.mark.parametrize('engine',['official','native'])
@pytest.mark.parametrize('bad',['none','arity','context_dtype','padded_lse','lse_dtype','lse_stride','nan','debug'])
def test_return_parser_requires_actual_unpadded_FP32_LSE_and_empty_debug(lse,engine,bad):
    result=list(api_result(engine))
    if bad=='arity':result.pop()
    if bad=='context_dtype':result[0]=result[0].float()
    if bad=='padded_lse':result[1]=torch.zeros(6,16,256)
    if bad=='lse_dtype':result[1]=result[1].to(torch.bfloat16)
    if bad=='lse_stride':result[1]=torch.zeros(6,196,16).transpose(1,2)
    if bad=='nan':result[1][0,0,0]=float('nan')
    if bad=='debug':result[-1]=torch.ones(1,dtype=torch.bfloat16)
    call=lambda:lse.parse_capture_result(tuple(result),engine=engine,batch=6)
    if bad=='none':
        context,value,meta=call()
        assert context.shape==(6,196,1152) and value.dtype==torch.float32 and value.stride()==(3136,196,1)
        assert meta['return_count']==(3 if engine=='official' else 9) and meta['debug_numel']==0
    else:
        with pytest.raises(ValueError,match='return|context|LSE|lse|debug|shape|stride|FP32'):call()


def cpu_API_pair(engine,*,fault=None):
    # Actual CPU attention math below the injected GPU boundary. This does NOT
    # claim to emulate FA2/CUDA rounding or supply the final capture anchors.
    def calculate(q,k,v):
        a,b,c=(t.permute(0,2,1,3) for t in (q,k,v)) if engine=='official' else (q,k,v)
        out=torch.nn.functional.scaled_dot_product_attention(a,b,c,dropout_p=0.,is_causal=False,scale=72**-.5)
        scores=(a.float()@b.float().transpose(-1,-2))*(72**-.5)
        normalizer=torch.logsumexp(scores,-1).contiguous()
        return out.permute(0,2,1,3) if engine=='official' else out,normalizer
    def plain(q,k,v,**kw):
        assert kw==({'dropout_p':0.,'softmax_scale':72**-.5,'causal':False,'window_size':(-1,-1),'softcap':0.,
            'alibi_slopes':None,'deterministic':False,'return_attn_probs':False} if engine=='official' else
            {'dropout_p':0.,'is_causal':False,'scale':72**-.5})
        if fault=='body':raise RuntimeError('actual plain body failed')
        return calculate(q,k,v)[0]
    def capture(q,k,v,**kw):
        assert kw==({'dropout_p':0.,'softmax_scale':72**-.5,'causal':False,'window_size':(-1,-1),'softcap':0.,
            'alibi_slopes':None,'deterministic':False,'return_attn_probs':True} if engine=='official' else
            {'dropout_p':0.,'is_causal':False,'return_debug_mask':False,'scale':72**-.5})
        out,normalizer=calculate(q,k,v)
        if fault=='capture_context':out=out+1
        if fault=='readonly':q.add_(1)
        if engine=='official':return out,normalizer,torch.empty(0,dtype=torch.bfloat16)
        return out,normalizer,None,None,196,196,torch.zeros(2,dtype=torch.uint64),torch.empty((),dtype=torch.uint64),torch.empty(0,dtype=torch.bfloat16)
    return plain,capture


@pytest.mark.parametrize('engine',['official','native'])
def test_two_fixed_groupings_keep_actual_input_views_and_require_R39_context(lse,engine):
    x=torch.zeros(18,196,3456,dtype=torch.bfloat16);before=x.clone()
    values=tuple(t.clone() for t in x.split(6)) if engine=='official' else x
    target=torch.zeros(18,196,1152,dtype=torch.bfloat16);plain,capture=cpu_API_pair(engine)
    raw,state=lse.capture_lse(engine,values,plain=plain,capture=capture,targets=target,synchronize=lambda:None)
    assert len(raw)==8 and state['complete'] and state['cleanup']['passed']
    assert state['plain_calls']==state['capture_calls']==4 and state['learned_weight_read_events']==0
    assert torch.equal(raw['input.qkv'],before) and torch.equal(raw['after.qkv'],before)
    for group in ('g6','g18'):
        assert torch.equal(raw[group+'.context_control'],target) and torch.equal(raw[group+'.context_lse_capture'],target)
        assert raw[group+'.lse'].shape==(18,16,196) and raw[group+'.lse'].dtype==torch.float32
    if engine=='native':
        assert [r['qkv_layout']['storage_offset'] for r in state['groups']['g6']]==[0,4064256,8128512]
        assert all(r['shares_original_QKV_storage'] for r in state['groups']['g6'])


@pytest.mark.parametrize('failure',['plain_context','capture_context','body','sync','readonly'])
def test_context_or_execution_failure_cannot_qualify_LSE_and_preserves_partial(lse,failure):
    x=torch.zeros(18,196,3456,dtype=torch.bfloat16);target=torch.zeros(18,196,1152,dtype=torch.bfloat16)
    plain,capture=cpu_API_pair('native',fault=failure)
    if failure=='plain_context':target+=1
    def sync():
        if failure=='sync':raise RuntimeError('LSE sync failed')
    with pytest.raises((ValueError,RuntimeError),match='context|body|sync|readonly|control') as caught:
        lse.capture_lse('native',x,plain=plain,capture=capture,targets=target,synchronize=sync)
    raw,state=caught.value.lse_partial
    assert raw and not state['complete']
    if failure=='plain_context':assert state['capture_calls']==0


def test_capture_mutating_new_old_group18_buffer_is_not_hidden_by_original_camera_copies(lse):
    values=tuple(torch.zeros(6,196,3456,dtype=torch.bfloat16) for _ in range(3));target=torch.zeros(18,196,1152,dtype=torch.bfloat16)
    plain,capture=cpu_API_pair('official')
    def mutate(q,k,v,**kw):
        result=capture(q,k,v,**kw)
        if q.shape[0]==18:q.add_(1)
        return result
    with pytest.raises(ValueError,match='readonly|operand') as caught:
        lse.capture_lse('official',values,plain=plain,capture=mutate,targets=target,synchronize=lambda:None)
    assert not caught.value.lse_partial[1]['complete']


def test_rejected_context_and_valid_partial_LSE_remain_available_for_failed_audit(lse):
    x=torch.zeros(18,196,3456,dtype=torch.bfloat16);target=torch.zeros(18,196,1152,dtype=torch.bfloat16)
    plain,capture=cpu_API_pair('native',fault='capture_context')
    with pytest.raises(ValueError,match='context') as caught:
        lse.capture_lse('native',x,plain=plain,capture=capture,targets=target,synchronize=lambda:None)
    partial,state=caught.value.lse_partial
    assert 'partial.g6.r0.context_control' in partial and 'partial.g6.r0.context_lse_capture' in partial and 'partial.g6.r0.lse' in partial
    assert torch.count_nonzero(partial['partial.g6.r0.context_lse_capture'])==6*196*1152 and not state['complete']


def test_reentrant_capture_is_rejected_then_scope_is_available_for_a_fresh_capture(lse):
    x=torch.zeros(18,196,3456,dtype=torch.bfloat16);target=torch.zeros(18,196,1152,dtype=torch.bfloat16)
    plain,capture=cpu_API_pair('native')
    def nested(*a,**kw):return lse.capture_lse('native',x,plain=plain,capture=capture,targets=target,synchronize=lambda:None)
    with pytest.raises(ValueError,match='Nested'):
        lse.capture_lse('native',x,plain=nested,capture=capture,targets=target,synchronize=lambda:None)
    _,state=lse.capture_lse('native',x,plain=plain,capture=capture,targets=target,synchronize=lambda:None)
    assert state['complete'] and state['cleanup']['passed']


def test_FP32_numeric_equality_and_signed_zero_bit_equality_are_separate(lse):
    a=torch.tensor([0.,-0.,1.,2.],dtype=torch.float32);b=torch.tensor([-0.,0.,1.,2.],dtype=torch.float32)
    result=lse.compare_lse(a,b)
    assert result['numeric_exact'] and result['numeric_unequal']==0 and result['max_abs']==0.
    assert not result['bit_exact'] and result['bit_unequal']==2
    b[2]=torch.nextafter(b[2],torch.tensor(float('inf')))
    result=lse.compare_lse(a,b)
    assert result['numeric_unequal']==1 and result['bit_unequal']==3 and result['max_abs']==2**-23
    with pytest.raises(ValueError,match='FP32|finite|dtype'):lse.compare_lse(a.to(torch.bfloat16),b)


def test_row_cross_tab_matches_actual_head_token_channels_not_flattened_artifact_axes(lse):
    a=torch.zeros(18,16,196);b=a.clone();x=torch.zeros(18,196,1152,dtype=torch.bfloat16);y=x.clone()
    # Four disjoint row cases, spanning camera/frame/head boundaries.
    b[7,3,5]=1.;y[7,5,216]=1.  # camera1/frame1/head3/token5: both
    b[12,15,195]=1.             # camera2/frame0/head15/token195: only LSE
    y[5,17,72]=1.              # camera0/frame5/head1/token17: only context
    result=lse.row_cross_tab(a,b,x,y)
    assert result['rows']==56448 and result['both_different']==1 and result['lse_only']==1 and result['context_only']==1 and result['both_equal']==56445
    assert result['per_camera'][1]['both_different']==1 and result['per_camera'][2]['lse_only']==1
    assert result['per_frame'][5]['context_only']==1 and result['per_head'][15]['lse_only']==1


@pytest.mark.parametrize('q,k',[(0.,0.),(2.,3.)])
def test_CPU_reference_is_natural_log_with_FP32_scale_promoted_to_FP64(lse,q,k):
    x=torch.zeros(6,196,3456,dtype=torch.bfloat16);x[:,:,:1152]=q;x[:,:,1152:2304]=k
    result=lse.reference_lse(x)
    scale=float(torch.tensor(72**-.5,dtype=torch.float32))
    expected=math.log(196)+72*q*k*scale
    assert result.dtype==torch.float64 and result.device.type=='cpu' and result.shape==(6,16,196)
    assert torch.allclose(result,torch.full(result.shape,expected,dtype=torch.float64),rtol=0.,atol=2e-14)
    if q:assert abs(expected-(math.log(196)+72*q*k*(96**-.5)))>1.


NATIVE_SCHEMA='aten::_scaled_dot_product_flash_attention(Tensor query, Tensor key, Tensor value, float dropout_p=0., bool is_causal=False, bool return_debug_mask=False, *, float? scale=None) -> (Tensor output, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, Tensor rng_state, Tensor unused, Tensor debug_attn_mask)'


@pytest.mark.parametrize('bad',['none','dropout','debug','result_name','missing_result','scale_keyword'])
def test_exact_native_schema_drift_rejects_before_an_observable_call(lse,bad):
    text=NATIVE_SCHEMA
    if bad=='dropout':text=text.replace('dropout_p=0.','dropout_p=0.1')
    if bad=='debug':text=text.replace('return_debug_mask=False','return_debug_mask=True')
    if bad=='result_name':text=text.replace('rng_state','philox_seed')
    if bad=='missing_result':text=text.replace(', Tensor unused','')
    if bad=='scale_keyword':text=text.replace(', *, float?',', float?')
    schema=torch._C.parse_schema(text)
    if bad=='none':
        record=lse.validate_native_schema(schema)
        assert record['returns'][1]['name']=='logsumexp' and len(record['returns'])==9 and record['arguments'][-1]['kwarg_only']
    else:
        with pytest.raises(ValueError,match='schema|Flash|signature'):lse.validate_native_schema(schema)


def test_public_FA2_signature_is_bound_without_calling_the_dependency(lse):
    def good(q,k,v,dropout_p=0.,softmax_scale=None,causal=False,window_size=(-1,-1),softcap=0.,alibi_slopes=None,deterministic=False,return_attn_probs=False):
        raise AssertionError('Signature validation must not run a kernel')
    def wrong(q,k,v,dropout_p=.1):raise AssertionError('wrong dependency must not run')
    record=lse.validate_public_signature(good)
    assert record['arguments'][-1]=='return_attn_probs' and record['dropout_p']==0.
    with pytest.raises(ValueError,match='signature|FA2'):lse.validate_public_signature(wrong)


@pytest.mark.parametrize('bad',['none','missing_role','dtype','capture_context','qkv_after','counter','views'])
def test_eight_array_capture_requires_typed_raw_and_actual_controls(lse,bad):
    qkv=torch.zeros(18,196,3456,dtype=torch.bfloat16);target=torch.zeros(18,196,1152,dtype=torch.bfloat16)
    plain,capture=cpu_API_pair('native');raw,state=lse.capture_lse('native',qkv,plain=plain,capture=capture,targets=target,synchronize=lambda:None)
    if bad=='missing_role':del raw['g6.lse'];raw['dummy']=torch.zeros(1)
    if bad=='dtype':raw['g18.lse']=raw['g18.lse'].double()
    if bad=='capture_context':raw['g6.context_lse_capture']+=1
    if bad=='qkv_after':raw['after.qkv'][0,0,0]=1
    if bad=='counter':state['capture_calls']=3
    if bad=='views':state['groups']['g6'][1]['qkv_layout']['storage_offset']=0
    call=lambda:lse.validate_capture(raw,state,engine='native',qkv_reference=qkv,target_reference=target)
    if bad=='none':
        proof=call();assert proof['tensors']==8 and proof['FP32_LSE_arrays']==2 and proof['learned_weight_read_events']==0
    else:
        with pytest.raises(ValueError,match='raw|role|dtype|context|QKV|capture|view|offset|calls'):call()


def test_mathematical_reference_is_unchanged_by_key_sequence_permutation(lse):
    x=torch.zeros(6,196,3456,dtype=torch.bfloat16)
    x[:,:,:1152]=.5;x[:,0,1152:2304]=2.;x[:,1,1152:2304]=-1.
    permuted=x.clone();permuted[:,:,1152:2304]=x[:,:,1152:2304].flip(1)
    a=lse.reference_lse(x);b=lse.reference_lse(permuted)
    assert torch.allclose(a,b,rtol=0.,atol=3e-14)
    scale=float(torch.tensor(72**-.5,dtype=torch.float32))
    expected=math.log(math.exp(72*scale)+math.exp(-36*scale)+194.)
    assert abs(float(a[0,0,0])-expected)<3e-14


def test_complete_LSE_comparison_retains_per_camera_frame_and_head_metrics(lse):
    a=torch.zeros(18,16,196);b=a.clone();b[7,3,5]=.25
    result=lse.compare_lse(a,b)
    assert result['per_camera'][0]['numeric_exact'] and result['per_camera'][1]['numeric_unequal']==1
    assert result['per_frame'][7]['bit_unequal']==1 and result['per_frame'][7]['camera']==1 and result['per_frame'][7]['frame']==1
    assert result['per_head'][3]['max_abs']==.25 and result['per_head'][4]['numeric_exact']


@pytest.mark.parametrize('tamper',['count','arg_stride','return_stride','max_q_type'])
def test_complete_capture_cannot_publish_forged_operand_or_return_layout_metadata(lse,tamper):
    qkv=torch.zeros(18,196,3456,dtype=torch.bfloat16);target=torch.zeros(18,196,1152,dtype=torch.bfloat16)
    plain,capture=cpu_API_pair('native');raw,state=lse.capture_lse('native',qkv,plain=plain,capture=capture,targets=target,synchronize=lambda:None)
    row=state['groups']['g6'][0]
    if tamper=='count':row['capture_returns']['return_count']=8
    if tamper=='arg_stride':row['API_operand_layouts'][1]['live_stride'][1]=0
    if tamper=='return_stride':row['capture_returns']['returns'][1]['live_stride']=[4096,256,1]
    if tamper=='max_q_type':row['capture_returns']['returns'][4]={'type':'float','value':196.}
    with pytest.raises(ValueError,match='return|operand|layout|capture|metadata'):
        lse.validate_capture(raw,state,engine='native',qkv_reference=qkv,target_reference=target)


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def support_manifest(lse,tmp_path):
    from test_hyvla_frozen_vit_attention_bridge_example import r39_manifest
    bridge=load_module(ROOT/'examples/vla_autotune/hy_trace/frozen_vit_attention_bridge.py','lse_fixture_bridge')
    inputs,current39=r39_manifest(bridge,tmp_path)
    root39=tmp_path/'snapshots/round039/code/examples/vla_autotune/hy_trace';root39.mkdir(parents=True)
    for file in ('frozen_vit_attention_bridge.py','frozen_vit_prefix.py'):shutil.copyfile(ROOT/'examples/vla_autotune/hy_trace'/file,root39/file)
    # Keep complete historical metadata shape, but R40 must NOT follow its old
    # tensor-loader path into any Gold trace or learned checkpoint.
    inputs['legacy_records']={}
    for job,folder,source in (('309182','hy-frozen-attention-309182','ddbb5faa2ce76ee4fe1cf78016bd60f210f88c0c'),
            ('309409','hy-sdpa-backends-309409','d78ecfb07fc91fadff427d01ab44a8682639628b')):
        p=tmp_path/'results'/folder;p.mkdir(parents=True);(p/'SOURCE_GIT_COMMIT').write_text(source+'\n')
        inputs['legacy_records'][job+'.source']=lse.file_record(p/'SOURCE_GIT_COMMIT')
        for engine in ('official','native'):
            for role,name in (('report','result.json'),('raw','operator_outputs.safetensors')):
                f=p/engine/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('{}')
                inputs['legacy_records'][job+'.'+engine+'.'+role]=lse.file_record(f)
    root=tmp_path/'results/hy-vit-attention-bridge-319502';root.mkdir(parents=True)
    (root/'SOURCE_GIT_COMMIT').write_text('e058161ad2b58808d3acd07e9abf18b9b7e434d8\n');(root/'inputs.json').write_text(json.dumps(inputs))
    old=load_module(root39/'frozen_vit_attention_bridge.py','lse_fixture_old_namespace')
    parents={e:json.loads((tmp_path/'results/hy-vit-prefix-318174'/e/'result.json').read_text()) for e in ('official','native')}
    sources=old._source_records(inputs,{'R38_reports':parents})
    for e in ('official','native'):
        f=root/e;f.mkdir();(f/'operator_outputs.safetensors').write_bytes(b'raw placeholder for metadata rejection tests')
        state={'complete':True,'cleanup':{'passed':True}}
        doc={'kind':'hy_actual_vit_attention_context_projection_bridge','status':'bridge_complete','engine':e,'job_id':'319502',
            'source_revision':'e058161ad2b58808d3acd07e9abf18b9b7e434d8','runtime':parents[e]['runtime'],'torch_git_version':parents[e]['torch_git_version'],
            'bf16_reduced_precision_reduction':True,'checkpoint':parents[e]['checkpoint'],'directory_checkpoint':{'fixture':'directory'},
            'input_manifest':inputs,'source_records':sources,'weight_read_events':4,'selected_checkpoint_keys':[old.PREFIX+k for k in old.attention_keys()],
            'full_model_constructed':False,'full_prefix_constructed':False,'attention_module_constructed':e=='official','performance_measurement':False,
            'observations':{'actual':state,'cuts':state if e=='native' else None}}
        (f/'result.json').write_text(json.dumps(doc))
    (root/'native/projection_interaction.safetensors').write_bytes(b'interaction placeholder')
    current=tmp_path/'snapshots/round040/code/flash_rt/models/hyvla/pipeline_thor.py';current.parent.mkdir(parents=True);shutil.copyfile(current39,current)
    records={str(p.relative_to(root)):lse.file_record(p) for p in root.rglob('*') if p.is_file()}
    manifest={'kind':'hy_actual_QKV_weightless_LSE_inputs','task_root':str(tmp_path),'R39_records':records,
        'frozen_bridge':lse.file_record(root39/'frozen_vit_attention_bridge.py'),'current_native':lse.file_record(current)}
    return manifest,current


@pytest.mark.parametrize('bad',['none','source_marker','raw_stat','namespace','source_bytes','runtime','weight_keys','checkpoint','native_bytes'])
def test_metadata_only_R39_admission_preserves_original_namespaces_before_API(lse,tmp_path,bad):
    manifest,current=support_manifest(lse,tmp_path);root=tmp_path/'results/hy-vit-attention-bridge-319502'
    if bad=='source_marker':
        p=root/'SOURCE_GIT_COMMIT';p.write_text('bad source');manifest['R39_records']['SOURCE_GIT_COMMIT']=lse.file_record(p)
    if bad=='raw_stat':(root/'official/operator_outputs.safetensors').write_bytes(b'changed')
    if bad=='namespace':
        f=tmp_path/'snapshots/round040/code/examples/vla_autotune/hy_trace/frozen_vit_attention_bridge.py';f.parent.mkdir(parents=True)
        shutil.copyfile(manifest['frozen_bridge']['path'],f);manifest['frozen_bridge']=lse.file_record(f)
    if bad=='source_bytes':
        p=Path(manifest['frozen_bridge']['path']);p.write_text('# wrong original helper');manifest['frozen_bridge']=lse.file_record(p)
    if bad in ('runtime','weight_keys','checkpoint'):
        p=root/'native/result.json';doc=json.loads(p.read_text())
        if bad=='runtime':doc['runtime']={'torch':'other'}
        if bad=='weight_keys':doc['selected_checkpoint_keys'][-1]='other.weight'
        if bad=='checkpoint':doc['checkpoint']['bytes']+=1
        p.write_text(json.dumps(doc));manifest['R39_records']['native/result.json']=lse.file_record(p)
    if bad=='native_bytes':current.write_text('# changed current native');manifest['current_native']=lse.file_record(current)
    called=[]
    def consumer(bridge,prefix,r39,r38):
        assert 'round039' in bridge.__file__ and 'round038' in prefix.__file__
        assert r39['official']['job_id']=='319502' and r38['native']['job_id']=='318174'
        called.append('API consumer')
    call=lambda:lse.admit_before_api(manifest,engine='native',runtime={'torch':'new'},torch_git='b'*40,reduction_flag=True,
        current_native=current,current_bridge=ROOT/'examples/vla_autotune/hy_trace/frozen_vit_attention_bridge.py',
        current_prefix=ROOT/'examples/vla_autotune/hy_trace/frozen_vit_prefix.py',factory=consumer)
    if bad=='none':call();assert called==['API consumer']
    else:
        with pytest.raises(ValueError,match='R39|R38|source|record|path|namespace|runtime|checkpoint|weight'):call()
        assert not called


def test_nonweight_reader_rejects_a_learned_trace_role_before_any_get_tensor(lse,tmp_path):
    from safetensors.torch import save_file
    path=tmp_path/'mixed.safetensors';save_file({'model.visual.weight':torch.zeros(2),'qkv':torch.zeros(1)},str(path))
    seen=[]
    class Reader:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def keys(self):return ['qkv','model.visual.weight']
        def get_tensor(self,key):seen.append(key);raise AssertionError('No tensor read may precede role admission')
    with pytest.raises(ValueError,match='role|weight|nonweight'):
        lse.read_nonweight(lse.file_record(path),kind='R39.native',opener=lambda *a,**kw:Reader())
    assert not seen


def test_LSE_CLI_allocation_guard_precedes_any_support_or_API_access(lse,tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    result=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert result.returncode==0 and '--official-lse' in result.stdout
    result=subprocess.run([sys.executable,str(SCRIPT),'--engine','native','--inputs','/missing','--source-revision','a'*40,
        '--output',str(tmp_path/'unused')],env=env,text=True,capture_output=True)
    assert result.returncode!=0 and 'allocated GPU srun' in result.stderr and not (tmp_path/'unused').exists()


def sparse_zero_raw(path,raw):
    # Complete real safetensors header/storage, zero-filled sparse data, not a
    # fake reader. This bounds retained test disk usage for the physical166raw.
    header={};offset=0
    for name,value in sorted(raw.items()):
        size=value.numel()*value.element_size()
        header[name]={'dtype':{torch.bfloat16:'BF16',torch.float32:'F32',torch.float64:'F64'}[value.dtype],'shape':list(value.shape),'data_offsets':[offset,offset+size]};offset+=size
    blob=json.dumps(header,separators=(',',':')).encode();blob+=b' '*((-len(blob))%8)
    with path.open('wb') as file:file.write(struct.pack('<Q',len(blob)));file.write(blob);file.truncate(8+len(blob)+offset)


def full_support(lse,tmp_path):
    from test_hyvla_frozen_vit_prefix_example import prefix_raw
    from test_hyvla_frozen_vit_attention_bridge_example import actual_schema
    manifest,current=support_manifest(lse,tmp_path);root=tmp_path/'results/hy-vit-attention-bridge-319502'
    old=load_module(manifest['frozen_bridge']['path'],'lse_complete_support_bridge')
    inputs=json.loads((root/'inputs.json').read_text());r38root=tmp_path/'results/hy-vit-prefix-318174'
    for e,raw in zip(('official','native'),prefix_raw()):
        f=r38root/e/'operator_outputs.safetensors';sparse_zero_raw(f,raw)
        p=r38root/e/'result.json';doc=json.loads(p.read_text());doc['observations']['layouts']={k:{'shape':list(v.shape),'dtype':str(v.dtype),
            'live_stride':list(torch.empty(v.shape,device='meta').stride()),'storage_offset':0} for k,v in raw.items()}
        p.write_text(json.dumps(doc))
    for name in inputs['R38_records']:inputs['R38_records'][name]=lse.file_record(r38root/name)
    (root/'inputs.json').write_text(json.dumps(inputs));refs={};docs={}
    for engine in ('official','native'):
        raw,state,reference=actual_schema(old,engine);refs[engine]=reference
        doc=json.loads((root/engine/'result.json').read_text());doc['input_manifest']=inputs;doc['observations']={'actual':state,'cuts':None}
        if engine=='native':
            context=raw['context'];cut={'complete':True,'cleanup':{'passed':True},'restored_context_exact':True,'projection_calls':8,'restore_projection_calls':1,
                'original_context_layout':state['layouts']['context'],'controlled_context_value_substitution':True,'cells':{}}
            for cell in ('N18','N6','O18','O6'):
                for field in ('input','output'):raw[f'cuts.{cell}.{field}']=context
                cut['cells'][cell]={'views':[{'shape':[6 if cell.endswith('6') else 18,196,1152],'dtype':'torch.bfloat16',
                    'live_stride':[225792,1152,1],'storage_offset':i*6*225792 if cell.endswith('6') else 0,'shares_live_storage':True}
                    for i in range(3 if cell.endswith('6') else 1)]}
            raw.update({'cuts.restored_context':context,'cuts.restore_projection':context});doc['observations']['cuts']=cut
            I=torch.zeros((),dtype=torch.float64).expand(18,196,1152);sparse_zero_raw(root/'native/projection_interaction.safetensors',{'projection_interaction':I})
            doc['raw_proof']=old.validate_bundle(raw,engine=engine,actual_state=state,references=reference,cut_state=cut,
                old_context=context,old_reference=reference['post'],interaction_array=I)
        else:doc['raw_proof']=old.validate_capture(raw,state,engine=engine,references=reference)
        sparse_zero_raw(root/engine/'operator_outputs.safetensors',raw);docs[engine]=doc
        (root/engine/'result.json').write_text(json.dumps(doc))
    manifest['R39_records']={n:lse.file_record(root/n) for n in manifest['R39_records']}
    return manifest,current


@pytest.mark.parametrize('bad',['none','learned_trace_role','QKV_value','context_layout'])
def test_full_disk_admission_reads_only_205_nonweight_arrays_before_API_consumer(lse,tmp_path,monkeypatch,bad):
    import safetensors
    manifest,current=full_support(lse,tmp_path);root=tmp_path/'results/hy-vit-attention-bridge-319502'
    if bad=='learned_trace_role':
        path=root/'official/operator_outputs.safetensors'
        with path.open('rb') as file:n=struct.unpack('<Q',file.read(8))[0];header=json.loads(file.read(n))
        raw={name:torch.zeros(tuple(info['shape']),dtype=torch.bfloat16,device='meta') for name,info in header.items()}
        raw['model.visual.weight']=torch.zeros(1,dtype=torch.bfloat16);sparse_zero_raw(path,raw)
    if bad=='QKV_value':
        path=root/'native/operator_outputs.safetensors'
        with path.open('rb') as file:n=struct.unpack('<Q',file.read(8))[0];header=json.loads(file.read(n))
        with path.open('r+b') as file:file.seek(8+n+header['qkv']['data_offsets'][0]);file.write(struct.pack('<H',0x3f80))
    if bad=='context_layout':
        path=root/'native/result.json';doc=json.loads(path.read_text());doc['observations']['actual']['layouts']['qkv']['live_stride'][1]=0
        path.write_text(json.dumps(doc))
    manifest['R39_records']={name:lse.file_record(root/name) for name in manifest['R39_records']}
    allowed={str(Path(rec['path']).resolve()) for name,rec in manifest['R39_records'].items() if name.endswith('.safetensors')}
    inputs39=json.loads((root/'inputs.json').read_text())
    allowed.update(str(Path(rec['path']).resolve()) for name,rec in inputs39['R38_records'].items() if name.endswith('.safetensors'))
    original=safetensors.safe_open;events=[]
    def guarded(path,**kw):
        assert str(Path(path).resolve()) in allowed,'No Gold trace/checkpoint/legacy tensor reader is permitted'
        opened=original(path,**kw)
        class Audit:
            def __enter__(self):self.sf=opened.__enter__();return self
            def __exit__(self,*a):return opened.__exit__(*a)
            def keys(self):return self.sf.keys()
            def get_tensor(self,key):
                assert 'weight' not in key and 'bias' not in key;events.append((str(path),key));return self.sf.get_tensor(key)
        return Audit()
    monkeypatch.setattr(safetensors,'safe_open',guarded);consumed=[]
    def consume(value):
        consumed.append(True);assert value['QKV'].shape==(18,196,3456) and value['contexts']['native'].shape==(18,196,1152)
        assert value['learned_weight_read_events']==0 and value['nonweight_tensor_read_events']==205
    call=lambda:lse.load_inputs(manifest,engine='native',runtime={'torch':'new'},torch_git='b'*40,reduction_flag=True,
        current_native=current,current_bridge=ROOT/'examples/vla_autotune/hy_trace/frozen_vit_attention_bridge.py',
        current_prefix=ROOT/'examples/vla_autotune/hy_trace/frozen_vit_prefix.py',consumer=consume)
    if bad=='none':call();assert consumed==[True] and len(events)==205
    else:
        with pytest.raises(ValueError,match='role|raw|reference|anchor|layout|QKV'):call()
        assert not consumed


@pytest.mark.parametrize('bad',['none','version','drop0_branch','signature','stat'])
def test_installed_FA2_source_metadata_checks_before_dependency_import(lse,tmp_path,bad):
    folder=tmp_path/'site-packages/flash_attn';folder.mkdir(parents=True)
    (folder/'__init__.py').write_text('__version__ = "2.7.4.post1"\n')
    wrapper='''def flash_attn_func(q,k,v,dropout_p=0.,softmax_scale=None,causal=False,window_size=(-1,-1),softcap=0.,alibi_slopes=None,deterministic=False,return_attn_probs=False):
    raise RuntimeError("must not run during metadata check")
class FlashAttnFunc:
    def forward(ctx,q,k,v,dropout_p,softmax_scale,causal,window_size,softcap,alibi_slopes,deterministic,return_softmax,is_grad_enabled):
        return _wrapped_flash_attn_forward(q,k,v,return_softmax=return_softmax and dropout_p > 0)
'''
    if bad=='version':(folder/'__init__.py').write_text('__version__ = "other"\n')
    if bad=='drop0_branch':wrapper=wrapper.replace('return_softmax and dropout_p > 0','return_softmax')
    if bad=='signature':wrapper=wrapper.replace('return_attn_probs=False','return_attn_probs=True')
    (folder/'flash_attn_interface.py').write_text(wrapper);binary=folder.parent/'flash_attn_2_cuda.fixture.so';binary.write_bytes(b'not imported')
    records={k:lse.file_record(p) for k,p in {'package':folder/'__init__.py','interface':folder/'flash_attn_interface.py','binary':binary}.items()}
    if bad=='stat':binary.write_bytes(b'changed')
    call=lambda:lse.check_FA2_files(records)
    if bad=='none':
        proof=call();assert proof['version']=='2.7.4.post1' and proof['dropout0_return_softmax_false'] is True
    else:
        with pytest.raises(ValueError,match='FA2|version|signature|source|stat|dropout'):call()


def capture_fixture(lse):
    qkv=torch.zeros(18,196,3456,dtype=torch.bfloat16);context=torch.zeros(18,196,1152,dtype=torch.bfloat16)
    plain,capture=cpu_API_pair('official')
    raw,state=lse.capture_lse('official',tuple(v.clone() for v in qkv.split(6)),plain=plain,capture=capture,targets=context,synchronize=lambda:None)
    return qkv,context,raw,state


@pytest.mark.parametrize('bad',['none','job','source','runtime','weight_events','context'])
def test_current_job_old_LSE_predecessor_is_mandatory_before_native_API(lse,bad):
    qkv,context,raw,state=capture_fixture(lse)
    report={'kind':'hy_weightless_actual_QKV_LSE','status':'LSE_capture_complete','engine':'official','job_id':'7','source_revision':'c'*40,
        'runtime':{'torch':'old'},'torch_git_version':'a'*40,'bf16_reduced_precision_reduction':True,'input_manifest':{'scope':'fixture'},
        'source_records':{'driver':{}},'learned_weight_read_events':0,'model_constructed':False,'observations':state,
        'raw_proof':lse.validate_capture(raw,state,engine='official',qkv_reference=qkv,target_reference=context)}
    if bad=='job':report['job_id']='6'
    if bad=='source':report['source_revision']='d'*40
    if bad=='runtime':report['runtime']={'torch':'other'}
    if bad=='weight_events':report['learned_weight_read_events']=1
    if bad=='context':raw['g6.context_lse_capture']+=1
    called=[]
    call=lambda:lse.admit_official_lse(report,raw,job_id='7',source_revision='c'*40,runtime={'torch':'old'},torch_git='a'*40,reduction_flag=True,
        inputs={'scope':'fixture'},source_records={'driver':{}},qkv_reference=qkv,target_reference=context,factory=lambda:called.append('native API consumer'))
    if bad=='none':call();assert called==['native API consumer']
    else:
        with pytest.raises(ValueError,match='predecessor|context|weight|runtime|source|control'):call()
        assert not called


def test_late_failure_keeps_complete_actual_raw_and_computed_CPU_reference(lse,tmp_path,monkeypatch):
    from safetensors.torch import load_file
    qkv=torch.zeros(18,196,3456,dtype=torch.bfloat16);target=torch.zeros(18,196,1152,dtype=torch.bfloat16);plain,capture=cpu_API_pair('native')
    error=ValueError('late LSE qualification failed')
    def reject(*a,**kw):raise error
    monkeypatch.setattr(lse,'validate_capture',reject)
    with pytest.raises(ValueError,match='late LSE') as caught:
        lse.qualified_observation('native',qkv,plain=plain,capture=capture,targets=target,synchronize=lambda:None)
    assert caught.value is error
    lse.write_failure(tmp_path,error)
    raw=load_file(str(tmp_path/'unqualified_partial.safetensors'));ref=load_file(str(tmp_path/'unqualified_reference.safetensors'))
    assert len(raw)==8 and ref['mathematical_LSE'].dtype==torch.float64
    assert json.loads((tmp_path/'failure.json').read_text())['qualified'] is False and not (tmp_path/'result.json').exists()


def test_raw_save_failure_preserves_original_error_without_completed_report(lse,tmp_path,monkeypatch):
    import safetensors.torch
    qkv,target,raw,state=capture_fixture(lse)
    def fail(*a,**kw):raise OSError('LSE raw write failed')
    monkeypatch.setattr(safetensors.torch,'save_file',fail)
    with pytest.raises(OSError,match='raw write'):
        lse.publish_capture(tmp_path,raw,{'engine':'official','observations':state},qkv_reference=qkv,target_reference=target)
    primary=ValueError('primary LSE anchor failure');primary.lse_partial=(raw,state)
    lse.write_failure(tmp_path,primary)
    assert any('raw write failed' in x for x in primary.__notes__)
    assert json.loads((tmp_path/'failure.json').read_text())['error']=='primary LSE anchor failure' and not (tmp_path/'result.json').exists()


def test_same_job_predecessor_reader_admits_only_eight_observation_roles(lse,tmp_path):
    qkv,target,raw,state=capture_fixture(lse);path=tmp_path/'actual.safetensors';sparse_zero_raw(path,raw)
    loaded=lse.read_nonweight(lse.file_record(path),kind='R40.actual')
    assert set(loaded)=={'input.qkv','after.qkv','g6.context_control','g6.context_lse_capture','g6.lse','g18.context_control','g18.context_lse_capture','g18.lse'}
    assert loaded['g6.lse'].dtype==torch.float32 and loaded['input.qkv'].dtype==torch.bfloat16


def test_native_undefined_sequence_placeholders_reach_full_capture_validation(lse):
    q=torch.empty(6,16,196,72,device='meta',dtype=torch.bfloat16)
    meta=torch.ops.aten._scaled_dot_product_flash_attention.default(q,q,q,0.,False,False,scale=72**-.5)
    assert meta[2] is None and meta[3] is None
    result=list(api_result('native'));result[2:4]=meta[2:4]
    _,value,proof=lse.parse_capture_result(tuple(result),engine='native',batch=6)
    assert value.shape==(6,16,196) and proof['returns'][2:4]==[{'type':'NoneType','undefined_tensor':True}]*2
    inputs=torch.zeros(18,196,3456,dtype=torch.bfloat16);target=torch.zeros(18,196,1152,dtype=torch.bfloat16);plain,capture=cpu_API_pair('native')
    raw,state=lse.capture_lse('native',inputs,plain=plain,capture=capture,targets=target,synchronize=lambda:None)
    assert lse.validate_capture(raw,state,engine='native',qkv_reference=inputs,target_reference=target)['all_contexts_exact_R39']


@pytest.mark.parametrize('position',[0,1,6,7,8])
def test_native_None_is_rejected_outside_defined_undefined_sequence_positions(lse,position):
    result=list(api_result('native'));result[position]=None
    with pytest.raises(ValueError,match='context|LSE|return|schema|debug'):
        lse.parse_capture_result(tuple(result),engine='native',batch=6)


@pytest.mark.parametrize('bad',['debug','padding','NaN'])
def test_parser_rejection_retains_uncanonicalized_return_tensors_and_metadata(lse,tmp_path,bad):
    from safetensors.torch import load_file
    inputs=torch.zeros(18,196,3456,dtype=torch.bfloat16);target=torch.zeros(18,196,1152,dtype=torch.bfloat16)
    plain,base_capture=cpu_API_pair('native')
    def rejected(*a,**kw):
        result=list(base_capture(*a,**kw))
        if bad=='debug':result[-1]=torch.ones(1,dtype=torch.bfloat16)
        if bad=='padding':result[1]=torch.ones(6,16,256,dtype=torch.float32)
        if bad=='NaN':result[1][0,0,0]=float('nan')
        return tuple(result)
    with pytest.raises(ValueError,match='LSE|debug') as caught:
        lse.capture_lse('native',inputs,plain=plain,capture=rejected,targets=target,synchronize=lambda:None)
    partial,state=caught.value.lse_partial
    assert partial['partial.g6.r0.capture_return0'].shape==(6,16,196,72)
    value=partial['partial.g6.r0.capture_return1'];assert value.shape==((6,16,256) if bad=='padding' else (6,16,196))
    assert 'after.qkv' in partial and not state['complete']
    assert state['groups']['g6'][0]['unqualified_capture_returns'][2:4]==[{'type':'NoneType','undefined_tensor':True}]*2
    lse.write_failure(tmp_path,caught.value)
    saved=load_file(str(tmp_path/'unqualified_partial.safetensors'))
    if bad=='NaN':assert torch.isnan(saved['partial.g6.r0.capture_return1'][0,0,0])
    else:assert torch.equal(saved['partial.g6.r0.capture_return1'],value)
    failed=json.loads((tmp_path/'failure.json').read_text());assert not failed['qualified'] and not (tmp_path/'result.json').exists()


def test_return_snapshot_failure_adds_note_without_masking_the_parser_error(lse,monkeypatch):
    inputs=torch.zeros(18,196,3456,dtype=torch.bfloat16);target=torch.zeros(18,196,1152,dtype=torch.bfloat16)
    plain,base_capture=cpu_API_pair('native');original_clone=torch.Tensor.clone
    def rejected(*a,**kw):
        result=list(base_capture(*a,**kw));result[1]=torch.ones(6,16,256,dtype=torch.float32);return tuple(result)
    def broken_snapshot(self,*a,**kw):
        if self.shape==(6,16,256):raise OSError('raw-return snapshot failed')
        return original_clone(self,*a,**kw)
    monkeypatch.setattr(torch.Tensor,'clone',broken_snapshot)
    with pytest.raises(ValueError,match='LSE') as caught:
        lse.capture_lse('native',inputs,plain=plain,capture=rejected,targets=target,synchronize=lambda:None)
    assert any('raw-return snapshot failed' in note for note in caught.value.__notes__)
    assert 'partial.g6.r0.capture_return0' in caught.value.lse_partial[0]

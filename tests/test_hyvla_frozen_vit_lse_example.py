import importlib.util
import math
from pathlib import Path

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
    return (out,normalizer,torch.empty(0,dtype=torch.int32),torch.empty(0,dtype=torch.int32),196,196,
        torch.zeros(2,dtype=torch.uint64),torch.empty(0),torch.empty(0,dtype=torch.bfloat16))


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
        return out,normalizer,torch.empty(0,dtype=torch.int32),torch.empty(0,dtype=torch.int32),196,196,torch.zeros(2,dtype=torch.uint64),torch.empty(0),torch.empty(0)
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

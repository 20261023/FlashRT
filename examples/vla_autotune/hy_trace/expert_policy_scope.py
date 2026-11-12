"""Per-request eager native observers; Expert changes exist only inside NFE."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import importlib.util
from pathlib import Path
from types import MethodType


def _sibling(name):
    s=importlib.util.spec_from_file_location('hy_policy_'+name,Path(__file__).with_name(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


common=_sibling('diagnose');expert=_sibling('expert_mean_tree');order=_sibling('suffix_key_order')


def _same(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and bool((a==b).all())


def _binding(a,b):
    return a is b or (getattr(a,'__func__',None) is not None
        and a.__func__ is getattr(b,'__func__',None) and a.__self__ is getattr(b,'__self__',None))


def _eager(frontend):
    pipe=frontend.pipe
    if (frontend.use_cuda_graph is not False or frontend._graph_cache or frontend._vit_graph_cache
            or frontend.numerical_profile!='official_local_fp32' or pipe.numerical_profile!='official_local_fp32'):
        raise ValueError('Native Expert scope requires eager profile and empty graph caches')
    if (any(getattr(frontend,n,False) is not False for n in ('use_fp8','use_fp8_vit','use_fused','use_fp4','use_fused_quant','use_autotune','use_ffn_mega'))
            or any(getattr(pipe,n,False) is not False for n in ('_fp8','_fp4','_fused_attn'))
            or any(getattr(pipe.W,n,False) is not False for n in ('_exp_fp8_ready','_exp_ffn_mega_ready'))):
        raise ValueError('Unsupported quantized/fused/mega kernel path')
    if pipe.rms_eps!=1e-5:raise ValueError('Unsupported RMS epsilon')
    if (frontend.chunk,frontend.max_action_dim,frontend.n_heads,frontend.n_kv,pipe.head_dim)!=(40,32,16,4,128):
        raise ValueError('Unsupported native Expert geometry')


def _record(tensors,state,name,value):
    common._rec(tensors,state['tensor_metadata'],name,value)


def _prefix_record(tensors,state,name,cache,perm,kv_heads):
    import torch
    P=perm.numel();repeat=cache.shape[2]//kv_heads;prefix=cache[...,:P,:]
    compact=prefix[:,:,::repeat]
    if not torch.equal(compact.repeat_interleave(repeat,dim=2),prefix):raise ValueError('Native repeated prefix KV heads differ')
    canonical=compact.index_select(3,perm.argsort()).permute(0,1,3,2,4)
    _record(tensors,state,name,canonical)
    state['tensor_metadata'][name]['representation']='canonical_original_token_order_compressed_native_kv_heads'


@contextmanager
def expert_policy_request(frontend,*,enabled,request_id,separate_block,tensors,state,
                          expected_nfe=10,synchronize=None):
    import torch
    if type(enabled) is not bool or type(expected_nfe) is not int or expected_nfe<1 or not isinstance(request_id,str) or not request_id:
        raise ValueError('Invalid Expert request flags/count/identity')
    if getattr(frontend,'_expert_policy_active',False):raise ValueError('Nested/reentrant Expert request scope')
    if enabled and (not callable(separate_block) or getattr(separate_block,'__self__',None) is not None):raise ValueError('Separate block must be an unbound source template')
    _eager(frontend)
    if frontend.sampler!='euler' or frontend.num_steps!=expected_nfe:raise ValueError('Unsupported sampler/NFE schedule')
    pipe=frontend.pipe;synchronize=synchronize or (lambda:torch.cuda.synchronize(frontend.device))
    original={n:getattr(pipe,n) for n in ('_rms_norm','_block','_attn')}
    hold={'assembly':0,'graph':0,'inside':False};stack=ExitStack();primary=None
    state.update(request_id=request_id,enabled=enabled,complete=False,nfe=0,prefill_calls=0,steps=[],
        tensor_metadata={},cleanup={'passed':False,'status':'active'})
    missing=object();active=frontend.__dict__.get('_expert_policy_active',missing)
    frontend._expert_policy_active=True
    def unchanged_bindings():
        if any(not _binding(getattr(pipe,n),f) for n,f in original.items()):raise ValueError('Expert binding escaped into prefill/outside phase')
    def assemble(old):
        def call(*a,**kw):
            if hold['assembly'] or hold['graph'] or hold['inside']:raise ValueError('Prefix assembly generation reused in one request')
            result=old(*a,**kw)
            if len(result)!=6:raise ValueError('Unexpected prefix assembly tuple')
            pe,pad,att,mm,idx,full=result
            if (pe.ndim!=3 or pe.shape[0]!=1 or pe.shape[-1]!=2048 or pad.dtype!=torch.bool or att.dtype!=torch.bool
                    or mm.dtype!=torch.bool or pad.shape!=pe.shape[:2] or att.shape!=pad.shape or mm.shape!=pad.shape):raise ValueError('Invalid actual prefix assembly geometry')
            perm=torch.cat((mm[0].nonzero().flatten(),(~mm[0]).nonzero().flatten()))
            hold.update(assembly=1,perm=perm.clone(),P=pe.shape[1],pe=pe.clone(),pad=pad.clone(),att=att.clone(),mm=mm.clone(),idx=idx,full=full)
            state.update(prefix_rows=pe.shape[1],n_vis=int(mm.sum()),generation=1,
                prefix_segments={'idx_ranges':[list(r) for r in idx],'full_ranges':[list(r) for r in full]})
            for name,v in (('prefix.embeddings',pe),('prefix.pad_mask',pad),('prefix.block_mask',att),('prefix.modality_mask',mm),('prefix.permutation',perm)):_record(tensors,state,name,v)
            return result
        return call
    def graph(old):
        def call(*a,**kw):
            _eager(frontend);unchanged_bindings()
            if hold['assembly']!=1 or hold['graph'] or len(a)!=11 or kw.get('use_graph',True) is not False:raise ValueError('Missing fresh prefix generation or unsupported graph call')
            P,nvis,pe,pmask,pcos,psin,smask,scos,ssin,st,noise=a;perm=hold['perm'];S=1+frontend.chunk
            if type(P) is not int or P!=hold['P'] or nvis!=state['n_vis'] or not _same(pe,hold['pe'].index_select(1,perm)):
                raise ValueError('Actual sorted prefix/permutation binding differs')
            expected=frontend._apply_segment_mask(frontend._make_att_2d(hold['pad'],hold['att']),hold['idx'],hold['full'])
            expected=expected.index_select(1,perm).index_select(2,perm)[:,None]
            if not _same(pmask,expected) or smask.shape!=(1,1,S,P+S) or smask.dtype!=torch.bool:
                raise ValueError('Actual prefix/suffix mask geometry differs')
            if not _same(smask[...,:P],hold['pad'].index_select(1,perm)[:,None,None,:].expand(1,1,S,P)):
                raise ValueError('Actual suffix prefix-pad columns differ')
            if (noise.shape!=(1,40,32) or noise.dtype!=torch.float32 or st.shape!=(1,32) or st.dtype!=torch.bfloat16
                    or any(v.dtype!=torch.float32 for v in (pcos,psin,scos,ssin))):raise ValueError('Unsupported native suffix inputs')
            hold['graph']=1;hold['graph_args']=a
            before={i:v.clone() for i,v in enumerate(a) if isinstance(v,torch.Tensor)}
            for name,value in (('prefix.sorted_embeddings',pe),('prefix.mask_sorted',pmask),('prefix.cos_sorted',pcos),('prefix.sin_sorted',psin),
                    ('suffix.mask_sorted',smask),('suffix.cos',scos),('suffix.sin',ssin),('prepared.state',st),('input.noise',noise)):_record(tensors,state,name,value)
            out=old(*a,**kw)
            if any(not _same(a[i],v) for i,v in before.items()):raise ValueError('Native graph-boundary read-only input mutated')
            _record(tensors,state,'actions',out)
            return out
        return call
    def prefill(old):
        def call(*a,**kw):
            _eager(frontend);unchanged_bindings()
            if hold['graph']!=1 or hold['inside'] or state['prefill_calls'] or len(a)!=7 or kw:raise ValueError('Unexpected native prefill phase')
            pe,nvis,mask,cos,sin,k,v=a;P=hold['P'];S=1+frontend.chunk;ga=hold['graph_args']
            if nvis!=state['n_vis'] or any(not _same(x,y) for x,y in ((pe,ga[2]),(mask,ga[3]),(cos,ga[4]),(sin,ga[5]))):raise ValueError('Prefill sorted input pairing differs')
            if k.shape!=(32,1,16,P+S,128) or v.shape!=k.shape or k.dtype!=torch.float32 or v.dtype!=torch.bfloat16:raise ValueError('Unsupported native prefix KV geometry')
            out=old(*a,**kw);unchanged_bindings();state['prefill_calls']=1
            hold.update(kbuf=k,vbuf=v,prefix_k=k[...,:P,:].clone(),prefix_v=v[...,:P,:].clone())
            for name,cache in (('prefix.k',k),('prefix.v',v)):_prefix_record(tensors,state,name,cache,hold['perm'],frontend.n_kv)
            return out
        return call
    def velocity(old):
        def call(*a,**kw):
            _eager(frontend);unchanged_bindings()
            if hold['inside']:raise ValueError('Reentrant Expert velocity')
            if hold['graph']!=1 or state['prefill_calls']!=1 or len(a)!=9 or kw or state['nfe']>=expected_nfe:raise ValueError('Unexpected native velocity phase/count')
            x,time,st,mask,cos,sin,k,v,P=a;S=1+frontend.chunk;idx=state['nfe']
            if (P!=hold['P'] or k is not hold['kbuf'] or v is not hold['vbuf'] or x.dtype!=torch.float32 or x.shape!=(1,40,32)
                    or time.dtype!=torch.bfloat16 or st.dtype!=torch.bfloat16 or st.shape!=(1,1,1024)
                    or time.shape!=(1,1024) or mask.dtype!=torch.bool or mask.shape!=(1,1,S,P+S)
                    or cos.dtype!=torch.float32 or sin.dtype!=torch.float32):raise ValueError('Unsupported actual nine-argument Expert geometry')
            if not _same(k[...,:P,:],hold['prefix_k']) or not _same(v[...,:P,:],hold['prefix_v']):raise ValueError('Native prefix KV changed before Expert')
            before=tuple(t.clone() for t in a[:6]);step={'block_calls':0,'attention_calls':0};hold['inside']=True
            for name,value in ((f'step{idx}.latent_before',x),(f'step{idx}.time_embedding',time)):_record(tensors,state,name,value)
            _record(tensors,state,f'step{idx}.time',torch.tensor([frontend._embedding_times[idx]],dtype=torch.float32))
            state['tensor_metadata'][f'step{idx}.time']['source']='frontend_schedule_reconstruction_not_observed_scalar'
            if idx==0:
                _record(tensors,state,'suffix.state_embedding',st);hold['conditions']=tuple(t.clone() for t in a[2:6])
            elif any(not _same(t,b) for t,b in zip(a[2:6],hold['conditions'])):raise ValueError('Readonly suffix conditions changed across NFEs')
            def block_call(original_block):
                bound=MethodType(separate_block,pipe) if enabled else original_block
                def invoke(*b,**bk):
                    if (len(b)!=11 or b[1]!=S or b[2] is not None or b[10]!=P
                            or any(bk.get(n) is not None for n in ('fp8w','fp8v','fp8t','fp4v','fp4t','ffn_mk'))):raise ValueError('Unsupported Expert block/kernel routing')
                    step['block_calls']+=1;return bound(*b,**bk)
                return invoke
            def attn_call(original_attn):
                def invoke(q,key,value,m):
                    if (q.shape!=(1,16,S,128) or key.shape!=(1,16,P+S,128) or value.shape!=key.shape
                            or q.dtype!=torch.float32 or key.dtype!=torch.float32 or value.dtype!=torch.bfloat16 or m.dtype!=torch.bool or m.shape!=(1,1,S,P+S)):
                        raise ValueError('Unsupported actual Expert attention geometry')
                    step['attention_calls']+=1;return original_attn(q,key,value,m)
                return invoke
            try:
                with ExitStack() as inner:
                    # Preserve original own-attribute absence around norm_scope's
                    # standalone-owner restoration of a bound callable.
                    common.wrap_method(inner,pipe,'_rms_norm',lambda original:original)
                    norm=inner.enter_context(expert.norm_scope(pipe,use_tree=enabled,expected_calls=129))
                    common.wrap_method(inner,pipe,'_block',block_call)
                    common.wrap_method(inner,pipe,'_attn',attn_call)
                    if enabled:
                        ordered={};inner.enter_context(order.suffix_original_order(frontend,hold['perm'],state=ordered,synchronize=synchronize))
                        step['order']=ordered
                    out=old(*a,**kw)
                step['norm']=norm
            finally:
                hold['inside']=False
            unchanged_bindings()
            if any(not _same(t,b) for t,b in zip(a[:6],before)):raise ValueError('Expert readonly latent/condition input mutated')
            if not _same(k[...,:P,:],hold['prefix_k']) or not _same(v[...,:P,:],hold['prefix_v']):raise ValueError('Expert modified readonly native prefix KV')
            if out.shape!=x.shape or out.dtype!=torch.bfloat16 or not torch.isfinite(out).all():raise ValueError('Unexpected raw Expert velocity')
            if (step['block_calls']!=32 or step['attention_calls']!=32 or norm['by_role']!={'torch.bfloat16:1024':65,'torch.float32:128':64}
                    or norm['original_norm_calls']!=(0 if enabled else 129)):raise ValueError('Actual Expert phase call counts differ')
            if enabled and (step['order']['attention_calls']!=32 or step['order']['cleanup']['passed'] is not True):raise RuntimeError('Expert attention order cleanup failed')
            _record(tensors,state,f'step{idx}.velocity',out);step['prefix_readonly']=True
            state['steps'].append(step);state['nfe']+=1
            return out
        return call
    try:
        common.wrap_method(stack,frontend,'_assemble_prefix',assemble)
        common.wrap_method(stack,frontend,'_graph_forward',graph)
        common.wrap_method(stack,pipe,'prefill',prefill)
        common.wrap_method(stack,pipe,'evaluate_velocity',velocity)
        yield state
        _eager(frontend);unchanged_bindings()
        if hold['assembly']!=1 or hold['graph']!=1 or state['prefill_calls']!=1 or state['nfe']!=expected_nfe:raise ValueError('Incomplete native request phase/NFE counts')
        for name,value in (('prepared.tokens',frontend._lang_tokens),('prepared.token_mask',frontend._lang_masks)):_record(tensors,state,name,value)
        state['complete']=True
    except BaseException as error:
        primary=error;raise
    finally:
        errors=[]
        try:synchronize()
        except BaseException as error:errors.append('sync failed: '+str(error))
        try:stack.close()
        except BaseException as error:errors.append('binding restore failed: '+str(error))
        try:
            if active is missing:frontend.__dict__.pop('_expert_policy_active',None)
            else:frontend._expert_policy_active=active
            unchanged_bindings()
        except BaseException as error:errors.append('phase restore failed: '+str(error))
        state['cleanup']={'passed':not errors,'errors':errors,'status':'passed' if not errors else 'failed'}
        if errors:
            state['complete']=False;state['_failed_owners']=(frontend,pipe,original)
            if primary is not None:primary.add_note('; '.join(errors))
            else:raise RuntimeError('Expert request cleanup failed: '+'; '.join(errors))

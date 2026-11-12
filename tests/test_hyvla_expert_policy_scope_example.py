import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def policy():
    p=ROOT/'examples/vla_autotune/hy_trace/expert_policy_scope.py'
    s=importlib.util.spec_from_file_location('hy_policy_scope_test',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


class Pipe:
    numerical_profile='official_local_fp32';rms_eps=1e-5
    n_heads=16;n_kv=4;head_dim=128;q_dim=2048;kv_dim=512
    _fp8=False;_fp4=False;_fused_attn=False
    def __init__(self):
        self.norm_seen=[];self.block_selfs=[];self.attention_prefix=[];self.separate_seen=0
        self.phase='outside';self.mutate=None;self.fail=False
        self.W=SimpleNamespace(_exp_fp8_ready=False,_exp_ffn_mega_ready=False)
    def _rms_norm(self,x,w):self.norm_seen.append((self.phase,x.shape[-1]));return x
    def _attn(self,q,k,v,mask):
        self.attention_prefix.append(k[0,0,:self.P,0].clone())
        return torch.zeros_like(q,dtype=torch.bfloat16)
    def _block(self,hs,nvis,text,vis,qk,mask,cos,sin,k,v,off,**kw):
        self.block_selfs.append(self)
        w=torch.ones(1024,dtype=torch.bfloat16);wk=torch.ones(128,dtype=torch.bfloat16)
        self._rms_norm(hs,w)
        q=self._rms_norm(torch.ones(1,16,hs.shape[1],128),wk)
        self._rms_norm(torch.ones(1,4,hs.shape[1],128),wk)
        self._attn(q,k,v,mask);self._rms_norm(hs,w)
        if self.mutate=='prefix':k[...,0,:].add_(1)
        return hs
    def prefill(self,emb,nvis,mask,cos,sin,k,v):
        self.phase='prefill';self.P=emb.shape[1]
        self._rms_norm(torch.ones(1,self.P,2048,dtype=torch.bfloat16),torch.ones(2048,dtype=torch.bfloat16))
        self._rms_norm(torch.ones(1,16,self.P,128),torch.ones(128,dtype=torch.bfloat16))
        values=emb[0,:,0].view(1,1,1,self.P,1).expand(32,1,16,self.P,128)
        k[...,:self.P,:].copy_(values);v[...,:self.P,:].copy_(values)
    def evaluate_velocity(self,x,time,state,mask,cos,sin,k,v,P):
        self.phase='expert'
        if self.fail:raise RuntimeError('body failed')
        hs=torch.ones(1,41,1024,dtype=torch.bfloat16)
        for li in range(32):hs=self._block(hs,41,None,(),(),mask,cos,sin,k[li],v[li],P)
        self._rms_norm(hs,torch.ones(1024,dtype=torch.bfloat16))
        if self.mutate=='latent':x.add_(1)
        return torch.zeros_like(x,dtype=torch.bfloat16)


def separate(self,*a,**k):
    self.separate_seen+=1
    return Pipe._block(self,*a,**k)


class Front:
    numerical_profile='official_local_fp32';device='cpu';chunk=40;max_action_dim=32
    n_heads=16;n_kv=4;num_steps=1;sampler='euler';proj_width=1024
    use_cuda_graph=False;use_fp8=False;use_fp8_vit=False;use_fused=False;use_fp4=False
    use_fused_quant=False;use_autotune=False;use_ffn_mega=False
    def __init__(self,mm=(False,True,False,True)):
        self.pipe=Pipe();self.mm=torch.tensor(mm,dtype=torch.bool)
        self._graph_cache={};self._vit_graph_cache={};self._embedding_times=[1.]
        self._lang_tokens=torch.zeros(1,2,dtype=torch.long);self._lang_masks=torch.ones(1,2,dtype=torch.bool)
        self.wrong_perm=False
    def _assemble_prefix(self,merged):
        p=len(self.mm);emb=torch.arange(p).view(1,p,1).expand(1,p,2048).to(torch.bfloat16).contiguous()
        one=torch.ones(1,p,dtype=torch.bool)
        return emb,one.clone(),one.clone(),self.mm[None].clone(),[],[]
    @staticmethod
    def _make_att_2d(pad,att):
        c=att.long().cumsum(1);return (c[:,None,:]<=c[:,:,None]) & pad[:,None,:] & pad[:,:,None]
    def _apply_segment_mask(self,mask,*a):return mask
    def _graph_forward(self,P,nvis,pe,pmask,pcos,psin,smask,scos,ssin,state,noise,use_graph=False):
        k=torch.zeros(32,1,16,P+41,128);v=torch.zeros_like(k,dtype=torch.bfloat16)
        self.pipe.prefill(pe,nvis,pmask,pcos,psin,k,v);x=noise.clone()
        for i in range(self.num_steps):
            raw=self.pipe.evaluate_velocity(x,torch.ones(1,1024,dtype=torch.bfloat16),torch.ones(1,1,1024,dtype=torch.bfloat16),smask,scos,ssin,k,v,P)
            x.add_(-.1*raw.float())
        return x
    def run(self):
        pe,pad,att,mm,idx,full=self._assemble_prefix(None);P=pe.shape[1]
        perm=torch.cat((mm[0].nonzero().flatten(),(~mm[0]).nonzero().flatten()))
        if self.wrong_perm:perm=perm.flip(0)
        pm=self._make_att_2d(pad,att)[:,perm][:,:,perm][:,None]
        suffix_pad=torch.ones(1,41,dtype=torch.bool);suffix_att=torch.tensor([[1,1]+[0]*39],dtype=torch.bool)
        sm=torch.cat((pad[:,perm][:,None,:].expand(1,41,P),self._make_att_2d(suffix_pad,suffix_att)),2)[:,None]
        return self._graph_forward(P,int(mm.sum()),pe[:,perm],pm,torch.ones(1,1,P,128),torch.zeros(1,1,P,128),sm,
            torch.ones(1,1,41,128),torch.zeros(1,1,41,128),torch.zeros(1,32,dtype=torch.bfloat16),torch.zeros(1,40,32),use_graph=False)


def context(policy,f,enabled=True,**kw):
    tensors={};state={}
    ctx=policy.expert_policy_request(f,enabled=enabled,request_id='cpu/unit',separate_block=separate,
        tensors=tensors,state=state,expected_nfe=f.num_steps,synchronize=kw.pop('synchronize',lambda:None),**kw)
    return ctx,tensors,state


def no_shadows(f):
    assert not set(('_assemble_prefix','_graph_forward')) & f.__dict__.keys()
    assert not set(('_rms_norm','_block','_attn','prefill','evaluate_velocity')) & f.pipe.__dict__.keys()


@pytest.mark.parametrize('enabled',[False,True])
def test_real_self_prefill_isolation_finalnorm_counts_and_descriptor_restore(policy,enabled):
    f=Front();ctx,raw,state=context(policy,f,enabled)
    with ctx:result=f.run()
    assert state['complete'] and state['cleanup']['passed'] and state['nfe']==1 and state['prefill_calls']==1
    step=state['steps'][0];assert step['block_calls']==step['attention_calls']==32
    assert step['norm']['calls']==129 and step['norm']['by_role']=={'torch.bfloat16:1024':65,'torch.float32:128':64}
    assert step['norm']['original_norm_calls']==(0 if enabled else 129)
    assert f.pipe.separate_seen==(32 if enabled else 0) and all(x is f.pipe for x in f.pipe.block_selfs)
    assert len(f.pipe.norm_seen)==(2 if enabled else 131) and f.pipe.norm_seen[:2]==[('prefill',2048),('prefill',128)]
    expected=torch.arange(4).float() if enabled else torch.tensor([1.,3.,0.,2.])
    assert torch.equal(f.pipe.attention_prefix[0],expected)
    assert result.shape==(1,40,32) and raw['step0.velocity'].dtype==torch.bfloat16
    assert torch.equal(raw['prefix.k'][0,0,:,0,0],torch.arange(4).float())
    no_shadows(f)


def test_fresh_request_recomputes_permutation_and_prefix_length(policy):
    f=Front()
    for mm in ((False,True,False,True),(True,False,False,True,False,True)):
        f.mm=torch.tensor(mm);ctx,raw,state=context(policy,f)
        with ctx:f.run()
        assert state['prefix_rows']==len(mm) and raw['prefix.permutation'].numel()==len(mm)
        assert torch.equal(f.pipe.attention_prefix[-1],torch.arange(len(mm)).float())
        no_shadows(f)


@pytest.mark.parametrize('bad',['wrong_perm','latent','prefix','body'])
def test_wrong_mapping_or_mutation_cannot_complete_and_bindings_restore(policy,bad):
    f=Front()
    if bad=='wrong_perm':f.wrong_perm=True
    if bad in ('latent','prefix'):f.pipe.mutate=bad
    if bad=='body':f.pipe.fail=True
    ctx,raw,state=context(policy,f)
    with pytest.raises((ValueError,RuntimeError),match='prefix|readonly|read-only|body|sorted|permutation'):
        with ctx:f.run()
    assert not state['complete'];no_shadows(f)


@pytest.mark.parametrize('bad',['graphs','cache','vitcache','fp8','fused','profile','epsilon'])
def test_unsupported_graph_precision_or_kernel_paths_reject_without_mutation(policy,bad):
    f=Front()
    if bad=='graphs':f.use_cuda_graph=True
    if bad=='cache':f._graph_cache['stale']={}
    if bad=='vitcache':f._vit_graph_cache['stale']={}
    if bad=='fp8':f.pipe._fp8=True
    if bad=='fused':f.pipe._fused_attn=True
    if bad=='profile':f.numerical_profile='semantic_bf16'
    if bad=='epsilon':f.pipe.rms_eps=.01
    ctx,_,_=context(policy,f)
    with pytest.raises(ValueError,match='graph|profile|precision|kernel|epsilon|RMS|unsupported'):
        with ctx:pass
    no_shadows(f)


def test_nested_request_scope_and_bound_separate_owner_rejected(policy):
    f=Front();ctx,_,_=context(policy,f)
    with ctx:
        inner,_,_=context(policy,f)
        with pytest.raises(ValueError,match='nested|active|reentrant'):
            with inner:pass
        f.run()
    no_shadows(f)
    owner=SimpleNamespace();from types import MethodType
    with pytest.raises(ValueError,match='unbound|separate'):
        with policy.expert_policy_request(f,enabled=True,request_id='bound',separate_block=MethodType(separate,owner),tensors={},state={},expected_nfe=1,synchronize=lambda:None):pass


def test_sync_failure_never_completes_and_restores_own_binding(policy):
    f=Front();original=f.pipe._attn;f.pipe._attn=original
    def fail():raise RuntimeError('sync failed')
    ctx,_,state=context(policy,f,synchronize=fail)
    with pytest.raises(RuntimeError,match='sync|cleanup'):
        with ctx:f.run()
    assert not state['complete'] and not state['cleanup']['passed'] and f.pipe._attn is original
    del f.pipe._attn;no_shadows(f)

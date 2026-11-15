"""Latency evidence/timing helpers, not a model or inference backend."""
from __future__ import annotations

import math
from statistics import mean,median
import time
from urllib.parse import urlparse

MODELS={'pi0':'π0','pi05':'π0.5','fastwam':'FastWAM','hyvla':'Hy'}


def validate_row(row):
    if row.get('model') not in MODELS or row.get('origin') not in ('project','author'):raise ValueError('Unknown latency model/origin')
    implementation=row.get('implementation','');origin=row['origin'];evidence=row.get('evidence_kind')
    if not isinstance(implementation,str) or not implementation.strip():raise ValueError('Latency implementation must be explicit')
    if origin=='author' and any(s in implementation.lower() for s in ('flashrt','flash_rt','project')):
        raise ValueError('Official Native must use the author implementation, not FlashRT/project Torch graph')
    if origin=='project' and any(s in implementation.lower() for s in ('official','native')):
        raise ValueError('Project implementation must not be labelled official Native')
    source=urlparse(row.get('source_url',''))
    if source.scheme!='https' or not source.netloc or source.username or source.password or source.query:raise ValueError('Public latency source must be a noncredentialed HTTPS URL')
    if origin=='author' and 'github.com/20261023/FlashRT' in row['source_url']:raise ValueError('Project source is not an author Native source')
    if evidence not in ('measured','published','not_reported'):raise ValueError('Unknown latency evidence class')
    value=row.get('latency_ms');status=row.get('status')
    if status!='complete':
        if status not in ('failed','not_reported') or value is not None or row.get('p95_ms') is not None:raise ValueError('Non-complete latency must be NR, never a measured number')
        return row
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:raise ValueError('Latency must be positive finite milliseconds')
    if evidence=='not_reported':raise ValueError('Not-reported latency cannot be complete')
    tail=row.get('p95_ms')
    if tail is not None and (not isinstance(tail,(int,float)) or not math.isfinite(tail) or tail<=0):raise ValueError('Tail latency must be positive finite or NR')
    if evidence=='measured' and (not row.get('job_id') or not row.get('hardware') or row.get('statistic') not in ('median','mean')):
        raise ValueError('Measured latency requires actual job/hardware/statistic')
    return row


def comparison_ratio(ours,official):
    validate_row(ours);validate_row(official);reasons=[]
    if ours['origin']!='project' or official['origin']!='author':reasons.append('not a project/author pair')
    if any(r['status']!='complete' or r['evidence_kind']!='measured' for r in (ours,official)):
        reasons.append('not two complete same-allocation self measurements')
    for key in ('model','hardware','job_id','precision','input_signature','sampler','nfe','boundary','statistic'):
        if ours.get(key) is None or official.get(key) is None or ours.get(key)!=official.get(key):reasons.append(key+' differs or is NR')
    return {'ratio':None if reasons else official['latency_ms']/ours['latency_ms'],'reasons':reasons,
        'interpretation':'Latency ratio only; not a quality-equivalence or kernel-only speedup claim.'}


def render_table(rows):
    lines=['| 模型 | 方法／证据 | GPU | NFE | 延迟统计 | ms | P95 ms |',
        '|---|---|---|---:|---|---:|---:|']
    def text(value):return 'NR' if value is None else str(value).replace('|','\\|').replace('\n',' ')
    def number(value):return 'NR' if value is None else f'{value:.4f}'
    for row in rows:
        validate_row(row)
        label=('本项目方法' if row['origin']=='project' else '官方Native')+'：'+row['implementation']
        if row.get('selection_status')=='baseline':label+='（基线，未选优）'
        if row['evidence_kind']=='published':label+='（作者发布）'
        if row['status']!='complete':label+='（'+row['status']+'）'
        lines.append('| '+' | '.join((MODELS[row['model']],text(label),text(row.get('hardware')),text(row.get('nfe')),
            text(row.get('statistic')),number(row.get('latency_ms')),number(row.get('p95_ms'))))+' |')
    return '\n'.join(lines)+'\n'


def latency_statistics(samples):
    if not samples or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in samples):
        raise ValueError('Latency samples must be positive finite values')
    values=sorted(samples);position=.95*(len(values)-1);lower=int(position);fraction=position-lower
    p95=values[lower]*(1-fraction)+values[min(lower+1,len(values)-1)]*fraction
    return {'count':len(values),'p50_ms':median(values),'p95_ms':p95,'mean_ms':mean(values),'min_ms':min(values),'max_ms':max(values)}


def measure_calls(infer,requests,*,warmup,repeats,synchronize,expected_shape,clock_ns=time.perf_counter_ns):
    import numpy as np
    if not requests or type(warmup) is not int or warmup<0 or type(repeats) is not int or repeats<len(requests) or repeats%len(requests):
        raise ValueError('Measurement requires balanced complete request coverage and valid warmup/repeats')
    record={'complete':False,'latency_ms':[],'actions':[],'request_indices':[],'warmup':warmup,'repeats':repeats}
    def checked(value):
        if not isinstance(value,np.ndarray) or value.dtype!=np.float32 or value.shape!=expected_shape:raise ValueError('Official action shape/dtype differs')
        if not np.isfinite(value).all():raise ValueError('Official action must be finite')
    try:
        for index in range(warmup):
            value=infer(requests[index%len(requests)]);synchronize();checked(value)
        for index in range(repeats):
            request=requests[index%len(requests)];synchronize();started=clock_ns();value=infer(request);synchronize();elapsed=(clock_ns()-started)/1e6
            record['latency_ms'].append(elapsed);record['actions'].append(value);record['request_indices'].append(index%len(requests));checked(value)
        record['statistics']=latency_statistics(record['latency_ms']);record['complete']=True
    except BaseException as error:error.latency_partial=record;raise
    return record


def selected_measurement(record,workload,selection):
    if record.get('state')!='measured' or record.get('cleanup_failure') or record.get('partial_measurement'):
        raise ValueError('Selected measurement is not complete or cleanup failed')
    candidate=record.get('candidate',{});backend=candidate.get('backend',{});identity=record.get('backend_metadata',{})
    if candidate.get('source_ref')!=selection['source_ref']:raise ValueError('Selected measurement source differs')
    from flash_rt.vla.autotune.schema import decode_backend,encode_backend
    expected=selection.get('backend_config')
    if expected is None or encode_backend(decode_backend(backend))!=encode_backend(decode_backend(expected)):
        raise ValueError('The complete selected backend configuration differs from the frozen method')
    for key in ('model','backend','precision'):
        if backend.get(key)!=selection[key] or identity.get(key)!=selection[key]:raise ValueError('Selected implementation identity differs')
    required={r['request_id'] for r in workload['requests']}
    if set(record.get('workload_request_ids',[]))!=required:raise ValueError('Selected measurement request coverage differs')
    measurement=record['measurement'];samples=measurement['latency_ms'];metadata=measurement['sampler_metadata']
    if (len(samples)!=measurement['repeats'] or len(metadata)!=len(samples) or len(samples)<len(required) or len(samples)%len(required)
            or record.get('quality',{}).get('status')!='passed'):raise ValueError('Selected measurement sample/quality coverage differs')
    if any(m.get('sampler')!=selection['sampler'] or type(m.get('nfe')) is not int or m['nfe']!=selection['nfe'] for m in metadata):
        raise ValueError('Selected actual NFE differs from frozen method')
    if backend['model']=='hyvla' and any(m.get('numerical_profile')!=backend['options'].get('numerical_profile','semantic_bf16') for m in metadata):
        raise ValueError('The selected Hy numerical profile differs from actual sampler metadata')
    if backend['model']=='fastwam' and identity.get('execution_coverage',{}).get('cuda_graph') is not backend['options'].get('use_cuda_graph',False):
        raise ValueError('The selected FastWAM graph mode differs from observed execution coverage')
    statistics=latency_statistics(samples)
    result={**{k:v for k,v in selection.items() if k!='backend_config'},'origin':'project','evidence_kind':'measured','status':'complete','job_id':record['job_id'],
        'hardware':record['gpu']['name'],'statistic':'median','latency_ms':statistics['p50_ms'],'p95_ms':statistics['p95_ms'],
        'mean_ms':statistics['mean_ms'],'sample_count':len(samples),'request_count':len(required)}
    return validate_row(result)

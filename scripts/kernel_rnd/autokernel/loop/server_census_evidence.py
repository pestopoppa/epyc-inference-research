"""Native server census evidence checks; no launch or performance inference."""
import re


def graph_evidence(full_log, phase_log, model_path):
    """Read actual compute-graph host metadata; reserve graphs cannot supply it."""
    models = {pointer for pointer, path in re.findall(r'^CENSUS_MODEL model=(\S+) path=(.+)$', full_log, re.M)
              if path == model_path and pointer != '(nil)'}
    ended = {int(i) for i, rc in re.findall(r'^CENSUS_API_END id=(\d+) ctx=\S+ rc=(-?\d+)$', phase_log, re.M)
             if int(rc) == 0}
    batches = {}
    for decode, ctx, tokens, seqs in re.findall(
            r'^CENSUS_API_BEGIN id=(\d+) ctx=(\S+) tokens=(\d+) scope=api_batch seq_ids=([\d,]+)$', phase_log, re.M):
        ids = [int(value) for value in seqs.rstrip(',').split(',')]
        batches[int(decode)] = {'ctx':ctx, 'tokens':int(tokens),
                                'tokens_per_seq':{str(i):ids.count(i) for i in set(ids)},
                                'scope':'api_batch, coupled to native tensor metadata by decode_id'}
    graphs = {}
    for gid, decode, ctx, model, count, rc in re.findall(
            r'^CENSUS_GRAPH gid=(\d+) id=(\d+) ctx=(\S+) model=(\S+) nodes=(\d+) rc=(-?\d+)$', phase_log, re.M):
        if model in models and int(decode) in ended and ctx != '(nil)' and int(rc) == 0:
            if int(decode) not in batches or batches[int(decode)]['ctx'] != ctx:
                raise ValueError('compute graph lacks matching original API batch')
            graphs[int(gid)] = {'decode_id':int(decode), 'ctx':ctx, 'model':model,
                                 'declared_nodes':int(count), 'nodes':[],
                                 'api_batch':batches[int(decode)]}
    for line in phase_log.splitlines():
        if not line.startswith('CENSUS_NODE '):
            continue
        row = dict(re.findall(r'(\w+)=(\S+)', line))
        gid = int(row['gid'])
        if gid in graphs:
            graph = graphs[gid]
            if row['ctx'] != graph['ctx'] or row['model'] != graph['model'] or int(row['id']) != graph['decode_id']:
                raise ValueError('graph node identity mismatch')
            for key in ('ne', 'src0_ne', 'src1_ne'):
                if key in row:
                    row[key] = list(map(int, row[key].split(',')))
                    if len(row[key]) != 4:
                        raise ValueError('tensor ABI shape requires four dimensions')
            graph['nodes'].append(row)
    counts, widths, summaries, types = {}, set(), [], {}
    for graph in graphs.values():
        rows = graph['nodes']
        if len(rows) != graph['declared_nodes'] or {int(x['node']) for x in rows} != set(range(len(rows))):
            raise ValueError('partial or duplicate native graph metadata')
        dense_widths = set()
        for row in rows:
            counts.setdefault(row['op'], {}).setdefault(row['backend'], 0)
            counts[row['op']][row['backend']] += 1
            type_key = ':'.join(row.get(key, 'missing') for key in ('op','dst_type','src0_type','src1_type'))
            types[type_key] = types.get(type_key, 0) + 1
            if row['op'] in ('MUL_MAT', 'MUL_MAT_ID') and 'src1_ne' in row:
                widths.add(tuple(row['src1_ne']))
                if row['op'] == 'MUL_MAT':
                    dense_widths.add(row['src1_ne'][1])
        summaries.append({key:value for key,value in graph.items() if key != 'nodes'} |
                         {'dense_matmul_ne11':sorted(dense_widths)})
    return {'nodes_total':sum(sum(row.values()) for row in counts.values()),
            'op_backend':counts, 'op_type':types, 'graphs':summaries,
            'matmul_src1_shapes':[list(row) for row in sorted(widths)],
            'scope':'native compute cgraph host tensor metadata; no tensor-data reads'}


def tensor_shape_observed(shape, evidence):
    """Require both native API sequence mapping and same-call dense tensor width."""
    per_seq = shape.n_tokens + 1 if shape.phase == 'verify' else shape.n_tokens
    wanted_ids = {str(i) for i in range(shape.n_seq)}
    for graph in evidence['graphs']:
        batch = graph['api_batch']
        counts = batch['tokens_per_seq']
        if set(counts) != wanted_ids:
            continue
        if shape.phase == 'prefill':
            exact = batch['tokens'] == shape.n_tokens and sum(counts.values()) == shape.n_tokens
            widths = {shape.n_tokens}
        else:
            exact = (batch['tokens'] == sum(counts.values()) and
                     all(value == per_seq for value in counts.values()))
            widths = {per_seq, per_seq * shape.n_seq}
        if exact and widths.intersection(graph['dense_matmul_ne11']):
            return True
    return False


def shape_evidence(shape, startup, phase_log, requests, responses, samples):
    """Check observed server counters against the frozen requested envelope.

    Slot concurrency is a server observation, not proof of one fused tensor's
    sequence count. Keep that distinction in the emitted structural evidence.
    Missing counters refuse rather than relabel decode length as verification.
    """
    width = shape.n_seq
    ids = set(range(width))
    configured = set(map(int, re.findall(r'n_seq_max\s*=\s*(\d+)', startup)))
    ubatches = set(map(int, re.findall(r'n_ubatch\s*=\s*(\d+)', startup)))
    prompt_rows = {int(slot): int(tokens) for slot, tokens in re.findall(
        r'id\s+(\d+)\s*\|\s*task\s+\d+\s*\| new prompt,.*?task.n_tokens = (\d+)', phase_log)}
    verified = {}
    for slot, accepted, drafted in re.findall(
            r'id\s+(\d+)\s*\|\s*task\s+\d+\s*\| accepted\s+(\d+)/\s*(\d+) draft tokens', phase_log):
        if int(accepted) > int(drafted):
            raise ValueError('native accepted count exceeds drafted count')
        verified.setdefault(int(slot), set()).add(int(drafted))
    occupancy = []
    for sample in samples:
        if sample.get('phase') == 'requests':
            occupancy.append({row['id'] for row in sample.get('slots', [])
                              if row.get('is_processing') is True})
    guards = {
        'configured_parallelism': width in configured,
        'request_count': len(requests) == len(responses) == width,
        'slot_occupancy_observed': any(ids <= busy for busy in occupancy),
        'terminal_responses': all(row.get('stop') is True and
                                  row.get('timings', {}).get('predicted_n', 0) > 0
                                  for row in responses),
        'prompt_counters_present': ids <= prompt_rows.keys(),
    }
    if shape.phase == 'prefill':
        guards['configured_ubatch'] = shape.n_tokens in ubatches
        # A server may keep a final checkpoint suffix outside the first ubatch.
        # The whole frozen prompt and the observed tensor ubatch are distinct.
        guards['frozen_token_width'] = all(isinstance(row.get('prompt'), list) and
                                          len(row['prompt']) >= shape.n_tokens for row in requests)
        guards['native_prompt_width'] = all(isinstance(row.get('prompt'), list) and
            prompt_rows.get(i) == len(row['prompt']) for i, row in enumerate(requests))
    elif shape.phase == 'verify':
        guards['native_verify_width'] = all(shape.n_tokens in verified.get(i, set()) for i in ids)
        guards['native_draft_counters'] = all(row.get('timings', {}).get('draft_n', 0) > 0
                                             for row in responses)
        guards['request_verify_budget'] = all(row.get('speculative.n_max') == shape.n_tokens
                                              for row in requests)
    elif shape.phase == 'decode':
        guards['plain_decode_request'] = all(row.get('speculative.n_max') == 0 for row in requests)
        guards['plain_decode_response'] = all(row.get('timings', {}).get('draft_n', 0) == 0
                                               for row in responses)
    else:
        raise ValueError('unknown phase')
    return {'guards': guards, 'observed': all(guards.values()),
            'configured_n_seq_max': sorted(configured), 'configured_ubatch': sorted(ubatches),
            'native_prompt_tokens_by_slot': prompt_rows,
            'native_verified_draft_widths': {k: sorted(v) for k, v in verified.items()},
            'occupied_slot_sets': [sorted(row) for row in occupancy],
            'sequence_scope': 'observed concurrent server slots; tensor-level n_seq unproven'}

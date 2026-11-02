"""Resolve historical LOMO targets by graph identity, never prediction scores."""
import re
import numpy as np
import pandas as pd

# HMDB0000654 is a secondary accession in this primary record (checked 2026-10-30).
HMDB_ALIASES = {'HMDB0000654': 'HMDB0001644'}
ALIAS_SOURCE = 'https://www.hmdb.ca/metabolites/HMDB0001644'


def select_historical_lomo_panel(historical, var, nodes, graph, counts, minimum=10):
    hmdb_nodes = {}
    for node, identifier in enumerate(graph.hmdb_ids):
        if identifier:
            hmdb_nodes.setdefault(identifier, []).append(node)
    retained, excluded = [], []
    seen = set()
    for entry in sorted(historical, key=lambda row: row['fold_idx']):
        identifier = entry['metabolite_id']
        fallback = re.fullmatch(r'metabolite_(\d+)', identifier)
        matches = [int(fallback[1])] if fallback else hmdb_nodes.get(identifier, [])
        if len(matches) != 1 or not 0 <= matches[0] < len(graph.hmdb_ids):
            excluded.append({**entry, 'reason': 'historical graph identity unresolved'})
            continue
        node = matches[0]
        if (graph.hmdb_ids[node] or f'metabolite_{node}') != identifier:
            raise ValueError('Historical graph identity changed.')
        if 'metabolite_idx' in entry and int(entry['metabolite_idx']) != node:
            raise ValueError('Historical node index and identifier disagree.')
        if node in seen:
            raise ValueError('Duplicate historical graph node.')
        seen.add(node)
        columns = np.flatnonzero(np.asarray(nodes) == node)
        if len(columns) != 1:
            excluded.append({**entry, 'node': node, 'reason': 'no unique non-excluded assay feature for graph node'})
            continue
        column = int(columns[0]); assay_hmdb = var.iloc[column].get('HMDB_trailing_zeros')
        graph_hmdb = graph.hmdb_ids[node]
        if graph_hmdb and pd.notna(assay_hmdb):
            canonical = HMDB_ALIASES.get(str(assay_hmdb), str(assay_hmdb))
            if canonical != HMDB_ALIASES.get(str(graph_hmdb), str(graph_hmdb)):
                excluded.append({**entry, 'node': node, 'feature': str(var.index[column]),
                    'reason': 'conflicting assay and graph HMDB identities; requires independent resolution'})
                continue
        count = int(counts[column])
        if count < minimum:
            excluded.append({**entry, 'node': node, 'reason': f'fewer than {minimum} measured outcomes'})
            continue
        retained.append(dict(feature=str(var.index[column]), node=node,
                             historical_fold=int(entry['fold_idx']), n_observed=count))
    return {'retained': retained, 'excluded': excluded}

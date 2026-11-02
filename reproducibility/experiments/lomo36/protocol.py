"""Fixed reviewer ablations; no new architecture or outcome-based task choice."""
import copy
import hashlib

VARIANTS = ('full', 'no_mcr', 'no_attention', 'no_balance', 'no_correlation', 'no_variance', 'no_aux')
SEEDS = (42, 43, 44)


def architecture(variant):
    from src.models.components import EnhancedModelConfig
    if variant not in VARIANTS:
        raise ValueError('Unknown ablation.')
    features = ['residual_connections', 'cross_modal_attention', 'mcr', 'flux_messages', 'mass_balance_projection']
    if variant == 'no_mcr': features.remove('mcr')
    if variant == 'no_attention': features.remove('cross_modal_attention')
    arch = EnhancedModelConfig.with_features(features)
    arch.layer_type = 'Relational'; arch.structural_loss_weight = 1.
    physics = dict(lambda_balance=.5, lambda_correlation=.1, lambda_variance=1., lambda_kl=0.)
    for key in ['balance', 'correlation', 'variance']:
        if variant in ('no_' + key, 'no_aux'):
            physics['lambda_' + key] = 0.
    return arch, physics


def digest_parameters(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def matched_model(config, seed, **dimensions):
    """Identical common parameter tensors even when component removal changes RNG use."""
    import torch
    from src.models.factory import create_model
    full_config = copy.deepcopy(config)
    full_config.enhanced_config = architecture('full')[0].to_dict()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = create_model(full_config, **dimensions)
    reference_digest = digest_parameters(reference)
    model = create_model(config, **dimensions)
    reference_state = reference.state_dict(); state = model.state_dict()
    matching = {k: reference_state[k] for k in state if k in reference_state and state[k].shape == reference_state[k].shape}
    state.update(matching); model.load_state_dict(state, strict=True)
    record = dict(full_reference_parameter_sha256=reference_digest, model_parameter_sha256=digest_parameters(model),
                  shared_state_entries=len(matching), parameters=sum(p.numel() for p in model.parameters()))
    return model, record


def make_tasks(panel):
    if len(panel) != 36 or len({p['node'] for p in panel}) != 36:
        raise ValueError('Expected the 36 uniquely mapped historical targets.')
    tasks = []
    def append(kind, variant, seed, target=None, stage='core'):
        identity = 'sample' if kind == 'sample' else 'lomo_n' + str(target['node'])
        tasks.append(dict(index=len(tasks), id=f'{identity}_{variant}_s{seed}', kind=kind,
                          variant=variant, seed=seed, stage=stage, **(target or {})))
    # Explicit first stage: the core panel and sample-holdout ablations.
    for p in panel:
        append('lomo', 'full', 42, p)
    for seed in SEEDS:
        for variant in VARIANTS:
            append('sample', variant, seed)
    for p in panel[:5]:
        for seed in SEEDS:
            for variant in VARIANTS:
                if seed == 42 and variant == 'full': continue
                append('lomo', variant, seed, p, stage='lomo_ablation')
    assert len(tasks) == 157 and len({t['id'] for t in tasks}) == 157
    return tasks

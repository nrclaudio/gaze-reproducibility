# LOMO Improvement Ideas

This document collects the main strategies we discussed for improving leave-one-metabolite-out (LOMO) performance in the concat-SMILES model. They are grouped roughly by effort and focus area so we can prioritise experiments.

## Data & Sampling Strategies

- **Neighbour coverage stratification**: keep using the new coverage scores (direct + far) to select held-out metabolites. Sample across quantiles so the evaluation covers both learnable and hard regions.
- **Centrality-aware selection**: pre-compute node centrality (degree, betweenness) and combine it with coverage when choosing LOMO folds or weighting losses. High-centrality nodes deserve special attention.
- **Neighbour dropout**: during training, randomly drop a fraction of currently measured neighbour metabolites (temporarily flipping their mask to `False`) before the forward pass. This forces the model to cope with sparse supervision similar to LOMO.
- **Neighbour-aware curriculum**: bias early epochs toward high-coverage nodes, then gradually mix in low-coverage metabolites so the model learns robust context before tackling sparse regions.

## Auxiliary Objectives

- **Masked-neighbour reconstruction**: treat the neighbour dropout as a self-supervised task—compute an auxiliary MSE on the masked metabolites and add it as a low-weight loss. This simulates mini LOMO episodes within each batch.
- **Neighbour dependency prediction**: pre-compute correlation/co-occurrence statistics for metabolite pairs and train a small classifier (or regressor) to predict them from embeddings. The model learns statistical structure that transfers to cold-start metabolites.
- **Reaction consistency classifier**: distinguish real reaction neighbourhoods from perturbed ones (e.g., shuffled stoichiometry) to encode physics priors without relying on brittle mass-balance losses.
- **Conditional density head**: predict simple distribution parameters (mean/variance) of a metabolite conditioned on labelled neighbours. This provides priors when the direct label is missing.

## Architectural Adjustments

- **Coverage-gated modules**: enable MCR, flux messaging, or mass-balance projection only when coverage exceeds a threshold; fall back to the baseline head for sparse nodes.
- **Neighbour-conditioned bias**: add a light MLP that pools labelled neighbour embeddings and outputs a bias term blended with the baseline prediction—a learnable, differentiable analogue to k-NN.
- **Centrality features**: append log-degree or other centrality scalars to node features so the encoder can distinguish hubs from leaves.
- **Hybrid heads**: mix baseline and MCR outputs via a learned gate so the model can down-weight noisy attention when neighbours provide little signal.

## Training Protocols

- **Contrastive + masked pretraining**: retain the current contrastive pretraining, then run a masked-neighbour self-supervised phase (or keep a light masked loss during fine-tuning). Contrastive warm-start stabilises the encoder before the masked task kicks in.
- **Two-stage fine-tuning**: after standard training converges, run a short refinement stage with small physics weights (`λ_bal` etc.) restricted to high-coverage metabolites; this prevents physics from overwhelming sparse nodes.
- **Gradient accumulation**: if batch sizes must increase for certain auxiliary tasks (e.g., contrastive), use the existing gradient-accumulation support to simulate larger batches without exceeding memory.

## Evaluation Enhancements

- Extend LOMO summaries with coverage × centrality buckets, auxiliary-task accuracy, and uncertainty estimates so we can isolate which interventions help which metabolite regimes.
- Log per-fold statistics (coverage score, centrality, neighbour counts) alongside R²/RMSE; this makes it easier to track improvements over baseline.

## Quick Wins to Prototype First

1. Implement neighbour dropout + masked reconstruction during training (joint auxiliary loss).  
2. Add coverage-gated fallbacks for MCR / flux messaging so sparse nodes rely on the baseline predictor.  
3. Augment node features with centrality scores and monitor their effect in LOMO with the new coverage-aware selection.  
4. Add a simple neighbour-dependency classifier head using pre-computed correlation labels and measure whether it boosts low-coverage metabolites.  
5. Warm-start LOMO folds with contrastive encoders + a short masked-pretrain stage before supervised fine-tuning.

These steps are incremental, reuse the current infrastructure, and specifically target the sparse-label regime that makes LOMO challenging.

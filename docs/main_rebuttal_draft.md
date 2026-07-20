# Main-Track Rebuttal Draft

We thank the reviewers for the constructive feedback. The main concerns were
scope, benchmark breadth, one-step behavioral relevance, and whether VISA is
only diagnostic. We added three targeted results. First, SAT-Bench++ adds 1,000
richer fixed-observation counterfactuals covering compositional and
temporal/procedural semantics; the gap remains strong (overall action
sensitivity 0.061, SAG 0.939). Second, VISA-Rerank extends VISA from
flag/defer to schema-conditioned action-boundary reranking: on target swaps,
wrong/ambiguous exposure drops from 0.318 to 0.000 among allowed actions; on
SAT-Bench++ compositional and temporal/procedural splits, allowed aligned
exposure is also 1.000. Third, K=50/K=100 rollout probes provide preliminary
independent behavior evidence, while we continue to avoid claiming full task
success. As auxiliary evidence on actionability, a small phase-aware
closed-loop correction pilot shows that selective action-boundary correction can
improve correct lift over Native without wrong-object contact/lift, though a
generic selective baseline is stronger, so we do not claim schema-specific
closed-loop superiority.

## Benchmark Breadth

To address the concern that SAT-Bench covers a narrow family of semantic
perturbations, we added SAT-Bench++ with two new fixed-observation
counterfactual families. The compositional split varies attribute, relation, or
attribute+relation semantics while holding the observation and robot state
fixed. The temporal/procedural split changes the first required subgoal, e.g.,
which object or affordance should be approached first. Across 1,000 examples,
semantic recovery remains 1.000 while native action sensitivity remains 0.061
(SAG 0.939). The gap also persists within the new splits: compositional action
sensitivity is 0.070 (SAG 0.930), and temporal/procedural action sensitivity is
0.048 (SAG 0.953). This shows the phenomenon is not limited to simple
target-name or spatial-relation swaps.

## VISA-Rerank

To address the concern that VISA is only diagnostic, we added VISA-Rerank. A
naive single schema prompt rewrite did not improve action sensitivity, so we do
not claim prompt-level repair. Instead, VISA-Rerank generates
schema-conditioned candidates and uses the action-boundary checker to allow,
defer, or select a target-consistent candidate. On N=600 target-swap examples,
Native has 0.682 aligned exposure and 0.318 wrong/ambiguous exposure;
VISA-Rerank allows 0.938 of cases and achieves 1.000 aligned exposure and 0.000
wrong/ambiguous exposure among allowed actions.

We further tested whether this interface generalizes beyond target swaps. On
compositional SAT-Bench++ examples, Native aligned exposure is 0.513, while
VISA-Rerank allows 0.890 and achieves 1.000 aligned exposure among allowed
actions. On temporal/procedural examples, Native aligned exposure is 0.590,
while VISA-Rerank allows 0.907 and again achieves 1.000 aligned exposure among
allowed actions. Thus, the interface is not limited to simple target
replacement.

## Behavior Evidence

Because ActCheck is used for reranking, we also ran independent K=50 and K=100
short-horizon trajectory probes that do not use ActCheck as the evaluation
metric.
In the K=100 all-case setting (N=50), Native executes every command while
VISA-Rerank may defer. VISA-Rerank improves final target preference (0.0002 to
0.0083), integrated target preference (0.0178 to 0.0216), and wrong-object
approach (0.720 to 0.600), but it has high defer rate (0.720) and does not
improve the first-close proxy. A K=50/N=100 probe shows the same qualitative
pattern: final preference improves from -0.0001 to 0.0032, integrated
preference from 0.0177 to 0.0194, and wrong-object approach drops from 0.640 to
0.520.

We therefore report a matched allowed-subset analysis to characterize the
coverage-quality tradeoff. On the same 23 K=100 rollout instances where
VISA-Rerank exposes at least one action, it improves final target preference
(-0.0047 to 0.0174) and integrated target preference (0.0467 to 0.0559)
relative to Native. However, wrong-object approach remains mixed in this
matched subset (0.739 for Native vs. 0.696 for VISA-Rerank), and no method
solves first-close contact. We therefore use these rollouts as short-horizon
anti-circularity evidence for improved trajectory-level preference under a
coverage-quality tradeoff, not as contact, lift, manipulation, or task-level
success. Target preference is a trajectory diagnostic here; it should not be
presented as evidence that the robot completes the manipulation.

We also added a Best-of-N no-schema baseline to separate schema conditioning
from the benefit of candidate generation plus action-boundary selection. This
baseline is strong: in K=100 all-case rollouts it improves final target
preference to 0.0069, integrated preference to 0.0208, and wrong-object
approach to 0.540. VISA-Rerank remains best on final/integrated target
preference, while the no-schema baseline is better on wrong-object suppression.
We will report this honestly: the action-boundary selection mechanism is an
important part of the gain, and the schema contributes structured grounding,
admissibility/defer decisions, interpretability, and handoff semantics rather
than being the sole source of improvement.

As an auxiliary check on actionability, we also tested a stricter phase-aware
selective residual interface in true LIBERO env-step execution. Unlike the
earlier continuous residual pilot, the selective variant modifies only
high-confidence wrong-target approach actions with small xy corrections and
then releases control back to the native policy near the target. On 10
base-policy-competent counterfactual episodes, Native reaches correct lift on
7/10 episodes, VISA-selective reaches 8/10, and an oracle-target selective
variant reaches 9/10, with no wrong-object contact or lift. VISA-selective
repairs 2/3 Native lift failures but breaks 1/7 Native lift successes. A
generic selective residual baseline reaches 9/10, so we do not attribute this
closed-loop gain specifically to schema conditioning. We use this only as
preliminary evidence that action-boundary correction can be made
non-destructive and behaviorally actionable, not as full controller success.

To test whether schema conditioning affects the candidate pool before reranking,
we added compute-matched N-sweep and schema-intervention analyses. In the
existing full candidate pools, generic prompting remains a strong baseline and
often catches up by larger N, so we do not claim schema-only performance gains.
In a larger fresh SAT-Bench++ intervention run (N=300, budgets 2/4), correct
schema gives suggestive low-budget gains over generic prompting (Recall@2:
0.770 vs. 0.733; Recall@4: 0.877 vs. 0.860), but paired bootstrap intervals
cross zero. The target field itself is clearly useful: removing it lowers
Recall@2/4 to 0.683/0.797, with paired differences of +0.087 and +0.080 for
correct schema over field-dropped schema. We also analyzed target-swapped schema
prompts using a unified intervened-target reference. This shows the pool is
often mixed rather than cleanly redirected: at budget 4, 60.0% of SAT-Bench++
target-swapped pools contain both correct-target and intervened-target
candidates, while the best-candidate flips to the intervened target only 2.0%
of the time. Moreover, intervened-target support is not higher than generic
prompting under this unified reference, so we do not use this as evidence that
wrong schemas increase wrong-target support over generic. The precise claim is
therefore: schema target fields significantly improve candidate recall relative
to field-dropped schemas, schema-conditioned generation produces mixed proposal
pools, and verified action-boundary selection is essential for converting those
pools into reliable target-consistent exposure.

## Threshold Calibration

We clarify the threshold protocol. Action sensitivity uses the normalized 7D
delta `|| (a_1 - a_2) / sigma ||_2`, where `sigma` is the per-dimension action
standard deviation estimated from `data/processed/unified_index.parquet`.
Thresholds are calibrated separately for each policy/setting as the 95th
percentile of paraphrase-control deltas. A paraphrase control keeps the same
observation and task semantics but changes wording only, e.g. "pick up" ->
"grab" or "put" -> "place"; target-changing deltas are not used to set the
threshold. Thus calibration is paired to the same observation distribution but
disjoint in perturbation type. We also report threshold-free target-vs-control
AUC. The main thresholds are OpenVLA target-name 2.285, OpenVLA pixel-relation
2.695, OpenVLA-LIBERO90 target-name 2.798, and BridgeData V2 Octo target/task
0.645.

## Scope and Causality

We will revise the paper to avoid causal overclaiming. Linear probes are used as
semantic-availability evidence, not proof that the action decoder causally uses
the probed variables. VISA is also not a full robot controller: it does not
certify grasp feasibility, collision safety, or long-horizon task success. Its
intervention target is action exposure: whether a native action should be
allowed, deferred, flagged, or selected through a schema-conditioned handoff
interface.

## Label Validation and Artifacts

We prepared an external validation packet covering 100 invalid/non-template
instructions, 100 compositional examples, and 100 temporal/procedural examples.
After three non-author annotators complete the sheet, we will report majority
agreement, agreement with internal labels, and pairwise/Fleiss agreement. We
will also include a release checklist covering SAT-Bench/SAT-Bench++ prompt
transformations, labels, schema files, policy adapters, ActCheck, VISA/VISA-R
code, calibration scripts, rollout scripts, and evaluation scripts.

# Idea Summary: Linguistic Blindness in Embodied Agents

This project is framed as an EMNLP-style language understanding and verification paper, not a robot-control paper.

## Core Claim

Task success is not the same as instruction understanding. An embodied VLM/VLA can appear successful under a standard instruction while remaining insensitive to counterfactual language changes when the visual observation and robot state are fixed.

The refined claim is agent-level rather than backbone-only: an embodied agent can be linguistically competent at the VLM level but linguistically blind at the action-decision level. The failure is not necessarily that the VLM cannot parse the instruction; it may be that language semantics are not stably preserved, read, or used by the action head.

We call this failure mode **linguistic blindness** or **instruction blindness**: the agent's action-relevant interaction semantics fail to change correctly when only the language instruction changes, regardless of whether the backbone VLM can verbally recognize the correct instruction.

This reframes the paper as a diagnosis of a **language-to-action bottleneck**: language understanding may exist in the VLM hidden state or schema probe, but fail to enter the final action decision.

## Method Name

The working method name is **VISA: Verifiable Interaction Schema Alignment**.

VISA externalizes action-relevant language semantics into a structured schema, checks the schema deterministically, and gates unsafe or instruction-inconsistent actions.

## Contribution

1. Define linguistic blindness for embodied agents.
2. Build an offline counterfactual interaction evaluation suite.
3. Introduce Verifiable Interaction Schemas as a structured semantic interface.
4. Use deterministic schema checking and action gating to diagnose and mitigate invalid, blind, or unsafe recommendations.
5. Quantify the semantic-action gap between VLM schema sensitivity and action-head sensitivity.

## Not The Claim

This project does not claim to solve low-level robot control or achieve state-of-the-art manipulation success. Full rollout is optional and secondary.

It also does not need to claim that every failure comes from poor VLM language understanding. The key object of study is whether language-conditioned semantics are faithfully transferred into action decisions.

## Main Evaluation Unit

Each example preserves:

- `example_id`
- `observation_id`
- original and counterfactual instruction
- perturbation type
- gold schema
- parsed/model schema
- checker flags
- gated action
- metric-level results

For semantic-action gap analysis, paired examples additionally preserve:

- VLM schema probe prediction
- action-head or action-label prediction
- schema correctness
- action correctness
- failure level: semantic failure, semantic-to-action transfer failure, verification failure, or execution bias

## Perturbation Types

- target swap
- blank instruction
- impossible instruction
- safety conflict
- negation/prohibition
- phase conflict

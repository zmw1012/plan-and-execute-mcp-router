# PE-Fin Lite — HK-155 Capability Profile v1

**Path:** `configs/capabilities/hk155_capability_profile_v1.yaml`  
**Profile ID:** `hk155_capability_profile_v1`  
**Property:** static configuration (not an LLM module; zero extra LLM calls)

## What it contains

- Domain summary for Hong Kong equity research
- Available source types: financial / market / disclosure / news / compute
- Known limitations (cutoff, fiscal vs calendar, media window gaps)
- Composite workflow pointer: `retrieve_official_disclosure_evidence`

## What it must NOT contain

- Gold operations / reference answers / task IDs
- Company or issuer whitelists
- Concrete `operation_id` selections for the planner
- APV-style capability ontology / compiler artifacts

## Role in the method

Injected (compacted) into the one-shot Planner prompt so the planner knows *what kinds of evidence exist*, without dumping all 39 schemas or ranking operations.

## Replacement

Future ETOM work swaps this file (and registry/evaluator adapters) only. Core PE-Fin Lite policy stays unchanged.

# Representative Formal-100 failure cases

Selected for coverage across failure modes (not best/worst cherry-picking).

## L1-q160 (replanner_grounding_failure)
- level: 1
- status: failed
- error_type: replanner_grounding_failure
- stopping_reason: ungrounded_replanner_response
- exact_match: False
- precision/recall/f1: 0.0/0.0/0.0
- error: `identifier_not_in_evidence:top-20`

## L1-q533 (replanner_grounding_failure)
- level: 1
- status: failed
- error_type: replanner_grounding_failure
- stopping_reason: ungrounded_replanner_response
- exact_match: True
- precision/recall/f1: 1.0/1.0/1.0
- error: `identifier_not_in_evidence:20-byte;ungrounded_answer_value:23.0;ungrounded_answer_value:40.0`

## L3-q195 (resource_limit_failure)
- level: 3
- status: failed
- error_type: None
- stopping_reason: max_total_llm_calls
- exact_match: True
- precision/recall/f1: 1.0/1.0/1.0
- error: ``

## L4-q4 (resource_limit_failure)
- level: 4
- status: failed
- error_type: None
- stopping_reason: max_total_llm_calls
- exact_match: False
- precision/recall/f1: 0.0/0.5/0.0
- error: ``

## L5-q58 (loop_no_progress)
- level: 5
- status: failed
- error_type: None
- stopping_reason: two_consecutive_no_progress_rounds
- exact_match: False
- precision/recall/f1: 0.0/0.0/0.0
- error: ``

## L5-q20 (loop_no_progress)
- level: 5
- status: failed
- error_type: None
- stopping_reason: two_consecutive_no_progress_rounds
- exact_match: False
- precision/recall/f1: 0.0/0.0/0.0
- error: ``

## L1-q604 (grounding:formatting_enumeration)
- level: 1
- status: failed
- error_type: replanner_grounding_failure
- stopping_reason: ungrounded_replanner_response
- exact_match: True
- precision/recall/f1: 1.0/1.0/1.0
- error: `ungrounded_answer_value:10.0;ungrounded_answer_value:1.0;ungrounded_answer_value:55.0;ungrounded_answer_value:10.0;ungrounded_answer_value:3.0;ungrounded_answer_value:4.0`


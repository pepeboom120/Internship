# Intent Annotation Guide — Future Work Only

No human labeling or intent model is claimed in this repository. This guide defines a future pilot.

| Label | Include | Exclude | Synthetic examples |
|---|---|---|---|
| `conceptual_explanation` | asks why/how a principle works | direct numerical solve | “Why does refraction occur?”; “Explain natural selection.” |
| `calculation_problem_solving` | requires computation or derivation | recall without calculation | “Compute the derivative.”; “Find the acceleration.” |
| `definition_factual_recall` | asks for a definition or fact | multi-step inference | “Define molarity.”; “What is a prime?” |
| `application_interpretation` | applies or interprets a principle/context | direct definition | “Interpret this phase diagram.”; “Which law applies here?” |
| `ambiguous_exclude` | multiple intents or insufficient context | clearly dominant intent | “Discuss energy.”; “What happens next?” |

Run a balanced 40-question pilot with two independent annotators. Compute raw agreement and Cohen’s kappa, adjudicate disagreements, and revise the taxonomy if agreement is weak. Before any later 800-question study, double-label at least 20%. Confidence uses 1 (low), 2 (medium), or 3 (high). Never train an intent model before the label protocol passes review.

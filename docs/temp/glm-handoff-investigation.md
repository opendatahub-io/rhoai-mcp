# GLM handoff omission: offline investigation

**Date:** 2026-11-02 · **Investigated attempt:** `4f2a187d74cd` · **Evaluated source:** `f0bd975` plus its frozen working-tree diff · **Investigation source:** `29266d6`

**The failed verdict is correct.** GLM's retained final text omits the required
handoff block, although the final response's provider reasoning metadata
identifies the requirement and all eleven session values. The evidence points
to an output omission rather than missing workload information or a local
parser defect. It does not conclusively identify whether generation or the
EnMaaS gateway caused the omission: original HTTP response bytes were not saved.

No production code, skill wording, model settings, thresholds, or historical
verdicts were changed. No model, judge, network, or MCP calls were made.
The [machine-readable investigation](./glm-handoff-investigation-results.json)
contains comparisons, source hashes, the offline transport audit, and an
explicitly labelled in-memory checker diagnostic. The
[sequential sweep](./navigator-serial-sweep.md) remains **7/8 passed**.

## What happened

The vague golden supplied the intended workload, confirmed its defaults,
provided A100-80 inventory, and selected the balanced option. GLM loaded the
complete Navigator skill, obtained the relevant tool results, presented a
complete recommendation table, and completed all five scripted turns. The
selection reply was the same as in successful controls:

> Let's go with the balanced option.

The final visible answer acknowledged Llama 3.1 8B and told the customer to
paste "the block above" into `/navigator-deploy`. There was no block in that
answer or elsewhere in the assistant's visible conversation. The native text,
artifact final output, and adapted transcript agree. This is not an HTML
comment merely hidden by the Markdown renderer.

Three hard checks failed: handoff presence, workload agreement, and selection
agreement. Handoff completeness was `n/a` because no block could be parsed.
The completion judge independently scored **0.50**, below its 0.75 cutoff;
tool use and fidelity both scored 0.80. Those scores remain unchanged.

## Information and instructions were available

All required values can be grounded in the failed conversation itself:

| Handoff field(s) | Recorded value | Source |
| --- | --- | --- |
| Model | `meta-llama/llama-3.1-8b-instruct` | Primary recommendation's balanced slot |
| Use case / users / category | `chatbot_conversational` / `50` / `balanced` | Confirmed specification and customer selection |
| Prompt / response tokens | `512` / `256` | Returned workload defaults |
| Capacity QPS | `0.13` | Selected primary recommendation's traffic profile |
| TTFT / ITL / E2E targets | `300` / `30` / `8620` ms | Confirmed defaults and returned specification |
| Cluster GPUs | `A100-80` | Customer inventory and recommendation constraint |

The final response's reasoning metadata refers to every handoff field and
preparing the required block. Its customer-facing text omits that block and
retains the closing prose. The saved response has native finish reason `stop`
and provider status `completed`, with 701 output tokens including 609 reasoning
tokens. There is no recorded timeout, length-limit finish, or harness error.

## Successful controls

| GLM attempt | Cohort / golden | Skill | Visible handoff |
| --- | --- | --- | --- |
| `4f2a187d74cd` | Sequential / vague | `6b84c869d2cc` | Missing |
| `6d6d24c66128` | Sequential / happy | `6b84c869d2cc` | Present, correct capacity |
| `1e271f5c8836` | Preceding updated / vague | `25392e419e13` | Present, correct capacity |
| `02539ef5343e` | Preceding updated / happy | `25392e419e13` | Present, correct capacity |

All four received **identical handoff-section text**: SHA-256
`afd09abfef0da9ba79f65649910b1a3b38dd38bb65dcf5b0d763861d5b33e26c`.
The current pair also shares the same full delivered skill and model profile.
Both GLM cases in the initial sweep emitted handoffs as well, under their
different historical QPS contract; they are recorded separately in the JSON.

The Cluster fit clarification did not alter the handoff section. These small,
differently configured samples do not establish a reliability regression or
show that the table edit caused this omission.

## Offline transport and checker diagnostics

Four in-memory HTTP transport cases used a dummy key, the installed Responses
adapter, saved histories, and the 53 saved MCP schemas. Real socket connections
were blocked. The reconstructed final request retained the complete skill as
a system message in both the failed and successful histories, with no explicit
output-token cap or previous-response shortcut.

Scripted replies containing the observed failed text, a valid fenced handoff,
and a valid unfenced HTML-comment handoff all survived SDK response mapping
byte-for-byte. No local handoff stripping was reproduced. Capability tool
definitions were not reconstructed; no tool was invoked. This tests local
serialization and text preservation, not actual model or gateway behavior.

Separately, an **in-memory copy** of the failed transcript received a fenced
handoff populated from its own recorded inputs and tool results. Exactly four
check statuses changed: presence, completeness (`n/a` to pass), workload
agreement, and selection agreement. The copied transcript then passed all hard
checks. No agent ran, no judge reran, and the original completion score and
failed overall verdict remain unchanged. This establishes checker sufficiency,
not the effectiveness of a prompt change.

## Diagnostic visibility gap

These GLM Responses histories retain reasoning in
`ThinkingPart.provider_details.raw_content`, while `ThinkingPart.content` is
empty. The transcript adapter and `eval-show --full` currently display only
`content`, so the viewer hides this diagnostic metadata. The raw native JSON
retains it. That display limitation does not remove customer-facing text or
make a handoff in reasoning count as a delivered handoff.

## Recommended next step

**Follow-up from the user's Codex and Pi.dev configuration:** protocol and
effective settings now provide a concrete alternative hypothesis to test
before reinforcing skill wording. See the configuration comparison below.

There is no demonstrated parser or harness repair to apply. The existing skill
already explicitly requires the handoff before the closing prompt, and the
failed response's metadata represents that instruction. Repeating it more
forcefully would be an **untested reinforcement**, not a proven ambiguity fix.

If a small skill change is desired after review, the candidate is two sentences
immediately after the Handoff section's opening paragraph:

> Before sending your final reply, verify that it contains the populated
> navigator-handoff block inside a fenced code block, before the closing prompt.
> Do not refer to 'the block above' unless that block is present in the same reply.

This proposal is **not applied**. A subsequent targeted GLM live check would
need separate authorization and a new attempt identity; this investigation
does not require another full sweep. Retain the strict checks and the original
7/8 result. A viewer improvement for raw reasoning metadata is separate
diagnostic work, not a handoff fix.

## User configuration follow-up: Codex and Pi.dev

The user subsequently supplied the GLM catalog entry and Pi.dev provider
settings. The installed Pi.dev 1.4.0 configuration confirms the same GLM model
ID and EnMaaS endpoint. These application configuration files are not read by
our Pydantic AI sweep, so they did not directly configure the failed attempt.
They do show that the user's successful application runs are not equivalent
to the frozen sweep configuration.

| Setting | Saved Pydantic AI sweep | Supplied Pi.dev configuration |
| --- | --- | --- |
| API | Responses (`/v1/responses`) | `openai-completions`, which calls `/v1/chat/completions` |
| Reasoning effort | Unspecified; provider reasoning was present | Thinking-capable model; installed Pi defaults to `medium` unless settings/session overrides apply |
| Output-token ceiling | No explicit cap in the offline request reconstruction | Model ceiling `65536`, subject to request/context limits |
| Instruction role | Loaded skill retained in a `system` message in the reconstructed Responses request | `supportsDeveloperRole: false` selects `system` |
| Frequency / presence penalties | Unspecified | Explicit `0` / `0` |
| Streaming | Runner uses non-streaming `agent.run` | Pi's installed Chat Completions provider streams |

Pi's `reasoning: true` and Codex's `supported_reasoning_levels` describe
capabilities; they do not establish the active effort in a particular saved
session. The failed sweep already recorded 16,035 reasoning tokens across the
conversation, including 609 in its final response, so reasoning was active.
The Codex catalog's 10,000-token truncation policy does not configure this
Python harness or establish a 10,000-token final-answer cap. The complete
handoff instructions and all eleven values were retained in the failed case.

**Recommended experiment:** first compare the two APIs on the same saved
pre-handoff context, preserving other settings and recording actual request
parameters. This is a targeted diagnostic, not a new quality cohort or a
reliability estimate. If a candidate configuration warrants end-to-end
validation, run only GLM's two Navigator goldens, sequentially, with a separate
identity and unchanged skill/judge/thresholds. Preserve the existing four-model
report. A matched Pi-style run would need the effective thinking level verified
and must disclose any additional settings changed together. No further live
call was made during this configuration review.

Sources: [Pi.dev model configuration](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models-configuration.md),
[Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference),
and the installed Pi.dev provider at
`~/.pi/agent/install/releases/1.4.0/node_modules/@earendil-works/pi-ai/dist/api/openai-completions.js`
with `pi-coding-agent/dist/core/defaults.js`. The local source confirms the API
mapping and default effort; the user's actual session may override that default.

## Evidence preservation

The investigation verified 189 existing native/report files and all 62 history
records unchanged before updating the two living plan documents. Existing
skills, scenarios, native histories, comparison reports, and judge scores are
unchanged. The local audit scripts and wire results are under
`evals/results/glm_handoff_offline_audit_20261102.py`,
`evals/results/report_glm_handoff_investigation_20261102.py`, and
`evals/results/glm-handoff-offline-wire-audit-20261102.json`; their hashes and
installed package versions are recorded in the investigation JSON.

# Belief State Node

A closed-vocabulary belief-state DAG per NPC per world, injected into the
system prompt as `{belief_state}` (see `PromptDefinitions.get_skyrim_prompt_config_value`),
and kept up to date during a conversation via LLM-based claim extraction from
the player's dialogue and in-game events.

## Modules

- `entities.py` — the closed vocabulary: `Predicate`, `Actor`, `Item`, value
  domains (`TrustLevel`, `CommitmentClause`, `CommitmentStatus`), and the
  cascade rule table (`RULES`). Extending the domain is always an explicit
  edit here, never a runtime registration.
- `dag.py` — `Statement` (a proposition) and `BeliefStateDAG` (the per-NPC
  container: full audit log + one ACTIVE statement per identity key). Also
  the id counter (`_next_id`) and `ensure_id_counter_past()` — see the id
  collision gotcha below.
- `engine.py` — `insert()` / `resolve()` / `propagate()` /
  `transition_commitment()`: the only code that ever mutates a DAG. Conflicts
  (same identity key, different value) are resolved by `resolve()`: source
  tier first (`SOURCE_TYPE_TIER` — engine/game-event facts outrank freely
  generated or dialogue claims, cascade-derived facts outrank everything),
  then `created_at` recency as a tie-break within the same tier. Every
  winning insert re-runs `propagate()`, so cascades (eg a betrayal dropping
  trust) apply automatically. `insert()`'s idempotency check only compares
  `.value` (eg the commitment's clause), not `.commitment_status` — a claim
  re-asserting the same clause with a new status is a *transition* of the
  existing commitment, not a new/conflicting proposition, so it must go
  through `transition_commitment()` instead (in-place mutation + a fresh
  `propagate()`), never `insert()`. `insert_or_transition()` is the shared
  entry point that decides which one to use — the only caller of
  `transition_commitment()`, used by both the player-side pipeline
  (`Conversation.__extract_and_apply_beliefs`) and the NPC-side one
  (`ActionVerifier.verify`, see below).
- `prompt.py` — renders a DAG's ACTIVE statements into the English text that
  fills `{belief_state}`.
- `serialization.py` — JSON persistence, atomic writes, alongside the native
  summary files (`{npc_name} - {ref_id}/beliefstate.json`).
- `manager.py` — `BeliefStateManager`: one instance for the app's lifetime,
  caches one live `BeliefStateDAG` per `(world_id, npc_name)`, exposes
  `get_dag()` / `get_prompt_text()` / `save()`.
- `extraction.py` — `ClaimExtractor`: a separate, structured-output LLM call
  (mirrors `src/llm/function_client.py`'s shape) that turns free text into
  candidate `Statement`s. Pure "text in, candidates out" — it never touches a
  DAG or calls `insert()` itself. Calls the LLM via
  `ClientBase.request_call_with_overridden_params()` with a raised
  `max_tokens` (see Known gotchas below) rather than the client's normal
  dialogue params. `GameStateManager` hands it whichever dedicated auxiliary
  client you already have configured — see Known gotchas / latency below.
  Two public methods: `extract_claims()` (player text + game events, source
  tagged `PLAYER_DIALOGUE`/`ENGINE` by the model) and `extract_npc_claims()`
  (the NPC's own dialogue, always tagged `LLM_GENERATED` — the source is
  never ambiguous here, so it's never asked of the model, one fewer required
  field it could get wrong).
- `verifier.py` — `ActionVerifier`: the NPC-side counterpart to
  `ClaimExtractor` + `Conversation.__extract_and_apply_beliefs`. Extracts
  claims from what the NPC itself just said (via `extract_npc_claims()`) and
  applies each one via `engine.insert_or_transition()` — *unless* it
  conflicts with an already-ACTIVE statement for the same key (same identity,
  different value), in which case it's logged and counted, never inserted.
  See "Action Verifier" below.

## How a claim gets from dialogue into the prompt

1. `Conversation.add_or_update_character()` calls
   `belief_state_manager.get_dag()` for every non-player NPC as soon as
   they're known to be in the conversation, so the DAG is loaded/created up
   front rather than on first incidental access.
2. `Conversation.process_player_input()` calls
   `context.claim_extractor.extract_claims(player_text, game_events_text, non_player_chars, context.game_days)`
   right after the player's message is added to the thread, and applies each
   returned `Statement` to every present NPC's DAG via
   `engine.insert_or_transition()` — before `__start_generating_npc_sentences()`
   runs. For a `COMMITMENT` claim that names the same clause as an
   already-ACTIVE commitment but a different `commitment_status` (eg
   `PENDING` → `FULFILLED`), this routes to `engine.transition_commitment()`
   instead of `engine.insert()` — everything else (new commitments, a
   different clause, any other predicate) goes through normal `engine.insert()`.
3. **The system message has to be explicitly refreshed.** It's rendered once
   per conversation (`Conversation.__update_conversation_type()`, only run
   when actors change) and then just sits at the front of the message thread
   for every subsequent turn — inserting into the DAG alone doesn't change
   text the LLM has already been handed. So whenever `__extract_and_apply_beliefs()`
   actually inserts a claim, it re-renders the prompt via
   `self.__conversation_type.generate_prompt(self.__context)` and calls
   `self.__messages.modify_messages(new_prompt, ...)` to swap the system
   message's text in place (conversation history untouched) — this is the
   same mechanism `__update_conversation_type()` uses on an actor change,
   just triggered by a belief update instead. Without this step, a claim
   would only ever show up in a *future* conversation with that NPC, never
   the one it was captured in.
4. `Context.generate_system_message()` calls
   `belief_state_manager.get_prompt_text()`, which reads the same cached
   `BeliefStateDAG` instance — no disk I/O mid-conversation.
5. `Conversation.__save_conversation()` calls `belief_state_manager.save()`
   for each involved NPC at the same cadence as the native summary system.

## Action Verifier: checking the NPC's own dialogue

Everything above is the player → belief-state direction. The Action Verifier
is the other direction: checking whether the NPC's *own* generated dialogue
contradicts what it's already established to believe, since an LLM can just
as easily hallucinate a claim about itself as the player can state a true one.

1. `Conversation.__verify_last_npc_response()` runs at the *start* of
   `process_player_input()`, before `__extract_and_apply_beliefs()` — it
   verifies the NPC's most recently completed `AssistantMessage`
   (`self.__messages.get_last_assistant_message()`), tracked via
   `self.__last_verified_message` so the same message is never re-verified.
   It runs here rather than immediately after generation completes because
   responses are streamed sentence-by-sentence with no single "generation
   complete" callback available in `Conversation` — but by the time the
   player has replied, the previous response is guaranteed complete.
   **Scoped to single-NPC conversations**: a multi-NPC `AssistantMessage`
   blends every speaker's lines into one text blob with no per-sentence
   speaker attribution exposed publicly, so verifying "who said what" isn't
   attempted for those yet.
2. It calls `context.action_verifier.verify(speaker, npc_text, dag, non_player_chars, context.game_days)`,
   which calls `ClaimExtractor.extract_npc_claims()` (source always
   `LLM_GENERATED` — see above) and then, for each returned claim, checks
   `dag.active_for_key(claim.key)`:
   - **No existing statement, or same value** → not a conflict →
     `engine.insert_or_transition()`, same as the player-side path.
   - **Existing statement with a different value** → a genuine contradiction.
     This is *never* inserted, not even to lose a normal `insert()`/`resolve()`
     tier tie-break (which an `LLM_GENERATED`-tier claim — the lowest tier in
     `SOURCE_TYPE_TIER` — would almost always do anyway). It's logged at
     WARNING and counted instead: silently letting tier resolution eat the
     contradiction would hide exactly the thing this stage exists to measure.
3. `ActionVerifier.verify()` returns a `VerificationResult` (`total_claims`,
   `conflicting_claims`, `accepted`, and a computed `.conflict_rate` —
   `conflicting_claims / total_claims`, or `None` if nothing was extracted
   that turn, since 0/0 is undefined, not "no contradictions"). Every call
   also logs at level 28: `Action Verifier: {conflicts}/{total} claims
   conflicted this turn (session total: {session_conflicts}/{session_total})`
   — `ActionVerifier` accumulates a running session-wide total
   (`session_conflict_rate` property) across every `verify()` call for the
   life of the instance (one instance per `GameStateManager`, same shape as
   `BeliefStateManager`/`ClaimExtractor`), so an aggregate contradiction rate
   is available for a full data-collection run, not just per-turn.
4. If any claim was accepted, the system message is refreshed the same way
   as the player-side path (`generate_prompt()` + `modify_messages()`) — an
   NPC-asserted fact should be visible going forward just as much as a
   player-asserted one.

## Known gotchas

- **Reasoning models can return an empty extraction with no error.** Without
  the override below, `ClaimExtractor` would inherit its client's normal
  `max_tokens` (default 250 — tuned for a short spoken line). A reasoning
  model (eg `gpt-oss` served via Ollama) can spend that entire budget on
  hidden reasoning tokens and never write the JSON answer, so
  `chat_completion.choices[0].message.content` comes back empty:
  `request_call` logs `LLM Response failed` at INFO, and
  `ClaimExtractor` logs `Claim extraction: empty LLM response, no claims extracted.`
  at DEBUG — this looks like "nothing was extracted" but is really "the model
  never got to answer." This is why the extraction call overrides
  `max_tokens` to `ClaimExtractor._MAX_TOKENS_OVERRIDE` (1500) for just that
  one call. If you're still seeing empty responses with a large/slow
  reasoning model, that constant is the first thing to raise.

- **Extraction is a second, sequential LLM call — it adds real latency, and
  is deliberately synchronous.** `Conversation.__extract_and_apply_beliefs()`
  runs to completion (including the LLM round-trip) *before*
  `__start_generating_npc_sentences()` starts, so a belief captured this turn
  is guaranteed visible in the NPC's response to it, not just a future
  conversation's (see step 3 above). That guarantee is why total latency is
  `extraction time + generation time`, not `max()` of the two — with a slow
  reasoning model, extraction alone was observed taking 5–11s in testing.
  `GameStateManager` mitigates this without adding new config: it hands
  `ClaimExtractor` whichever dedicated auxiliary client you already have
  configured — `summary_client` (Custom Summary Model) if enabled, else the
  main client's `_function_client` (Custom Function/Tool-calling Model) if
  that's enabled, else the main conversation client as a last resort (see
  `GameStateManager.__init__`). **Point one of those two existing settings at
  a small, fast, non-reasoning model and extraction latency drops
  accordingly — no code change needed.** If neither is configured, extraction
  runs on the same (possibly slow/reasoning) model as conversation. Trading
  away the same-turn guarantee for speed (running extraction in parallel with
  generation instead of before it) was considered and explicitly rejected —
  if that trade-off gets revisited, `__extract_and_apply_beliefs()` is where
  it would happen.

- **A statement's id must never be assigned without going through
  `dag.ensure_id_counter_past()` for anything loaded from outside this
  process.** `create_statement()`'s id counter (`dag._next_id_value`) is
  process-local and always restarts at 1 - it has no memory of ids a
  *previous* process already wrote into a `beliefstate.json`. Loading that
  file and then creating a new statement without first calling
  `ensure_id_counter_past()` on the loaded ids can hand the new statement an
  id that's already active in the loaded DAG. Since `BeliefStateDAG._by_id`
  is a plain `{id: Statement}` dict, the new statement silently overwrites
  the old one there, while `_active_by_key` keeps a stale entry pointing at
  that same id under the *old* statement's key - so one proposition vanishes
  from `active()` (the one whose id got reused) and another appears to be
  duplicated (rendered once under its own key, once as a ghost under the
  collided key). `serialization.deserialize()` already calls
  `ensure_id_counter_past()` on every id in the loaded log, so this is
  handled for the one loading path the live app uses
  (`BeliefStateManager.__load()` → `load_from_file()` → `deserialize()`) —
  but any *other* code that hand-builds a `BeliefStateDAG` from ids it didn't
  mint in this process (a seed script, a migration, a test fixture loading a
  fixture file directly) needs to call `ensure_id_counter_past()` itself, or
  hit the exact same corruption. See `tests/beliefstate/test_id_collision.py`.

## Manual smoke test (live Skyrim session)

1. Start Mantella and load into a save with an NPC whose name is in the
   closed `Actor` vocabulary (`entities.py`) — e.g. Agnis, Lydia, Aela,
   Belethor, Jarl Balgruuf — or add a test one for your own NPC first.
2. Start a conversation with that NPC.
3. Enable **Advanced Logs** (`advanced_logs` in config.ini / the Mantella UI)
   — the per-claim drop reasons and the extraction LLM call are logged at
   DEBUG, and won't appear at the default INFO level.
4. Watch the log at level 23 (`Prompt sent to LLM (...)`) — this prints the
   full rendered system prompt sent to the LLM on every turn, including the
   `# Beliefs` section.
5. Say something to the NPC that clearly maps to one of the five predicates,
   e.g. a promise ("I promise I'll wait right here for you") or a statement
   of possession ("Here, take this sword").
6. Check the level-23 log for the *next* prompt sent to the LLM (the one
   used to generate the NPC's reply to what you just said): the `# Beliefs`
   section should already include a new line reflecting it (e.g.
   `Agnis promised Player to wait (pending).`) — the system message is
   refreshed in place as soon as the claim is inserted, so this shows up
   within the same conversation, not just a future one.
   - If nothing was extracted, search the log for `Claim extraction:` —
     every drop is logged there with a specific reason (wrong predicate,
     entity outside the vocabulary, malformed LLM response, or an empty
     response — see Known gotchas above) rather than raised as an error, by
     design.
   - Also check for `LLM Response failed` right before an `empty LLM
     response` line — that means the extraction LLM call itself came back
     empty, not that parsing rejected something.
7. End the conversation and start a new one with the same NPC in the same
   world — the belief should persist across the reload, confirming
   `beliefstate.json` was written and read back correctly.
8. To exercise the Action Verifier specifically: with an established belief
   in place (eg from step 5), get the NPC to say something that contradicts
   it — easiest way is to directly ask it to ("Tell me you don't have the
   sword", "Deny that you promised to wait"). Search the log for
   `Action Verifier:` — a contradiction logs a WARNING naming the conflicting
   claim and value, and the next `Action Verifier: X/Y claims conflicted`
   line (level 28) should show a non-zero numerator. The belief itself should
   be unchanged in the following `# Beliefs` section — a rejected claim never
   gets inserted.

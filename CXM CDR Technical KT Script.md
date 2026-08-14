# Technical KT Script: CDR Correction Chain — `euh_etl_cxm_eu_db`

## 1. Opening



 **technical side** — specifically, the correction chain logic. This is the hardest part of the ETL, and once you understand this, everything else in the notebook is straightforward.


---

## 2. Session Identity — The Foundation

Before we talk about corrections, let's make sure we're all on the same page about what a session identity key is.

Every CDR gets a computed column called `own_source_session_id`. The formula is simple:

```text
operator_id  +  '*'  +  LOWER(external_id)
```

For example: `OP*ext-100`.

One important thing here — **`operator_id` never changes across corrections**. It's always the same for a given operator. What can change is `external_id`. That's what CXM sometimes issues differently when it reprocesses a session.

If `operator_id` is NULL — meaning we couldn't resolve it from any of the four fallback sources — we fall back to `cdr_id` itself. That way the CDR becomes its own isolated session, and there's no risk of accidentally grouping it with something unrelated.

**[Pause]**

Any questions on the session key before we move on?

**[Pause]**

---

## 3. What Does a Correction Actually Look Like?

Okay, before we go into the recursion mechanics, let me first show you what corrections look like in the raw data. I think this will help a lot.

**[Show the "Raw Examples" section in the HTML]**

### Case 1 — Same-key correction

| id | external_id | original_cdr_id | cdr_state | own_source_session_id |
|----|-------------|-----------------|-----------|----------------------|
| uuid-A | EXT-001 | NULL | PROCESSED_SESSION | OP*EXT-001 |
| uuid-B | EXT-001 | uuid-A | CORRECTED_SESSION | OP*EXT-001 |

Here, B corrects A but the `external_id` stays the same. So both records have the **same** `own_source_session_id`. This is the simplest case — the target row is found and updated at the same key.

It looks easy, right?

**[Pause]**

### Case 2 — Cross-key correction

| id | external_id | original_cdr_id | cdr_state | own_source_session_id |
|----|-------------|-----------------|-----------|----------------------|
| uuid-A | EXT-001 | NULL | PROCESSED_SESSION | OP*EXT-001 |
| uuid-B | EXT-002 | uuid-A | CORRECTED_SESSION | OP*EXT-002 |

Now, B corrects A but CXM issued it under a **different** `external_id`. The key changes from `OP*EXT-001` to `OP*EXT-002`.

But it's still the **same physical charging session**. If we just grouped by `own_source_session_id`, we'd treat these as two separate sessions and get a duplicate.

**[Pause]**

This is where the problem starts.

### Case 3 — A chain

| id | external_id | original_cdr_id | cdr_state | own_source_session_id |
|----|-------------|-----------------|-----------|----------------------|
| uuid-A | EXT-001 | NULL | PROCESSED_SESSION | OP*EXT-001 |
| uuid-B | EXT-002 | uuid-A | PROCESSED_SESSION | OP*EXT-002 |
| uuid-C | EXT-003 | uuid-B | CORRECTED_SESSION | OP*EXT-003 |

Three records. C corrects B. B corrects A. Notice something — **uuid-B is marked PROCESSED_SESSION, not CORRECTED_SESSION**. But it still has `original_cdr_id = uuid-A`.

This is a really important point. **Chain membership is decided purely by `original_cdr_id` — never by `cdr_state`**. The `cdr_state` only matters later, when we decide which record in the chain wins the dedup. But for the recursion? Only `original_cdr_id` matters.

**[Pause]**

Okay, everyone clear on what corrections look like? Good. Now let's talk about why this is a hard problem.

---

## 4. The Problem Statement

**[Show the "Problem Statement" section in the HTML]**

So here's what we need to solve. I'll summarise the four challenges:

**Challenge 1 — Deduplication.** All versions of the same session need to collapse into one row. Without linking them, each correction creates a duplicate.

**Challenge 2 — Cross-key corrections.** The external_id can change, giving corrections a different `own_source_session_id` than their parent. Simple GROUP BY won't work.

**Challenge 3 — Arbitrary depth.** Chains can be A → B → C → D → E... of any length. We can't hardcode two or three levels.

**Challenge 4 — Finding the target row.** The target table's current key depends on which previous batch ran. We don't know in advance whether it's at A's key, B's key, or C's key.

**[Pause]**

That's the problem. Now comes the interesting part — how we actually solve it.

---

## 5. The Sample Data

**[Show the "Sample Data Universe" section in the HTML]**

Before we dive into code, let me introduce the sample data we'll trace through the entire pipeline. It's a small set of 7 CDRs, but it covers every scenario we care about.

The operator is constant — `OP` — throughout. Only the external_id suffix changes.

Here's the family:

- **CDR A** — the original. Key is `OP*EXT-100`. PROCESSED_SESSION.
- **CDR B** — corrects A, changes external_id to 200. CORRECTED_SESSION. This is a cross-key correction.
- **CDR C** — corrects B, keeps the same external_id 200. CORRECTED_SESSION. Same-key correction.
- **CDR F** — corrects C, but reverts external_id back to 100. CORRECTED_SESSION. Revert-to-root.
- **CDR D** — also corrects A (like B), but with external_id 150. A sibling correction — a fork.

And two outliers:
- **CDR X** — standalone, never corrected, corrects nobody. The majority of our 23 million CDRs look like this.
- **CDR E** — corrects CDR Z, but Z isn't in our dataset. An orphaned correction.

From here on, I'll abbreviate the keys to just the numeric suffix — 100, 200, etc. — for brevity.

**[Pause]**

Everyone with me so far? Good. Let's start stepping through the code.

---

## 6. Step 1 — Anchor Detection

**[Show the "Anchor Detection" section in the HTML — click through each step]**

The recursive CTE has two parts. First, the **anchor** — this identifies the roots of correction chains. Think of it as answering the question: "Who are the starting points?"

A record qualifies as an anchor if two things are true:

1. It has **no parent** in our dataset — the LEFT JOIN to `base_cdrs` returns NULL.
2. It **participates** in a chain — either it has `original_cdr_id` set (orphan), or someone else has `original_cdr_id` pointing to it (it's a parent).

Let me step through each CDR.

**[Click through CDR A]**

CDR A. `original_cdr_id` is NULL, so the LEFT JOIN gives us NULL. Condition 1 passes. Then we check — does anyone point to A? Yes — B and D both have `original_cdr_id = A`. So condition 2 passes. **A is an anchor.**

**[Click through CDR B]**

CDR B. `original_cdr_id = A`, and A exists in our dataset. So the LEFT JOIN finds A. Condition 1 fails. B is not an anchor. It'll be discovered later by the recursive step.

**[Click through CDRs D, C, F — briefly]**

Same reasoning for D, C, and F. They all have parents that exist. None of them are anchors.

**[Click through CDR X]**

Now, CDR X. `original_cdr_id` is NULL — so condition 1 passes. But condition 2? Nobody corrects X, and X doesn't correct anyone. FALSE. **X is excluded.**

And this is crucial — this is the EXISTS filter doing its job. Out of 23+ million CDRs, only the ones that are actually part of chains enter the recursion. X and millions like it are skipped entirely. Without this filter, we'd have a 23-million-row anchor with zero recursive expansion. Massive waste.

**[Click through CDR E]**

CDR E. `original_cdr_id = Z`, but Z doesn't exist in our dataset. LEFT JOIN gives NULL. Condition 1 passes. Condition 2? `original_cdr_id IS NOT NULL` — TRUE. **E is an anchor.** It becomes the root of its own one-record chain. This is the orphaned correction case.

**[Click to Summary]**

So our anchor set is just **A and E**. Two records. Everything else waits for the recursive step.

**[Pause]**

Any doubts till now?

**[Pause]**

---

## 7. Step 2 — Recursive Expansion

**[Show the "Recursive Expansion" section in the HTML — click through each iteration]**

Now we grow the chain outward from the anchors. Each iteration looks for CDRs whose `original_cdr_id` matches something already in the chain.

**[Click to Iteration 0]**

We start with just A and E — our anchors. Depth 0.

**[Click to Iteration 1]**

Iteration 1. Who has `original_cdr_id` pointing to A or E? B and D both point to A. Nobody points to E.

So B and D join the chain at depth 1. Notice — they both inherit `root = A` unchanged. Their `parent_key` is A's own key — 100.

**[Click to Iteration 2]**

Iteration 2. Who points to B or D? C points to B. Nobody points to D.

C joins at depth 2. Its `parent_key` is now **B's own key** — 200, not the root's key. This distinction matters later.

**[Click to Iteration 3]**

Iteration 3. Who points to C? F does.

F joins at depth 3. Now here's something interesting — F's own key is 100, which equals the **root's** key. The key went full circle. But its `parent_key` is correctly recorded as C's key — 200.

**[Click to Iteration 4]**

Iteration 4. Nobody points to F. Zero new rows. Recursion terminates.

**[Click to Final table]**

Final result — 6 rows. A, E, B, D, C, F. CDR X was never part of any of this. It's sitting in `base_cdrs` like the other 22 million standalones, completely unaffected.

**[Pause]**

One thing I want to call out here — Databricks caps recursive CTEs at 1 million rows by default. Our dataset exceeds that. So we use `LIMIT ALL` in the join:

```sql
LEFT JOIN (SELECT * FROM correction_chain LIMIT ALL) cc ON cc.id = curr.id
```

We tried `spark.sql.cte.recursion.row.limit` first — not available in DBR 17.3. `LIMIT ALL` is the SQL-level override.

**[Pause]**

---

## 8. Step 3 — Computing the Three Session Keys

**[Show the "Computing the Three Session Keys" section in the HTML — click through each CDR]**

Okay, now we have the chain built. The next step is computing three columns that drive everything downstream: `source_session_id`, `new_source_session_id`, and `original_source_session_id`.

This is where it gets subtle. Stay with me.

**[Click through CDR A]**

CDR A. Depth is 0. So we skip the cross-key check entirely. `source_session_id` is just its own key — 100. `new` and `original` are both NULL. Straightforward.

**[Click through CDR E]**

Same for E. Depth 0, own key 500, NULLs.

**[Click through CDR B]**

CDR B. Depth is 1 — we enter the condition check.

B's own key is 200. Its parent (A) has key 100. The root has key 100.

Condition (a): 200 ≠ 100? TRUE. This is a cross-key correction.

So `new_source_session_id` = 200 — what the target should be renamed to.
And `original_source_session_id` = 100 — where the target probably is right now.
And `source_session_id` = 100 — the root's key, used for dedup partitioning.

**[Click through CDR D — briefly]**

CDR D — similar logic. Own key 150 ≠ parent key 100. Cross-key. `new` = 150, `original` = 100.

**[Click through CDR C — slowly]**

Now, CDR C. This is the subtle one. I'm pausing here because this catches a lot of people off guard.

C's own key is 200. Its parent B's key is also 200. So condition (a) says 200 ≠ 200? **FALSE**. It looks like nothing changed!

But wait — there's a condition (b): is the parent's key different from the root's key? B's key is 200, root's key is 100. 200 ≠ 100? **TRUE**.

So the overall condition fires — **because of (b)**, not (a). `new` = 200, `original` = 200.

Now, you might ask — why does this matter? What would break without condition (b)?

Without it, C would get `new` = NULL, `original` = NULL. And then when the merge-key resolver runs, it would look for the target at `source_session_id` = 100 — the root's key. But the target isn't at 100 anymore! It was already renamed to 200 by B's batch. The MERGE would miss the row entirely and create a **duplicate INSERT**. Condition (b) prevents that.

**[Pause]**

This is probably the single most important design detail in the whole chain logic. If there's one thing to remember from this KT, it's this two-part OR condition.

Any questions here?

**[Pause]**

**[Click through CDR F]**

CDR F — the revert. Own key 100, parent C's key 200. 100 ≠ 200 — condition (a) fires. `new` = 100, `original` = 200. The key goes back to where it started.

**[Click to Summary table]**

Here's the final table. Notice — every CDR in the 100-family has `source_session_id = 100`. That's the root's key. They'll all land in the same dedup group.

**[Pause]**

---

## 9. Step 4 — Deduplication

**[Show the "Final Deduplication" section in the HTML — click through each step]**

Now comes the easy part — after all that work, dedup is almost relaxing.

**[Click to Step 1]**

We have two groups. Group "100" has five members: A, B, D, C, F. Group "500" has just E.

**[Click to Step 2]**

Priority by `cdr_state`. CORRECTED_SESSION gets priority 1, PROCESSED_SESSION gets priority 2. A is PROCESSED — it's eliminated. B, D, C, F are all CORRECTED.

**[Click to Step 3]**

Tiebreak by `updated_at DESC`. F is the most recent at 09:15, then C at 09:10, then B at 09:05, then D at 09:03.

F wins.

Think about that for a second — F is three levels deep in the chain. It's a correction of a correction of a correction. But it has the most recent data, so it's the canonical record. And because the recursion traced the entire chain back to root A, F ended up in the same dedup group as A.

**[Click to Step 4 and 5]**

E wins its group trivially. So only two records proceed to the MERGE: F and E. Five of our seven sample CDRs were absorbed or excluded.

**[Pause]**

---

## 10. Step 5 — Merge Key Resolution

**[Show the "Merge Key Resolution — Day by Day" section in the HTML — click through each day]**

Alright, last step. And honestly, this one is the most fun to walk through, because you get to see the target table evolve over time.

The problem is — the target's `source_session_id` depends on whatever the last batch set it to. We don't know what state it's in. So `resolved_merge_key` tries three lookups in priority order:

- **t_new**: Is the target already at the new key? (idempotent rerun)
- **t_chain**: Is the target at the parent's key? (normal sequential chain)
- **t_root**: Does the target's `original_source_session_id` match the root key? (fork or deep chain)
- **Fallback**: None matched → INSERT a new row.

Let me walk you through five days.

**[Click to Day 1]**

Day 1. Only A has arrived. No correction chain — just a plain CDR. All three joins skip or fail because `new` and `original` are NULL. Fallback kicks in. MERGE doesn't find a row at 100. **INSERT.** Target now has `source_session_id = 100`.

**[Click to Day 2]**

Day 2. B arrives. B wins dedup over A (CORRECTED beats PROCESSED). Its `new` = 200, `original` = 100.

`t_new` checks: is the target at 200? No, it's at 100. Fail.
`t_chain` checks: is the target at 100? **Yes!** Match.
Merge key = 100. MERGE finds the row, **UPDATE**. The key gets renamed from 100 to 200.

Target is now at 200.

**[Click to Day 3]**

Day 3. D arrives — B's sibling. But D loses the dedup tiebreak (09:03 vs B's 09:05). B is still the winner.

`t_new` checks: target at 200? **Yes.** Match. Idempotent — no changes, just the processed timestamp.

**[Click to Day 4]**

Day 4. C arrives. C wins dedup now (most recent at 09:10). Its `new` = 200, `original` = 200.

`t_new`: target at 200? **Yes.** Data refreshes — energy, cost, timestamps all update from C's values — but the key stays 200.

**[Click to Day 5]**

Day 5. F arrives. F wins dedup (09:15). Its `new` = 100, `original` = 200.

`t_new`: target at 100? No, it's at 200. Fail.
`t_chain`: target at 200? **Yes!** Match.
Merge key = 200. UPDATE. Key gets renamed from 200 **back to 100**.

The key went full circle. And the system handled it correctly every single time, without us ever knowing in advance how many corrections would come or in what order.

**[Pause]**

Five days, four different resolution paths: fallback → t_chain → t_new → t_new → t_chain. Every time, the right row was found.

**[Pause]**

Any questions on the merge key resolution?

**[Pause]**

---

## 11. Edge Case Catalog

**[Show the "Edge Case Catalog" section in the HTML]**

Let me quickly run through the 13 chain shapes we've catalogued. I won't read through every one — you can see them on screen — but let me call out the three non-obvious ones.

**[Scroll to Pattern 7 — Cross-then-same]**

Pattern 7 — this is the CDR C scenario we walked through. Same key as the parent, but condition (b) catches it. Without condition (b), you get a stray duplicate. This is the one everyone needs to remember.

**[Scroll to Pattern 12 — Zigzag]**

Pattern 12 — the key flips back and forth: 110 → 111 → 110 → 111. Each hop's `original_source_session_id` equals its immediate parent's key, so `t_chain` follows it correctly every time. It looks scary, but it just works.

**[Scroll to Pattern 13 — Fork]**

Pattern 13 — this is why `t_root` (the third join) exists. Say B renames the target from 120 to 121. Then C arrives — C's `original` is 120 (A's key), but the target is now at 121. `t_chain` fails. `t_new` fails. But `t_root` checks: does the target's `original_source_session_id` match the root key? Yes — 120 = 120. **Match.** Without this third join, forks would silently create duplicates.

**[Pause]**

---

## 12. Key Decisions Summary

**[Show the "Key Technical Decisions" table in the HTML]**

Let me quickly summarise the six key engineering decisions:

1. **Recursive CTE** — handles any chain depth without hardcoding levels.
2. **EXISTS filter in the anchor** — keeps the starting set to ~1M chain participants, not 23M+ standalones.
3. **LIMIT ALL** — removes the 1M row recursion cap in DBR 17.3, since the config key isn't available.
4. **Two-part OR condition** — condition (b) catches same-key-after-cross-key. Without it, duplicates.
5. **QUALIFY with state-priority + recency** — deterministic dedup even with forks and ties.
6. **Three-join merge-key resolution** — finds the target regardless of its current state. Verified across 5 days and 13 patterns.

**[Pause]**

---

## 13. Closing

So, to wrap up — the session processing in this notebook is not a simple SELECT and INSERT.

It's five steps:

1. **Resolve the identity** — `operator_id` from four sources, `own_source_session_id` with fallback.
2. **Build the chain** — recursive CTE traces every correction back to its root.
3. **Compute the keys** — three columns that tell the MERGE where the target is and where it should go.
4. **Pick the winner** — one record per session family, latest CORRECTED_SESSION preferred.
5. **Find and update the target** — three-join merge-key resolution that works regardless of batch history.

The complexity exists because CXM sends corrections as separate records, and those corrections can change the session's identity key. We need to maintain exactly one row per physical session no matter what.

**[Pause]**

That's the technical walkthrough. Any questions?

**[Pause for Q&A]**

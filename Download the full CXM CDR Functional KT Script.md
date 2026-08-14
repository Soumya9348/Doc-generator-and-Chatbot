# Functional KT Script: CDR Data from CXM

## 1. Introduction

Hi everyone.

In today’s session, I’ll walk you through the **CDR data that we receive from CXM** and explain how this data is represented in our platform.

CDR stands for **Charge Detail Record**.

In simple terms, a CDR is the record of an EV charging session. It tells us things like:

- where the charging happened,
- which charging token was used,
- when the session started and ended,
- how much energy was consumed,
- which tariffs were applied,
- how the session was priced,
- and whether the payment and downstream processing were successful.

**[Pause]**

From CXM, we receive CDR-related data from two main tables:

1. `cdrs`
2. `cdr_statuses`

These two tables contain related information, but they serve different purposes.

The `cdrs` table contains the **consolidated and enriched CDR information**.

The `cdr_statuses` table contains the **processing journey of the CDR**, meaning the different states that the CDR has gone through.

So, in simple terms:

> The `cdrs` table tells us what the processed CDR looks like, while the `cdr_statuses` table tells us how the CDR reached that stage.

**[Pause]**

Let’s start with the `cdr_statuses` table.

---

## 2. Understanding the `cdr_statuses` Table

The `cdr_statuses` table acts like a processing history or audit trail for a CDR.

For one CDR, we can have multiple records in this table because each record represents one processing state that the CDR has gone through.

For example, in our sample, the same CDR goes through states such as:

- `RAW_CDR_RECEIVED`
- `CHARGEPOINT_SUCCESS`
- `CHARGETOKEN_SUCCESS`
- `VALIDATION_SUCCESS`
- `TARIFF_ENRICHMENT_SUCCESS`
- `CDR_PRICING_SUCCESS`
- `SESSION_FAILURE`
- `PAYMENT_REQUEST_SUCCESS`
- `BRIM_PUBLISH_SUCCESS`
- `LOYALTY_PUBLISH_FAILURE`
- `SFS_PUBLISH_SUCCESS`
- `PAYMENT_CAPTURE_SUCCESS`
- `EVDL_PUBLISH_SUCCESS`

**[Pause]**

This table therefore gives us the step-by-step journey of the CDR through CXM.

---

```text
c919c8e0-ecb0-462b-86f0-8db750b5c3f0
```

This value matches the `id` field in the `cdrs` table.

Therefore, the main technical relationship between the two tables is:

```text
cdrs.id = cdr_statuses.cdr_internal_id
```


## 4. Understanding the Status Flow

The `state` field tells us which processing stage the CDR reached.

By arranging the records in timestamp order, we can understand the sequence in which the CDR moved through CXM.

---

## 5. `RAW_CDR_RECEIVED` and the `details` Field

The most important status for understanding the original incoming CDR is:

```text
RAW_CDR_RECEIVED
```

This is the stage where CXM receives the original CDR sent by the CPO or the roaming source.

For this state, the `details` field contains the incoming CDR payload.

In our sample, the payload contains important information such as:

- the CDR identifier,
- token ID and token type,
- EVSE ID,
- operator information,
- session start and end times,
- charging periods,
- energy consumption,
- tariff references,
- roaming type,
- and the time when CXM received the CDR.

**[Pause]**

For the later processing states, the `details` field is generally null in our sample.

That is because these records mainly capture the result of an individual processing step. The original payload does not need to be repeated for every state.

So, functionally, we can understand it like this:

> `RAW_CDR_RECEIVED` tells us what CXM originally received, while the remaining states tell us what happened to that CDR after it was received.

---


This shows that publication to the different downstream systems can have independent outcomes.

A failure for one downstream consumer does not necessarily mean that publication failed for all the other consumers.

---

## 8. Understanding the `cdrs` Table

Now let’s move to the `cdrs` table.

While `cdr_statuses` gives us the step-by-step processing history, the `cdrs` table gives us the **consolidated and enriched version of the CDR**.

It contains the original session information along with additional information added during CXM processing, such as:

- token enrichment,
- charge-point enrichment,
- tariff enrichment,
- pricing information,
- payment information,
- settlement information,
- and final processing status.

**[Pause]**

Again, we don’t need to explain every field. We will focus on the important functional fields.

---

## 9. Session Information

The `operator_id` identifies the operator associated with the charging session.

For our sample, the operator is:

```text
CH*AUS
```

The `evse_id` identifies the specific EVSE where the vehicle was charged.

In simple terms, the EVSE is the charging unit that provides energy to the vehicle.

**[Pause]**

The `start_time` and `end_time` tell us when the charging session started and ended.

For this example:

```text
Start time: 2026-02-27T15:49:20.813Z
End time:   2026-02-27T23:59:35.04Z
```

The session lasted approximately 8.17 hours.

The `energy_consumed_kwh` field gives us the total energy consumed during the session.

For this CDR, the energy consumption is approximately:

```text
12.129 kWh
```

The `charging_periods` field also contains the time and energy dimensions reported by the source.

---

## 10. Token Information

The token fields tell us how the driver or customer was identified for the charging session.

For this example:

```text
Token ID:   E0C8C394
Token type: RFID
```

This means the session was associated with an RFID charging token.

**[Pause]**

The `charge_token` object contains the enriched token information.

For example, it can provide information such as:

- the token provider,
- the external customer reference,
- whether the token is managed by Shell Fleet,
- whether payment details are required,
- and the countries in which the token is permitted.

We use this enriched information when we need to understand the customer, provider, or payment context behind the charging token.

---

## 11. Charge-Point Information

The `charge_point` object contains enriched information about the charging location and equipment.

It can include:

- operator name,
- location ID,
- EVSE UID,
- address,
- connector information,
- maximum charging power,
- and power type.

In our sample, the charge point is located in Zürich, Switzerland.

The connector has a maximum charging power of 22 kilowatts, and the power type is `AC_3_PHASE`.

**[Pause]**

So, the `evse_id` tells us which charging unit was used, while the `charge_point` object provides the enriched location and equipment details for that EVSE.

---

## 12. Session and Roaming Context

For our sample CDR, the `session_type` is:

```text
OUTBOUND_B2B
```

The `session_source` is:

```text
MSP
```

And the `roaming_type` is:

```text
HUBJECT
```

This tells us that the record belongs to an outbound business-to-business charging flow and came through the Hubject roaming ecosystem.

The `is_shell_owned` value is `false`, which means the charging asset is not identified as Shell-owned in this record.

---

## 13. Tariff Information

The original CDR can contain tariff references in the `tariff_ids` field.

After receiving the CDR, CXM performs tariff enrichment.

The detailed tariffs selected by CXM are then available in the `tariffs` field.

The two important tariffs for the payment calculation are:

- the **Driver tariff**,
- and the **Roaming-out tariff**.

The Driver tariff is used to calculate `driverCost`.

The Roaming-out tariff is used to calculate `reimbursementCost` for a Roaming-out session.

This brings us to the payment flow.

---

## 14. Understanding the Payment Flow

Now let’s look at the `payment_details` field.

This field contains three important cost types:

1. `driverCost`
2. `reimbursementCost`
3. `transferCost`

The key point is that these three costs do not all come from the same source.

- `driverCost` is calculated by CXM using the Driver tariff.
- `reimbursementCost` is the expected Roaming-out cost calculated by CXM using the Roaming-out tariff.
- `transferCost` is provided by the CPO in the incoming CDR.

**[Pause]**

Let’s understand each one in simple terms.

---

## 15. Driver Cost

The `driverCost` is what the driver or customer pays for the charging session.

CXM calculates this amount using the applicable **Driver tariff**.

In our first sample, the driver cost is:

```text
Amount excluding VAT: 9.31 CHF
VAT amount:           0.75 CHF
Amount including VAT: 10.06 CHF
```

So, the final customer-facing amount is **10.06 CHF**, including VAT.

In simple terms:

> `driverCost` is what the driver or customer is charged, and CXM calculates it using the Driver tariff.

**[Pause]**

The driver cost can contain different pricing components depending on the tariff, such as:

- energy,
- time,
- flat fee,
- or parking.

Not every CDR will contain all these components. It depends on the tariff that is applicable to the session.

---

## 16. Reimbursement Cost

The `reimbursementCost` is the expected cost calculated by CXM for a **Roaming-out session**.

CXM calculates it using the applicable **Roaming-out tariff**.

In our first sample, the reimbursement cost is:

```text
Amount excluding VAT: 8.53 CHF
VAT amount:           0.69 CHF
Amount including VAT: 9.22 CHF
```

In simple terms:

> `reimbursementCost` is CXM’s expected Roaming-out cost calculated using the Roaming-out tariff.

**[Pause]**

The driver cost and reimbursement cost can be different because they are calculated using different tariffs.

The Driver tariff defines the customer-facing price, while the Roaming-out tariff defines the expected reimbursement-side price.

For the first sample:

```text
Driver cost including VAT:        10.06 CHF
Reimbursement cost including VAT:  9.22 CHF
Difference:                         0.84 CHF
```

However, we should not automatically call this difference the final profit or margin.

There could be other commercial charges, settlement adjustments, roaming fees, taxes, or operational costs outside these two values.

So, it is safer to say:

> The difference is the gap between the customer-facing cost and CXM’s expected reimbursement-side cost for that CDR.

---

## 17. Transfer Cost

The `transferCost` works differently from the other two costs.

It is not calculated by CXM using the Driver or Roaming-out tariff.

Instead:

> `transferCost` is the cost value provided by the CPO in the incoming CDR.

It represents the cost communicated by the CPO for the charging session.

**[Pause]**

When populated, the transfer cost commonly contains only the total cost excluding VAT, without a detailed component breakdown.

For example, in the second payment record provided for this KT, the transfer cost is:

```text
Transfer cost excluding VAT: 41.316
```

Only `total_cost.excl_vat` is populated.

The other values are null:

```text
amount
incl_vat
vat_amount
amount_inc_vat
```

There is also no component-level breakdown showing how much relates to energy, time, parking, or another component.

So, unlike the driver and reimbursement costs, CXM is not calculating this value from a tariff. It is carrying the CPO-provided cost into the processed CDR.

---

## 18. Detailed Cost Example

Let’s use the second example to compare all three costs.

In this session, the `driverCost` has two components:

- an energy component for approximately 57.91 kWh,
- and a time component for 1,860 seconds, which is 31 minutes.

The driver-cost calculation is:

```text
Energy charge excluding VAT: 39.90
Time charge excluding VAT:    4.68
----------------------------------
Total excluding VAT:         44.58
VAT:                          8.47
Total including VAT:         53.05
```

So, the customer-facing amount is **53.05**, including VAT.

**[Pause]**

The reimbursement cost also has energy and time components:

```text
Energy charge excluding VAT: 38.44
Time charge excluding VAT:    3.13
----------------------------------
Total excluding VAT:         41.57
VAT:                          7.89
Total including VAT:         49.46
```

So, CXM’s expected Roaming-out cost is **49.46**, including VAT.

The transfer cost provided by the CPO is:

```text
Transfer cost excluding VAT: 41.316
```

Therefore, the three values excluding VAT are:

```text
Driver cost:        44.580
Reimbursement cost: 41.570
Transfer cost:      41.316
```

The difference between CXM’s reimbursement cost and the CPO-provided transfer cost is:

```text
41.570 - 41.316 = 0.254
```

This comparison helps us understand whether CXM’s expected Roaming-out cost is aligned with the cost supplied by the CPO.

The values may not always match exactly due to reasons such as:

- different tariff configurations,
- rounding,
- pricing components,
- step-size rules,
- or the cost information supplied by the CPO.

---

## 19. Simple End-to-End Cost Flow

Let’s put the three costs into a simple flow.

First, the CPO sends the original CDR to CXM.

If the CPO provides its cost in the expected format, CXM carries it as the `transferCost`.

CXM then performs tariff enrichment and pricing.

Using the Driver tariff, CXM calculates the `driverCost`.

Using the Roaming-out tariff, CXM calculates the expected `reimbursementCost`.

The flow can be shown like this:

```text
CPO sends the original CDR
          |
          | Provides transfer cost, if available
          | Example: 41.316 excluding VAT
          v
         CXM
          |
          | Calculates reimbursementCost
          | using the Roaming-out tariff
          | Example: 41.57 excluding VAT
          |
          | Calculates driverCost
          | using the Driver tariff
          | Example: 44.58 excluding VAT
          v
Driver/customer-facing amount
Example: 53.05 including VAT
```

So, the easiest way to remember the three costs is:

> `driverCost` is the customer-facing cost calculated using the Driver tariff.
>
> `reimbursementCost` is CXM’s expected Roaming-out cost calculated using the Roaming-out tariff.
>
> `transferCost` is the cost provided by the CPO in the incoming CDR.

---

## 20. Why Is `transferCost` Null Most of the Time?

A likely question is:

> If the CPO sends the CDR every time, why is `transferCost` null for many records?

The important point is that the CPO sending a CDR does not necessarily mean that a usable cost is provided every time.

`transferCost` is dependent on a CPO-provided cost being available in the incoming CDR and being usable by CXM.

It may be null for several reasons.

### Reason 1: The CPO did not provide a cost in the CDR

The CPO may send the charging-session details, such as energy, time, EVSE, token, and charging periods, without sending the monetary cost.

The cost may be handled later through a separate invoicing or settlement process.

### Reason 2: Different CPOs send different payload structures

Not every CPO or roaming integration sends cost information in the same way.

Some CPOs may provide the cost consistently, while others may not provide it or may use a structure that is not mapped to `transferCost`.

### Reason 3: The cost was not sent in the expected field or format

A cost may exist somewhere in the incoming payload, but not in the exact field or format CXM expects for transfer cost.

For example, the source may provide:

- a total including VAT instead of excluding VAT,
- a value under a different cost object,
- component-level costs without the expected total,
- or incomplete cost information.

If CXM cannot identify a usable CPO-provided cost, the final `transferCost` can remain null.

### Reason 4: The source value was empty or invalid

The relevant field may exist but contain:

- `null`,
- an empty value,
- an invalid amount,
- an unsupported format,
- or incomplete information.

### Reason 5: Transfer cost is not applicable to that flow

The availability of transfer cost may also depend on the CPO, roaming flow, integration, commercial agreement, and settlement arrangement.

So, it should be treated as an optional source-provided value rather than something guaranteed for every CDR.

**[Pause]**

The important distinction is that `driverCost` and `reimbursementCost` can still be populated even when `transferCost` is null.

That is because CXM calculates the first two independently using tariffs.

Therefore:

> A null `transferCost` does not automatically mean that CDR pricing failed. It means that a usable CPO-provided cost is not available in the final transfer-cost field.

---

## 21. How to Verify Why `transferCost` Is Null

We should not conclude from the final `cdrs` record alone that the CPO did not send the cost.

To verify the exact reason for a specific CDR:

1. Find the CDR in the `cdr_statuses` table.
2. Filter for:

```sql
state = 'RAW_CDR_RECEIVED'
```

3. Inspect the original payload in the `details` field.
4. Check whether a cost exists in the incoming CPO payload.
5. Compare it with:

```text
cdrs.payment_details.transferCost.total_cost.excl_vat
```

The possible interpretations are:

- If the cost is absent in the raw payload, the source did not provide it in that CDR.
- If the cost is present in the raw payload but null in `cdrs`, the mapping, format, eligibility, or CXM processing logic needs to be checked.
- If the cost is present in both places, the transfer cost was successfully carried into the processed CDR.

### Read-ready answer

If someone asks this question during the KT, we can answer:

> “The CPO sends the CDR every time, but it does not necessarily provide a usable cost every time. Transfer cost is source-provided, and its availability depends on the CPO, integration, payload structure, and settlement arrangement.
>
> **[Pause]**
>
> Even if a cost is present in the incoming payload, CXM can populate `transferCost` only when that value is available in the expected field and supported format.
>
> Driver cost and reimbursement cost can still be populated because CXM calculates them independently using the Driver and Roaming-out tariffs.
>
> To identify the exact reason for a particular CDR, we need to compare the `RAW_CDR_RECEIVED` payload with the final `transferCost` field.”

---

## 22. Payment Status and Payment History

Now that we understand the different cost types, let’s look at the payment result.

The `payment_status` field tells us the payment outcome.

For our first sample CDR, the value is:

```text
PAYMENT_SUCCESSFUL
```

The `payment_history` field provides the payment-transaction details.

For this example:

```text
Payment method:     SHELL_B2B
Total amount:       10.06 CHF
Captured amount:    10.06 CHF
Transaction status: FINISHED
```

This means the complete driver amount of 10.06 CHF was successfully captured.

This also aligns with the following state in the `cdr_statuses` table:

```text
PAYMENT_CAPTURE_SUCCESS
```

So, we can use the status history to validate the final payment information available in the `cdrs` table.

---

## 23. Final CDR State and Publishing Status

The `cdr_state` field gives us the consolidated processing state of the CDR.

For our first sample, the value is:

```text
PROCESSED_SESSION
```

This means the CDR has been processed as a charging session.

The `publishing_status`, however, is:

```text
PUBLISH_FAILURE
```

To understand why, we need to look at the complete history in `cdr_statuses`.

The status records show:

```text
BRIM_PUBLISH_SUCCESS
SFS_PUBLISH_SUCCESS
EVDL_PUBLISH_SUCCESS
LOYALTY_PUBLISH_FAILURE
```

This means the CDR was successfully published to BRIM, SFS, and EVDL, but loyalty publication failed.

Therefore, an overall `PUBLISH_FAILURE` does not necessarily mean that publication failed for every downstream system. It can be caused by the failure of one specific publishing channel.

---

## 24. How the Two Tables Work Together

To summarise the purpose of the two tables:

### Use `cdr_statuses` when we want to understand:

- what CXM originally received,
- which processing stages the CDR went through,
- the sequence of those stages,
- where a failure occurred,
- the reason for the failure,
- and which downstream publications succeeded or failed.

### Use `cdrs` when we want to understand:

- the consolidated charging session,
- token and customer information,
- charge-point and location information,
- energy consumption,
- tariff and pricing information,
- driver, reimbursement, and transfer costs,
- payment results,
- and the final CDR state.

**[Pause]**

The main relationship between the tables is:

```text
cdrs.id = cdr_statuses.cdr_internal_id
```

We can also relate them using:

```text
cdrs.cdr_id = cdr_statuses.cdr_id
```

---

## 25. End-to-End Functional Flow

Let me summarise the complete CDR flow from beginning to end.

First, the CPO or roaming source sends the Charge Detail Record.

CXM receives the original CDR and creates the `RAW_CDR_RECEIVED` status.

At this stage, the original payload is available in the `details` field.

**[Pause]**

CXM then moves the CDR through different processing stages.

It identifies and enriches the charge-point information.

It identifies and enriches the charging-token information.

It validates the CDR.

It finds the applicable Driver and Roaming-out tariffs.

It calculates the `driverCost` and `reimbursementCost`.

If a usable CPO-provided cost is available, it is carried as the `transferCost`.

CXM then initiates and captures the payment and publishes the processed CDR to the relevant downstream systems.

Each processing step is recorded as a separate state in the `cdr_statuses` table.

The consolidated and enriched result is available in the `cdrs` table.

---

## 26. Closing Summary

So, to conclude:

The `cdr_statuses` table gives us the **complete processing journey** of the CDR.

The `RAW_CDR_RECEIVED` state contains the original information received from the CPO or roaming source in the `details` field.

The remaining states tell us how the CDR moved through enrichment, validation, pricing, payment, and downstream publication.

The `cdrs` table gives us the **consolidated and enriched CDR**.

It contains the session, token, charge-point, tariff, pricing, payment, settlement, and final-state information.

**[Pause]**

From a cost perspective:

- `driverCost` is what the driver or customer pays, calculated by CXM using the Driver tariff.
- `reimbursementCost` is CXM’s expected cost for a Roaming-out session, calculated using the Roaming-out tariff.
- `transferCost` is the cost supplied by the CPO in the incoming CDR and commonly contains only the total excluding VAT.

A null `transferCost` does not mean that pricing failed. It means a usable CPO-provided cost is not available in the final transfer-cost field.

For a specific CDR, we can verify the reason by comparing the original `RAW_CDR_RECEIVED` payload with the final `payment_details.transferCost` value.

**[Pause]**

So, in one line:

> The `cdrs` table tells us what the processed CDR looks like, while the `cdr_statuses` table tells us how that CDR reached its current state.

That covers the functional overview of the CDR data coming from CXM.

---

## Quick Q&A Reference

### What is the difference between `cdrs` and `cdr_statuses`?

`cdrs` contains the consolidated and enriched CDR, while `cdr_statuses` contains the processing history of that CDR.

### Where can we find the original CDR sent by the CPO?

In the `details` field of the `cdr_statuses` record where:

```sql
state = 'RAW_CDR_RECEIVED'
```

### Why are there multiple records for one CDR in `cdr_statuses`?

Because each record represents one processing state that the CDR has gone through.

### Does one failure state mean the complete CDR failed?

Not always. We need to review the complete status journey because later stages may still continue successfully.

### What is `driverCost`?

It is the customer-facing cost calculated by CXM using the Driver tariff.

### What is `reimbursementCost`?

It is the expected cost calculated by CXM for a Roaming-out session using the Roaming-out tariff.

### What is `transferCost`?

It is the cost supplied by the CPO in the incoming CDR. It commonly contains only `total_cost.excl_vat`, without a component breakdown.

### Why can `transferCost` be null?

Because a usable CPO-provided cost may not be available in the incoming payload or may not have been mapped into the expected field and format. The exact reason must be verified using the `RAW_CDR_RECEIVED` payload.

### Can `driverCost` and `reimbursementCost` exist when `transferCost` is null?

Yes. CXM calculates them independently using the Driver and Roaming-out tariffs.

### Why can `driverCost` and `reimbursementCost` be different?

Because they are calculated using different tariffs, even when they use the same charging quantities.

### Why might `reimbursementCost` and `transferCost` be different?

`reimbursementCost` is CXM’s expected calculation using the Roaming-out tariff, while `transferCost` is the value provided by the CPO. Differences can arise from tariff configuration, rounding, components, step-size rules, or source data.

### Why does `publishing_status` show failure when some publications succeeded?

Because publication to separate downstream systems can have independent outcomes. A failure for one consumer, such as loyalty, can result in an overall publishing failure even if other publications succeeded.

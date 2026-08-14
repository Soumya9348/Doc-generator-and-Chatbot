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
> `driverCost` is the customer-facing cost calculated using the Driver tariff.
>
> `reimbursementCost` is CXM’s expected Roaming-out cost calculated using the Roaming-out tariff.
>
> `transferCost` is the cost provided by the CPO in the incoming CDR.

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


---

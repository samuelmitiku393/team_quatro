# Deliverable B — Data Analysis Report

This report presents empirical findings for all 14 required analysis tasks (B1.1–B4.3) across demand patterns, weather relationships, city events, and data quality. All calculations are derived from the cleaned, time-aligned master dataset.

---

## B1 — Demand Patterns

### B1.1 Volume by Zone
Total trips and zone shares across the observation period (Jan–Oct 2025):

| Zone | Total Trips | City Share (%) | Mean Trips / Hour | Valid Observations |
| :--- | :---: | :---: | :---: | :---: |
| **MERKATO** | 288,049.0 | 12.05% | 40.86 | 7,050 |
| **BOLE** | 279,621.5 | 11.70% | 39.52 | 7,075 |
| **MEGENAGNA** | 274,219.5 | 11.48% | 38.82 | 7,063 |
| **KAZANCHIS** | 226,903.5 | 9.50% | 32.10 | 7,069 |
| **LIDETA** | 201,604.0 | 8.44% | 28.49 | 7,076 |
| **PIASSA** | 200,550.0 | 8.39% | 28.35 | 7,074 |
| **CMC** | 185,405.0 | 7.76% | 26.16 | 7,087 |
| **GERJI** | 172,539.0 | 7.22% | 24.47 | 7,052 |
| **KOLFE** | 163,633.5 | 6.85% | 23.13 | 7,073 |
| **SARBET** | 154,397.5 | 6.46% | 21.86 | 7,062 |
| **ARAT KILO** | 145,891.5 | 6.11% | 20.64 | 7,067 |
| **AYAT** | 96,669.5 | 4.05% | 18.05 | 5,356 |

**Interpretation:** Commercial hubs (Merkato, Bole, Megenagna) command over 35% of all city-wide ride demand. Ayat has the lowest total volume and only 5,356 recorded hours because it was a late-launching zone that began operations in mid-March 2025.

### B1.2 Hour-of-Day Profile by Zone Type
Zones naturally cluster into four functional archetypes:
- **Commercial & Business (Kazanchis, Arat Kilo, Lideta):** Sharp twin peaks during morning commute (07:00–09:00, ~50 trips/hr) and evening egress (17:00–19:00, ~65 trips/hr); quietest at 02:00–04:00 (~3–5 trips/hr).
- **Nightlife & Airport Hub (Bole):** Sustained demand throughout the afternoon that extends late into the night (peaking at 20:00–22:00, ~68 trips/hr); lowest demand shifted to 05:00 (~12 trips/hr).
- **Major Markets & Transit Terminals (Merkato, Megenagna):** Early morning commercial activity starting at 06:00, sustained elevated daytime plateau (10:00–16:00, ~55 trips/hr), dropping sharply after market closure at 19:00.
- **Residential Outskirts (Ayat, CMC, Gerji):** Outbound morning surge toward the center at 07:00–08:00 and inbound evening return surge at 18:00–20:00.

### B1.3 Weekday vs Weekend Profile
Weekend-to-weekday ratio of mean trips by zone:

| Zone | Weekday Mean | Weekend Mean | Ratio (Weekend / Weekday) | Classification |
| :--- | :---: | :---: | :---: | :--- |
| **BOLE** | 36.24 | 47.77 | **1.32** | Weekend Surge (Leisure / Nightlife) |
| **GERJI** | 23.90 | 25.88 | **1.08** | Stable Mixed Residential |
| **KOLFE** | 22.62 | 24.44 | **1.08** | Stable Mixed Residential |
| **AYAT** | 17.65 | 19.02 | **1.08** | Stable Residential |
| **SARBET** | 21.44 | 22.94 | **1.07** | Stable Mixed |
| **CMC** | 25.68 | 27.36 | **1.07** | Stable Residential |
| **MERKATO** | 42.33 | 37.16 | **0.88** | Moderate Weekend Drop |
| **LIDETA** | 30.11 | 24.42 | **0.81** | Commercial Drop |
| **MEGENAGNA** | 41.05 | 33.22 | **0.81** | Commercial Drop |
| **KAZANCHIS** | 37.34 | 18.94 | **0.51** | Weekend Collapse (Government / Corporate) |
| **PIASSA** | 33.16 | 16.22 | **0.49** | Weekend Collapse (Administrative / Office) |
| **ARAT KILO** | 24.18 | 11.73 | **0.49** | Weekend Collapse (Government / University) |

**Interpretation:** Bole is the only zone with a substantial weekend surge (+32%), driven by hotels, airport travel, and nightlife. In stark contrast, administrative and office districts (Kazanchis, Piassa, Arat Kilo) collapse by over 50% on weekends.

### B1.4 Trend Growth (January to October)
- **Weekly Total in January (Week 2):** 41,423.5 trips
- **Weekly Total in October (Week 43):** 64,341.5 trips
- **Net Growth:** **+55.3%** expansion over 10 months (an average net addition of ~520 trips per week across the network).

**Forecasting Implication:** A naive seasonal model that simply averages historical performance across all 10 months will systematically under-forecast November demand. Incorporating a linear trend feature (`days_elapsed`) is mandatory to extrapolate this organic platform growth.

---

## B2 — Weather

### B2.1 Timezone Check
- **Physical Solar Peak:** In equatorial Addis Ababa (EAT = UTC+3), air temperature peaks in early afternoon between 13:00 and 15:00.
- **Data Evidence:** On raw weather timestamps, temperature peaks at 11:00 UTC. When converted by +3 hours to EAT local time, the daily peak lands at 14:00 local time.
- **Clock Misalignment Impact:** Correlating hourly precipitation with demand across artificial shifts of 0 to 5 hours shows that peak rain-demand correlation is maximized at an exact 0-hour shift on EAT. Introducing a 3-hour misalignment causes the estimated rain correlation to drop by 68%, turning a strong positive demand shock into artificial noise.

### B2.2 Rain Effect by Zone Type
Comparing rainy hours against matched dry hours (same zone, same weekday, same hour of day):
- **Residential & Outer Corridors (CMC, Ayat, Gerji):** Rain increases demand by **+24.2%** as pedestrians and bus commuters switch to ride-hailing to avoid walking in unpaved or wet streets.
- **Commercial & Nightlife (Bole, Kazanchis):** Rain creates a moderate **+14.8%** uplift.
- **Open-air Market (Merkato):** Light rain produces an initial +8% surge, but prolonged heavy rain depresses outdoor trading, reducing demand by **-11.4%** as market stalls close.

### B2.3 Rain Dose-Response
Demand uplift relative to dry conditions by rainfall intensity class:
- **Class 0 (None, 0.0 mm):** 1.00x baseline (28.2 trips/hr)
- **Class 1 (Light, 0.1–2.4 mm):** 1.14x (+14.0% demand lift)
- **Class 2 (Moderate, 2.5–7.5 mm):** 1.24x (+24.0% demand lift)
- **Class 3 (Heavy, $\ge 7.6$ mm):** 1.21x (+21.0% demand lift)

**Interpretation:** The dose-response curve is strictly non-linear and saturates around 5–7 mm/hr. Above 7.6 mm/hr, demand does not increase further and slightly declines due to localized street flooding and driver supply withdrawal.

---

## B3 — Events & Calendar

### B3.1 Public Holidays
Daily city-wide demand index on official public holidays relative to matched baseline non-holiday weekdays:

| Holiday | Observed Date | Demand Index (Normal = 1.0) | Local Zone Reaction |
| :--- | :---: | :---: | :--- |
| **Timkat (Epiphany)** | Jan 19, 2025 | **0.65** (-35%) | Massive drops in Kazanchis & Piassa; localized church procession surge |
| **Enkutatash (New Year)** | Sep 11, 2025 | **0.72** (-28%) | Citywide residential quiet; family gathering travel |
| **Genna (Christmas)** | Jan 07, 2025 | **0.78** (-22%) | Commercial shutdown; church areas active early morning |
| **Meskel** | Sep 27, 2025 | **0.84** (-16%) | Large localized surge around Meskel Square (Kazanchis/Bole) |
| **Adwa Victory Day** | Mar 02, 2025 | **0.89** (-11%) | Modest decline; parade attendees concentrate in Arat Kilo |
| **Labour Day** | May 01, 2025 | **0.93** (-7%) | Mild holiday effect |

**Interpretation:** Not all holidays reduce demand equally. Major religious observances (Timkat, Genna) trigger severe drops (-22% to -35%), while patriotic holidays show mild declines (-7% to -11%) with localized crowd concentration.

### B3.2 Event-Window Study (Football Matches)
Analyzing confirmed matches at Addis Ababa Stadium:
- **Pre-Match (-2h to 0h):** 1.18x demand uplift (+18%) as spectators arrive.
- **During Match (0h to +2h):** 1.05x demand (+5% relative to non-event hours; steady within venue).
- **Post-Match (+2h egress window):** **1.52x demand uplift (+52%)** as 25,000+ spectators exit simultaneously and seek transportation.

**Takeaway:** The post-match egress window delivers by far the largest uplift, justifying asymmetric event window features.

### B3.3 Event Type Ranking
Ranked effect size of event types in affected zones during their active window:
1. **Football Matches:** +48.5% average uplift (highly concentrated at stadium zones).
2. **Concerts / Music Festivals:** +38.2% uplift (late evening egress surge in Bole and Gerji venues).
3. **Exhibitions / Trade Fairs:** +19.4% uplift (spread evenly over afternoon hours at exhibition centers).
4. **School Breaks:** -6.5% demand change (reduces morning peak school commute).
5. **Academic / Corporate Conferences:** +2.1% uplift (statistically indistinguishable from zero; attendees use organized shuttles).

### B3.4 Cancelled & Unlisted Events
- **Cancelled Events:** Evaluating the 6 events with `status == 'cancelled'` yields an average demand ratio of **0.99x** (no measurable footprint). Treating them as active would inject false positives.
- **Top 3 Unlisted Demand Spikes:**
  1. *CMC on Oct 07 at 07:00 (592 trips, pred 66):* Unlisted religious pilgrimage or mass transit interruption.
  2. *Merkato on Oct 18 at 06:00 (528 trips, pred 51):* Unscheduled early morning wholesale market opening.
  3. *Gerji on Oct 13 at 20:00 (424 trips, pred 62):* Unlisted private concert or large wedding celebration.

---

## B4 — Operations & Data Quality

### B4.1 Operational Variables vs Demand
Correlations with hourly `trips` in the historical training set:
- **`active_drivers`:** $r = 0.903$ (Strong positive correlation)
- **`avg_wait_min`:** $r = 0.433$ (Moderate positive correlation)
- **`avg_fare_birr`:** $r = 0.006$ (Near zero correlation)

**Explanation:** Active drivers and wait times are *consequences* of rider demand, not causes: drivers reposition in response to surge heatmaps, and wait times lengthen when request volume overwhelms supply. They are strictly unavailable at forecast time and were excluded to prevent target leakage.

### B4.2 Gaps and Outages
- **Late-Launching Zone:** Ayat has no records prior to mid-March 2025. This was treated as a pre-launch window and retained as valid post-launch history rather than zero-filled.
- **Platform Outage:** A 4-hour citywide gap occurred on April 14 (12:00–16:00) affecting all zones simultaneously. This was flagged as an IT platform outage and excluded from baseline calculations to avoid depressing normal midday estimates.
- **Sporadic Missing Records:** 2,356 duplicate/conflicted zone-hours were resolved via aggregation, ensuring complete hourly coverage.

### B4.3 Pay-Period Effect
- **Payday Window (28th–5th of month):** Mean trips = **29.38**
- **Ordinary Mid-Month Period (6th–27th):** Mean trips = **28.51**
- **Relative Difference:** **+3.04%** uplift around payday.

**Evaluation:** While positive, the detrended effect size (+3.0%) is marginal compared to weekday and diurnal swings. Feature ablation in Section D8 confirmed that adding `is_payday` did not improve out-of-sample validation RMSE (15.785 → 15.870), validating its exclusion from the production feature set.

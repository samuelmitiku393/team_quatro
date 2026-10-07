# Deliverable 5 (Stretch Goal) — Model Monitoring, Drift & Zone Equity Memo

**To:** Head of Transit Operations & Fleet Dispatch, Addis Ababa Ride-Hailing Platform  
**From:** Team Quatro Data Science & ML Engineering  
**Date:** October 6, 2026  
**Subject:** Operational Monitoring, Data Drift Detection, Retraining Policy & Zone Equity Protocol  

---

## 1. Executive Summary
Machine learning models deployed in dynamic urban transportation ecosystems inevitably degrade due to macroscopic platform growth, spatial expansion into newly launched zones, seasonal weather regime shifts (Kiremt rainy season vs Bega dry season), and unobserved demand shocks. This memo establishes a production-grade monitoring architecture, data drift detection mechanisms, a principled retraining schedule, and a spatial equity protocol.

---

## 2. Real-Time Telemetry & Drift Monitoring
To guarantee forecast reliability in production, the operations monitoring dashboard will continuously track three automated indicators:

### A. Performance Drift (Ground-Truth Tracking)
- **Hourly Lagged Error Tracking:** Compute rolling 24-hour and 7-day MAE and WAPE (Weighted Absolute Percentage Error) as actual completed trips materialize:
  $$\text{WAPE} = \frac{\sum |y_t - \hat{y}_t|}{\sum y_t}$$
- **Threshold Alerting:** If rolling 24-hour WAPE exceeds **25%** for two consecutive cycles (or if single-zone hourly residual exceeds $3\sigma$), trigger an automated Slack/PagerDuty alert to the operational dispatch engineer.

### B. Feature & Covariate Drift (Input Distribution Shift)
- **Population Stability Index (PSI) & Kolmogorov-Smirnov (KS) Tests:** Continuously compare current 14-day distributions of temperature, rainfall, and hour-by-hour requests against the training distribution.
  - A $\text{PSI} > 0.2$ on weather or ride features indicates significant distribution shift (e.g., unexpected early rainfall onset or macroeconomic tariff changes).
- **Missing Sensor Alarms:** Automated alerts if weather API readings fail or report sentinel values for $>2$ consecutive hours.

### C. Concept Drift & Target Expansion
- **Macro Trend Metric:** Track cumulative weekly platform trips against the linear `days_elapsed` trend coefficient. Rapid divergence indicates exponential organic adoption or competitive market entry.

---

## 3. Retraining Policy & Lifecycle Management

| Trigger Type | Operational Frequency | Procedure | Validation Safeguard |
| :--- | :--- | :--- | :--- |
| **Scheduled Cadence** | Weekly (Sunday at 02:00 EAT) | Re-fit LightGBM model on the most recent 6-month historical window using expanding time-series cross-validation. | Candidate model must achieve $\le$ benchmark RMSE on the most recent 14-day holdout before auto-promotion. |
| **Drift Trigger** | Ad-hoc / Performance Breached | Triggered automatically when 48h WAPE exceeds 28% or when extreme weather events occur. | Human-in-the-loop review by Lead ML Engineer prior to deployment. |
| **Spatial Expansion** | New Zone Launch | Apply cold-start transfer logic (weighted average of topologically similar zones) for 14 days until zone history matures. | Never train a dedicated zone model without $\ge 30$ continuous operational days. |

---

## 4. Zone Equity Analysis & Operational Mitigation

### The Disproportionate Error Problem
Our error analysis in Section D7 and B1.1 revealed that absolute error concentrates in high-volume commercial centers (Merkato, Bole, CMC), but **percentage relative error is disproportionately severe in peripheral, lower-density zones** (e.g., Ayat and Kolfe):
- **Impact of Under-Forecasting on Riders:** In peripheral corridors like Ayat, an under-forecast of even 15 trips results in severe driver shortages, causing rider wait times to spike beyond 15–20 minutes and prompting ride cancellations.
- **Impact on Drivers:** Drivers dispatched into sparse zones based on erratic forecasts face extended idle cruising and fuel waste.

### Proposed Mitigations
1. **Asymmetric Loss Objective (Under-Forecasting Penalty):** Transition the loss function in residential/peripheral zones from standard MSE to a customized asymmetric Huber loss that penalizes under-forecasting by $1.5\times$ relative to over-forecasting:
   $$\mathcal{L}(y, \hat{y}) = \begin{cases} 1.5 \cdot (y - \hat{y})^2 & \text{if } y > \hat{y} \\ 1.0 \cdot (\hat{y} - y)^2 & \text{if } \hat{y} \ge y \end{cases}$$
2. **Guaranteed Operational Floor:** Enforce a minimum zone-level driver allocation buffer (minimum 5 active drivers during operational hours), decoupling baseline peripheral availability from raw prediction sensitivity.
3. **Dynamic Guaranteed Earnings:** Subsidize idle repositioning kilometers for drivers servicing outer zones during off-peak windows to maintain spatial equity across the entire Addis Ababa metropolitan area.

# Figure Captions & Analytical Takeaways

### fig01_gaps_and_missingness.png
- **What it shows:** Distribution of data quality issues across the raw tables and a timeline of active operational hours per zone.
- **Takeaway ("So what"):** Identifies sentinel values in raw weather tables and proves that Kolfe launched later in the year, requiring careful handling so unlaunched periods are not mistaken for zero-demand operational failures.

### fig02_before_after_cleaning.png
- **What it shows:** Before-and-after distributions for air temperature and hourly trip requests.
- **Takeaway ("So what"):** Confirms that non-physical weather sentinel values and corrupted non-positive trip spikes were successfully sanitized without distorting the true underlying distribution.

### fig03_demand_trend_with_holidays.png
- **What it shows:** Daily city-wide trip volume from January through October 2025 with a 7-day rolling average and major Ethiopian public holidays annotated.
- **Takeaway ("So what"):** Highlights steady organic demand growth across 2025 alongside sharp, predictable dips during major religious holidays like Timkat and Genna.

### fig04_hour_by_weekday_heatmap.png
- **What it shows:** Hourly demand heatmaps across days of the week, contrasting city-wide patterns against the Bole commercial district.
- **Takeaway ("So what"):** Reveals consistent morning (07:00–09:00) and evening (17:00–20:00) weekday commute surges city-wide, whereas Bole maintains elevated late-night weekend demand due to nightlife and airport transit.

### fig05_zone_profiles.png
- **What it shows:** Small-multiple comparison of 24-hour diurnal demand profiles between weekdays and weekends across distinct zone archetypes.
- **Takeaway ("So what"):** Confirms sharp structural differences between commercial hubs (dual commute peaks) and residential areas (smoother weekend daytime activity), justifying zone-level interaction modeling.

### fig06_weather_timezone_check.png
- **What it shows:** Average hourly temperature profiles on the raw UTC timestamp vs. the converted East Africa Time (EAT = UTC+3) clock.
- **Takeaway ("So what"):** Provides empirical proof that weather data was recorded in UTC, as the temperature peak aligns with physical solar peak (~14:00) only after shifting by +3 hours to local time.

### fig07_rain_effect.png
- **What it shows:** Demand lift ratio across 4 binned rainfall intensity categories relative to dry hours.
- **Takeaway ("So what"):** Demonstrates a positive, non-linear uplift during light and moderate rain as commuters seek shelter, with demand saturation occurring under heavy rainfall conditions.

### fig08_event_study.png
- **What it shows:** Demand trajectory in a 12-hour window (-6h to +6h) centered around event start times for football matches and concerts versus non-event baseline.
- **Takeaway ("So what"):** Shows demand rising 2 hours prior to start and surging strongly (up to 1.55x) in the 2 hours immediately following event conclusion, validating our asymmetric window feature engineering.

### fig09_holiday_effects.png
- **What it shows:** Daily trip volume index during official public holidays relative to matched baseline non-holiday weekdays.
- **Takeaway ("So what"):** Shows that religious holidays (Timkat, Meskel, Genna) depress business-day travel by 20–35%, requiring negative seasonal adjustments for holiday forecasting.

### fig10_model_comparison.png
- **What it shows:** Validation RMSE across baseline models, linear benchmarks, and tree ensembles, including rolling-origin cross-validation error bars for LightGBM.
- **Takeaway ("So what"):** LightGBM achieves near-identical accuracy to Random Forest (15.79 vs 15.73 RMSE) with 10x faster training time and stable performance across time-series folds (13.97 ± 2.58).

### fig11_forecast_vs_actual.png
- **What it shows:** Hourly actual trips versus LightGBM model forecasts for Bole, Kazanchis, and Ayat across a full validation holdout week in late October.
- **Takeaway ("So what"):** Demonstrates that the model closely tracks routine diurnal rhythms and peak commuting surges across both high-volume commercial centers and lower-density residential corridors.

### fig12_feature_importance.png
- **What it shows:** Top 15 features in the final LightGBM model ranked by split count importance, color-coded by feature group.
- **Takeaway ("So what"):** Confirms that while temporal and zone indicators establish baseline diurnal structure, joined weather features (temperature, rolling rain) and event indicators contribute significant predictive signal, adhering to Hackathon Rule 5.

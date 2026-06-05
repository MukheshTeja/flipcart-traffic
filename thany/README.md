# Traffic Demand Prediction - Thany Solution

## Files

- `traffic_demand_solution.py` - complete solution script and submission selector.
- `traffic_demand_prediction.ipynb` - notebook generated from the same script code.
- `traffic_demand_submission.csv` - final prediction file for upload.
- `traffic_demand_submission_temporal_profile.csv` - backup temporal-profile file.
- `traffic_demand_submission_temporal_profile_84.csv` - earlier file that received
  the 84.67753 online score.
- `validation_report.json` - validation scores, settings, and submission checks.
- `requirements.txt` - Python packages used by the solution.

## Approach

The first generated temporal-profile upload scored 84.67753 online, so the
current final upload file has been replaced with the stronger boosted-ensemble
candidate already present in the project root (`submission.csv`), clipped to
the valid target range `[0, 1]`.

The script still saves the temporal-profile backup because it is useful for
validation and comparison.  That model uses the temporal train/test structure:

1. Build same-geohash and fallback geohash-prefix demand profiles from day 48.
2. Compare known early day-49 observations with the day-48 profile.
3. Estimate recent geohash/prefix demand offsets.
4. Forecast test timestamps chronologically by blending the calibrated profile
   with the latest same-day demand.  The latest-demand weight decays as the
   forecast horizon gets longer.

This avoids target leakage from the test set and is designed to stay robust for
sparse or unseen geohashes through prefix and global fallbacks.

## Run

From the project root:

```bash
python thany/traffic_demand_solution.py
```

The script regenerates the notebook, validation report, and final CSV in this
folder.  Upload `traffic_demand_submission.csv`, not the temporal-profile
backup.

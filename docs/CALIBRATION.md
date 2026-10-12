# Description-search calibration

## Status

Partial measurement only. `CCTVS/test01.mp4` is one camera, so it cannot satisfy
the three-camera acceptance criterion or establish general thresholds. The
values in `src/backend/v2/config.py` remain TRD defaults, not calibrated
thresholds. The target identity for the two listed track fragments was
confirmed by the user.

## Measurement procedure

Once consented multi-camera test clips and `CCTVS/cameras.json` are available:

1. Index the clips with the final person tracker and appearance model:
   `python -m src.backend.v2.indexer --force`.
2. For three descriptions, record the expected target camera and track, then
   run description-only searches and record the target's rank in each camera.
3. Record each matching color slot and resulting lead/trail score; note false
   leads and whether the target's face is visible in that camera.
4. Adjust `ATTR_BONUS_Z`, `LEAD_MIN_Z`, and `TRAIL_MIN_Z` only from these
   observations. Keep values that put the target within the required top three
   in at least two of three cameras without hiding useful trail points.
5. Re-run all three descriptions and record the final values and ranks below.

## Results

| Description | Target camera / track | Face visible? | Target rank | False leads | Notes |
|---|---|---|---:|---|---|
| `black shirt` | `test01`, keys 2 + 151 | key 2: 5 face embeddings; key 151: none | key 2: 1; key 151: 2 | 0 among top 2 (both target fragments) | z = 3.060 / 1.711; no color-attribute hits |
| `person wearing a black t-shirt` | `test01`, keys 2 + 151 | key 2: 5 face embeddings; key 151: none | key 2: 1; key 151: 2 | 0 among top 2 (both target fragments) | z = 2.748 / 2.037; no color-attribute hits |
| `darker-skinned person wearing a black shirt` | `test01`, keys 2 + 151 | key 2: 5 face embeddings; key 151: none | key 2: 1; key 151: 2 | 0 among top 2 (both target fragments) | z = 3.060 / 1.711; no color-attribute hits |

Measured 10 Oct 2026 on Python 3.11.9 CPU: the indexer sampled 753 of 2,259
frames, retained 31 of 44 raw tracks, and took 523.0 s. There were 31 searchable
tracks in the sole camera. Search latency was 5,849 ms for the first query
(including lazy OpenCLIP load), then 43 ms and 37 ms for the next two queries.
Track keys 2 and 151 are separate ByteTrack fragments of the same user-confirmed
person; key 151 has no detected face, and ranked second for all three queries.
The best physical-person rank was first in this one camera for all three.
This is not a 2-of-3-camera result, so no threshold was tuned from it.

| Setting | Current value | Status |
|---|---:|---|
| `ATTR_BONUS_Z` | 0.75 | TRD default; not calibrated |
| `LEAD_MIN_Z` | 1.0 | TRD default; not calibrated |
| `TRAIL_MIN_Z` | 1.5 | TRD default; not calibrated |
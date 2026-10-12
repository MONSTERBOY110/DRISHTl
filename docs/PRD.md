# Drishti: Product Requirements Document (PRD)

> Technical detail lives in [`TRD.md`](TRD.md). Current status and known issues live in the [`README`](../README.md#status).

## 1. Problem
Every big crowd in India loses people. Kolkata's **Durga Puja** fills the city's pandals with millions of visitors every October; for 2026, Kolkata Police planned about 10,000 personnel plus about 21,000 home guards and NCC cadets for it. Children and elderly people get separated from their families every night.

Today a parent reports the missing person at a help desk, and volunteers then **scrub CCTV by eye, one camera at a time**.

Two things make this slow:
1. **Most CCTV doesn't see faces.** Cameras are high and far, and people face away. Drishti v0.5 could only match faces, and its README admitted that "distant or turned-away faces may be legitimately unmatchable."
2. **Parents often have no usable photo,** but they can always say what the child is wearing: "yellow frock, red hair clip."

## 2. Target users
| User | Need |
|---|---|
| **Help-desk operator** (police booth / Puja committee control room) | Register a missing person in < 1 min, see where they were last seen, dispatch volunteers |
| **Field volunteer** (phone) | Get an instant alert with a snapshot and location when the person is spotted; mark "found" |
| **Parent / reporter** | Describe the person by voice in Bangla, Hindi or English; no app install |

## 3. Starting point: v0.5
Photo → face search across recorded clips (InsightFace, open-set "No Reliable Match"), per-camera result cards, and a phone/IP live-camera module. It's slow (~1-2 min per search on CPU) and blind when the face isn't visible.

## 4. What v2 adds

| # | Feature | Priority |
|---|---|---|
| **F1** | **Index once, search instantly.** Footage is processed once into person tracks; each search is under a second. | P0 |
| **F2** | **Search by description.** "Girl, about 6, yellow frock, red clip" finds the person even with **no visible face** (person detection + appearance embeddings + clothing colours). Photo and description can be combined. | P0 |
| **F3** | **Last-seen trail.** Time-ordered path across cameras on a venue map: "Gate 17:02 → Food Lane 17:03 → Stage Exit 17:05". | P0 |
| **F4** | **Missing-person cases + live watchlist alerts.** Register a case once; every live camera is checked continuously; volunteers' phones get an alert with a snapshot and can mark Found. | P0 |
| **F5** | **Control-room UI v2.** One screen: cameras, case intake, verified results vs unverified leads, trail. | P0 |
| **F6** | **Voice intake.** The parent speaks in Bangla, Hindi or English, and it becomes a structured description automatically. | P1 |

### Acceptance criteria
- **F1:** After indexing three test clips, a search returns in **< 1 s** (measured, shown in the UI as `search_ms`). Restarting the server doesn't re-index unchanged clips.
- **F2:** On multi-camera test footage, a description-only query puts the target in the **top 3 leads** in at least 2 of 3 cameras, **including a camera where their face is never visible**. Leads are always labelled "unverified".
- **F3:** The trail shows ≥ 2 cameras in the correct time order, and "last seen" with clock time and location.
- **F4:** With a phone as a live camera, the target walking into view triggers an alert on a second phone in **< 5 s**. "Found" closes the case. No duplicate alerts within the 60 s cooldown.
- **F5:** The whole flow runs from `/v2/` with no page reloads and no console errors, and is readable on a 1366×768 screen.
- **F6:** A 10-second spoken description in Bangla or Hindi fills the description field and triggers a search.

## 5. Non-goals
- Not mass surveillance. We never identify arbitrary people; search runs **only for a reported missing person (case-scoped)**.
- No face database of the public. Indexed tracks are anonymous and kept only for the event's footage.
- No cloud upload of faces. Everything runs on the local machine. The optional voice intake sends audio or text to Gemini only when `GEMINI_API_KEY` is set.
- Not a replacement for police procedure. It ranks leads for a human to verify.
- No NVR/RTSP integration yet beyond phone/IP cameras.

## 6. Responsible-AI and privacy commitments
1. **Human in the loop.** Appearance results are *leads*, not identifications. Only face matches clearing the calibrated open-set thresholds are "verified".
2. **Honest scores.** No fake confidence percentages. "No Reliable Match" is a valid answer.
3. **Case-scoped and short-lived.** Data is deleted when the case closes or the event ends (`data/v2/` purge).
4. **Consent in testing.** Test footage shows only people who agreed to be recorded.

## 7. Example scenario
1. A parent reports Riya missing near Gate 2 with an old photo and "yellow top, blue jeans, black cap".
2. The operator creates the case and searches, with instant results:
   - **Verified** at the Gate (face).
   - **Lead** at Food Lane, where her back is to the camera. v0.5 would have missed this.
   - **Verified** at Stage Exit.
3. Trail map: Gate → Food Lane → Stage Exit, last seen 17:05.
4. Live: the watchlist is on, Riya walks past a phone camera, and the volunteer's phone buzzes with a snapshot. The volunteer taps **Found**.

## 8. Success metrics (measured; recorded in `docs/CALIBRATION.md` and the README)
- Search latency v0.5 vs v2 on the same clips
- Target rank by description per camera
- Live alert latency
- Zero false "verified" on distractors

## 9. Risks
See TRD §13. The biggest is how well descriptions rank on real crowd footage; clothing-colour attributes are the explainable fallback.

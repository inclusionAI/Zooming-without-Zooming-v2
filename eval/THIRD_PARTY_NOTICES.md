# Third-party code and data

Apache-2.0 applies only to newly authored framework code. Source under
`zwz_eval/resources/vendor/` retains its upstream or project-specific rights.
`vendor/SOURCES.json` is provenance, not a license grant.

On 2026-10-21 the human project owner confirmed authorization to publicly
distribute owner-authored adaptations and authorized contributor changes. The
scope is recorded in `resources/release_permissions.json`. This does not grant
rights over third-party originals or dataset/model redistribution; unresolved
upstream attribution/license checks below remain in force.

| Source | Identified terms / review status |
| --- | --- |
| MMLongBench | MIT upstream; original text in third_party/licenses/MMLongBench-MIT.txt |
| MileBench | Apache-2.0 code; data separately CC-BY-4.0; verify snapshot notices |
| MemLens | Original code MIT notice included; data separately CC-BY-4.0 and not bundled |
| ScreenSpot-Pro | Original MIT notice included; verify adapted GUI code provenance separately |
| UI-Vision | Dataset card indicates MIT; code notice requires separate verification |
| AgentVista adaptation | Upstream redistribution rights not yet confirmed; release blocker |
| Shared project inference and GUI adaptations | Owner/contributor public-distribution permission confirmed; upstream attribution review remains |
| MindCube Tiny raw_qa prompt | `raw_mindcube.py` preserves RawQATemplate from MindCube `src/prompt_generation/templates.py`; authors: Qineng Wang, Baiqiao Yin and the MindCube contributors. The official README declares MIT; an upstream license-text/commit verification remains part of public-release review. No benchmark question/gold is copied into the converter. |

No dataset, model weights, credentials or private evaluation outputs are intended for
inclusion. Private deployment paths in source snapshots have been mechanically redacted;
a final security scan and manual review remain mandatory.

License texts were retrieved from the named upstream repositories on 2026-10-19:
MMLongBench `LICENSE.txt`, MileBench `LICENSE`, MemLens `LICENSE-CODE`, and
ScreenSpot-Pro `LICENSE`. The archive manifest hashes these files. A repository's
license file alone does not settle ownership of later local adaptations or dataset
redistribution. Preserve original attribution; do not treat this inventory as legal clearance.
# Additional audited reference excerpts (2026-10-19)

`zwz_eval/resources/audited/` contains selected definitions from the
project's adapted Chartography, MIMIC, MMSI, SPAR, ERQA, ViewSpatial, MindCube
and evaluate_v3 scoring/parser sources. MMStar and BabyVision share the latter
platform parser. Their source/excerpt hashes and definition identities are in
`MANIFEST.json`; they are not full snapshots of the private evaluation platform.
These excerpts are **not relicensed under Apache-2.0** by inclusion. Original
authors and upstream licenses still require attribution/license review before
redistribution; owner adaptation permission has been confirmed separately.
This development archive is not cleared for
public publication. No API credentials, private deployment paths, datasets or
weights are included in these excerpts.

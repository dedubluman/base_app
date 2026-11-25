# Raw SAM 3.1 masks (checkpoint `checkpoints/sam3.1/sam3.1_multiplex.pt`, CPU port)

Per-frame 8-bit PNG masks (0/255) at the source size 1232x1078, frame index = file name.

| folder | prompt | tool | notes |
|---|---|---|---|
| `diver/` | text "person" | `diver_masks.py` | whole diver (body, shorts, fins); head/snorkel often outside |
| `shorts/` | "shorts" | `diver_parts.py` + `part_track.py` | frames 228-279 and 312-436 filled by tracker-only propagation |
| `fins/` | "flipper" | `diver_parts.py` + `part_track.py` | largest fin per chunk |
| `fins2/` | "flipper" | `fins_both.py` | every detected fin instance, both fins tracked (frames 40-436) |
| `hair/` | "hair" | `diver_parts.py` + `part_track.py` | tracker drifted onto the leg in 255-335 and 364-391; `build_classes.py --hair-drop` ignores those |

`masks_refined/player_export/` holds the temporally refined soft diver alpha (`refine_masks.py`).
`build_classes.py` merges everything into the class weights used by `render_refined.py`.

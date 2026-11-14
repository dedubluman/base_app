You are a senior video colorist and computer-vision engineer working in Claude Code on a CPU-only Linux machine (4 cores, ~15 GB RAM, no GPU). Your goal is to reproduce, on the user's own diving clip, the exact colour change shown in a reference Reels screen recording, and to grade the diver on a separate layer through a SAM 3.1 video mask.

<background>
<reference_video path="/root/.claude/uploads/1f307e5c-a909-58fc-b728-02271011b1cc/98f2e20d-ScreenRecording_10-06-2026_19-04-30_1.mov">
Screen recording of an Instagram Reel, 1290x2796, 60 fps, 9.97 s, HEVC. The picture area is roughly y 240-1790; ignore the status bar, the top icons, the right-side like/comment icons, the bottom caption and the comment bar. A man stands on a rock over a sandy seabed with a bubble column rising behind him. The camera is fixed, so the same screen position is comparable before and after.

Timeline:
- 0-1.9 s: title card ("The power of Colour Grading *Gopro*", yellow text) over the raw footage. Flat, green-cyan, bright.
- ~2.0-4.4 s: labels appear one by one: "1. Darken", "2. Restore skin tones", "3. Restore white and blue to sand". The green frame darkens from the top down while the three labels are on screen. A grid-tile wipe is drawn between steps. The tiles, text and wipe are Reels effects, not part of the grade.
- ~4.4 s to the end: the fully graded look holds. Use frames at t = 0.1 s (raw) and t = 4.8-9.9 s (graded) as measurement frames.

Measured medians (RGB 0-255, hue in degrees, S and V in 0-1), taken over patches that avoid UI text:

| Region | Raw | Graded | What changed |
|---|---|---|---|
| Top water | (42,60,57) H170 S0.30 V0.24 | (19,20,24) to (23,40,54), H207-228, S0.21-0.57, V0.09-0.21 | Near-black navy; V multiplied by about 0.4-0.5 |
| Mid water | (40,86,77) H168 S0.53 V0.34 | (17,24,31) H210 S0.45 V0.12 | V multiplied by about 0.35; blue-black |
| Bright sand | (104,226,176) H155 S0.54 V0.89 | (153,198,208) H191 S0.26 V0.82 | Highlights hold their brightness; saturation halves; near-white with a blue tint |
| Shaded sand and ripples | (88,215,170) H159 S0.59 V0.84 | (74,127,143) H194 S0.48 V0.56 | Turquoise-blue shadows with strong contrast against the white |
| Rock | n/a | (80,97,101) H191 S0.21 V0.40 | Neutral blue-grey |
| Skin, lit chest | n/a | (232,188,164) H21 S0.29 V0.91 | Warm peach, never green or cyan |
| Skin, shaded belly | n/a | (172,138,122) H19 S0.29 V0.67 | Same hue, darker |
| Black shorts | n/a | (34,54,62) H197 V0.24 | Near-black with only a faint blue reflection |

Reading of the three steps in these numbers:
1. Darken: shadows and mid-water go down hard (V x0.35-0.5) while highlights stay (V 0.89 to 0.82). It is a steep tone curve plus a vertical weighting that is strongest at the top of the frame, not a flat exposure drop.
2. Restore skin tones: the diver has his own warm colour (hue about 20, saturation about 0.29) that does not take the water's colour shift.
3. Restore white and blue to sand: every water hue rotates about +36 to +52 degrees, from green-cyan (155-170) to blue-cyan (191-210). Bright sand desaturates toward white, shaded sand stays turquoise-blue, local contrast rises so the light ripples stand out.
</reference_video>

<target_clip path="/root/.claude/uploads/1f307e5c-a909-58fc-b728-02271011b1cc/c60c22ea-player_export.mov">
The user's clip: 1232x1078, 30000/1001 fps (29.97), 573 frames, 19.12 s, H.264 yuv420p with an AAC track. A free diver (bare skin, light shorts, black fins, snorkel with a yellow tip) starts at the surface head-down with the legs above the water, dives, tumbles, and shrinks as he swims down over a flat teal sandy floor. The camera tilts, so the surface is at the top of the frame at first and the floor fills the frame later. Bubbles and surface reflections appear early. The footage has a flat blue-teal cast. The diver wears light shorts, not black as in the reference; the shorts must stay clean off-white.

Measured on the raw clip (median over diver-free patches; RGB 0-255, hue in degrees): water is (19-38, 91-108, 103-126), H183-197, S0.63-0.85, V0.39-0.49 from the surface scene to the floor scene. The diver reads teal because of the cast: torso (62,138,130) H174, shorts (75,146,143) H177 S0.49 V0.57, fins (13,46,57) V0.22. So this clip starts much closer to the reference's graded hue than the reference's raw footage did: the water needs a small hue move (about +8 to +25 degrees toward 195-210), a large darkening (V about 0.4-0.5 down to about 0.12-0.21 at the top), and a large saturation drop (S 0.8 down to about 0.45-0.55). There is no bright white sand in this clip; the floor is flat teal at V about 0.45.
</target_clip>

<environment>
- Repo: /home/user/base_app. SAM 3.1 weights: checkpoints/sam3.1/sam3.1_multiplex.pt (3.5 GB, already present). SAM 3 source: sam3/ (never edit it).
- CPU port of the SAM 3.1 video predictor: shades/sam31_cpu.py. It generates a patched copy of the package in .cache/ and leaves sam3/ untouched. shades/masking.py (run_sam31) runs it in overlapping chunks with a resumable on-disk cache.
- Existing helpers: diver_masks.py (forward detector + tracker), diver_masks_bwd.py (backward tracker pass), matte.py (edge snapping), reels_grade.py and diver_reels.py (an earlier, looser version of this grade), shades/io.py (ffmpeg rgb48 decode and dithered H.264 encode).
- Torch and the sam3 package are not installed at the start. Set up with: python3.12 -m venv .venv; pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cpu; pip install -e sam3/; pip install -r requirements_three_shades.txt.
- SAM 3.1 on CPU costs about 10-20 s per frame, so the full 573-frame clip takes 2-3 hours. Run long jobs with nohup in the background, log to a file, and rely on the chunk cache to resume.
</environment>
</background>

Here are the rules for this task:
1. SAM 3.1 is mandatory for the diver mask. Do not substitute rembg, MediaPipe, colour keying or any other segmenter for the mask itself. If SAM 3.1 cannot be loaded, stop and report exactly what failed instead of faking a mask.
2. Before the heavy run, confirm that torch imports, the CPU port builds, the checkpoint loads, and one frame yields a diver mask. Print the SAM 3.1 device (CPU port) in the log.
3. Prompt SAM 3.1 with the text "person" and track the single diver object. Run forward propagation, then a backward tracker pass, and merge them per frame: the forward mask is primary, the backward mask fills frames where the forward mask is missing or collapses. Fill short dropouts from neighbouring frames. The diver is half out of frame at the start and tiny at the end, so both ends must hold.
4. The mask covers the body, head, snorkel, shorts and fins. It excludes bubbles, the surface reflection and the floor. Snap the edge to the body (matte.py), erode 1 px, feather a few pixels scaled to the diver's size, and apply a symmetric 3-frame temporal median so the edge does not flicker and does not lag. No halo of the original green may remain around the body.
5. Grade in two layers composited through the mask, with parameters fixed for the whole clip. Do not use per-frame auto exposure or auto white balance.
   - Background (outside the mask): reproduce Steps 1 and 3 from the measured table. Match the reference's graded end state, not its hue delta, because this clip's water is already blue-cyan (H183-197) while the reference's raw water was green (H155-170); a flat +40 degree rotation would push this water into violet-blue. Use a steep shadow and midtone curve with a vertical weighting strongest at the top of the frame (top about x0.35-0.45, mid about x0.55, bottom of the frame close to x0.95), water hue moved to 195-210, saturation lowered toward 0.45-0.55, bright areas desaturated toward a blue-tinted near-white, shaded areas turquoise-blue, and local contrast that makes ripples and surface light pop. The bottom of the frame keeps the brightest water, the way the reference's sand stays bright under its near-black top. Keep a luminance floor of about V 0.06 so the surface reflections in the first seconds do not clip to flat black.
   - Diver (inside the mask): reproduce Step 2. Neutralise the water cast and bring skin to hue 19-21, saturation about 0.29, value following the pixel's own shading (lit about 0.9, shaded about 0.67). Keep the shorts a clean off-white, the fins and snorkel dark. Darken the diver moderately so he sits in the scene.
6. Match the reference by measurement, not by eye alone. Sample the graded output by region role (top water, mid water, bright floor, shaded floor, skin) and compare it with the table. Report each pair side by side. Water hue must land within 8 degrees of the target, water value within 0.06, skin hue within 6 degrees.
7. Keep the clip's native resolution, frame rate, duration and frame count. Do not crop, rescale, retime or add text, cards or transitions. Work in float with dithering at the encode.
8. Read and probe before you write code (ffprobe, extract frames). Do not assume formats.
9. Preview first. Run SAM 3.1 and the grade on frames 0-89 (3 s), export a contact sheet and a mask-overlay sheet, show them to the user, and wait for approval. Then run the full clip.
10. If part of the request cannot be done, say so plainly and name the part.

<task>
Apply the reference's three-step grade (Darken, Restore skin tones, Restore white and blue to sand) to the user's diving clip, with the diver segmented by SAM 3.1 and graded on his own layer. Start with the 3-second preview, then, after approval, process the full clip.
</task>

Think through how the measured reference transforms map onto this clip's different content (open water and a flat teal floor instead of rock and bright sand) before choosing curve and hue-rotation parameters, and verify against the table before you finish.

<verification>
Before you finish, check these by looking at the images:
- Mask overlay contact sheet every 2 s plus the first and last frames. No spill onto bubbles, surface or floor. No lag or lost limbs at the edges. Fix the mask and re-run the affected chunks if it fails.
- Before/after contact sheet of the same frames next to the graded reference frame. The look must match the three steps.
- The measurement table of target against result, with every row inside the tolerances in rule 6, or the rows that miss listed with the reason.
- ffprobe of the output: 1232x1078, 29.97 fps, 573 frames, H.264, yuv420p.
</verification>

Format the deliverable as follows. Write exactly one video, `output/diver_graded.mp4`: libx264, yuv420p, BT.709, CRF 16, preset slow, with no audio stream (`-an`). Keep masks, preview clips and contact sheets in a working folder, not in output/. End with a short final message in the user's language (Turkish) that gives the output path, the SAM 3.1 checkpoint, device and prompts used, the grade parameters chosen, the measurement table, and anything you could not fully fix.

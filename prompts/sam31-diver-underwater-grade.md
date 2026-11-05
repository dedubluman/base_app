You are a senior video colorist and computer-vision engineer working in Claude Code on a GPU machine. Your goal is to reproduce the 3-step GoPro underwater color grade shown in a reference Instagram Reel on the user's own diving clip, using a SAM 3.1 video mask so the diver is graded separately from the water.

<inputs>
<reference_reel path="/root/.claude/uploads/036b47ce-7915-59c5-b800-9ec41b26900f/074f5e31-ScreenRecording_10-06-2026_19-04-30_1.mov">
Screen recording of a Reel ("The power of Colour Grading *Gopro*"), 1290x2796, ~10 s. It shows a swimmer standing on a rock over a sandy seabed. The original footage is flat, green/teal, and washed out. The grade is applied in three labeled steps:
1. Darken: pull exposure down overall and much more toward the top of the frame (the surface side), producing a deep near-black navy gradient at the top that fades into lighter water below.
2. Restore skin tones: the person's skin returns to warm, natural tones (no green/cyan cast), and the black shorts read as neutral black.
3. Restore white and blue to sand: the sand turns bright, near-white with clean turquoise-blue shadows. The green cast is replaced by blue/cyan, contrast rises, and the caustic light patterns pop.
Final look: dark navy top → saturated blue mid-water → bright white-turquoise seabed, with a warm, natural-looking subject. Ignore the Instagram UI, captions and text overlays. They are not part of the grade.
</reference_reel>

<target_clip path="/root/.claude/uploads/036b47ce-7915-59c5-b800-9ec41b26900f/cccdc403-player_export.mov">
The user's clip: 1232x1078, 29.97 fps, ~17 s. A free diver (light-colored shorts, dark fins) starts at the surface, dives head-first, and shrinks as they swim down toward a bright sandy triangle surrounded by seagrass. There is a dense bubble trail beside and behind the diver. The footage has a flat green-teal cast.
</target_clip>
</inputs>

<rules>
1. Segment the diver with SAM 3.1 (video predictor). Use the text/concept prompt "person" (or "diver"), then add point or box prompts on a clear early frame if needed. Track the single diver object only.
2. Run propagation in both directions (forward from the prompt frame and backward from it, or from a late-frame prompt too) and merge the results, so the mask holds when the diver is half out of frame at the start and tiny at the end. Pick the mask with the highest confidence per frame, and fill short dropouts from neighboring frames instead of leaving gaps.
3. The mask must not spill or drift. It covers the diver's body, head, hair, shorts and fins. It excludes bubbles, the surface reflection, sand and seagrass. If a frame's mask clearly includes background or lags the body, fix it with an extra prompt on that frame and re-propagate. Do not hand-wave it.
4. Refine the mask edges conservatively: erode 1–2 px, then feather slightly (a few px, scaled to the diver's size), so no halo of the original green color appears around the body. Apply light temporal smoothing to stop edge flicker, but do not let it cause lag on fast motion.
5. Grade in two layers composited with the mask:
   - Background layer (everything outside the mask): Steps 1 and 3, a vertical darkening gradient strongest at the top, removal of the green cast toward blue/cyan, brighter near-white sand with turquoise shadows, and higher contrast.
   - Diver layer (inside the mask): Step 2, a moderate overall darkening consistent with the scene, the cyan/green cast neutralized, and warm natural skin restored. Light shorts stay clean off-white, not cyan. Keep the diver believable in the scene, not cut-out bright.
6. Match the reference look closely. Sample reference frames before and after the grade (avoid the UI areas) and compare tonal range and hue for the top water, mid water, sand and skin. Keep the grade temporally stable: no frame-to-frame exposure pumping as the camera tilts.
7. Work at full resolution and frame rate. Do not crop, scale, change speed, or add text.
8. Read and probe the files before writing code (ffprobe, extract frames). Do not assume formats. If SAM 3.1 is not installed, install it from the official source and say which checkpoint you used. If something cannot be done, say so plainly rather than faking the result.
</rules>

<task>
Produce the graded version of the target clip: a SAM 3.1 diver mask (bidirectional propagation), then the Reels-style grade (Darken → Restore skin tones → Restore white and blue to sand), with the diver graded separately through the mask.
</task>

<verification>
Before you finish, check your work visually:
- Export a mask overlay contact sheet (e.g. every 2 s, plus the first and last frames) and look at it. Confirm there is no spill onto bubbles or background and no drift or lag at the edges. Fix and re-run if it fails.
- Export a before/after contact sheet of the same frames next to a graded reference frame, and confirm the look matches the three steps.
- Confirm the output is 1232x1078, 29.97 fps, the same duration and frame count as the source, H.264, with no audio stream.
</verification>

<output>
Deliver one file: `output/diver_graded.mp4`, H.264 (libx264, yuv420p, CRF ~16, preset slow), original resolution and frame rate, with no audio (strip it with `-an`). Keep the intermediate mask frames and contact sheets in a working folder. In your final message, give the output path, the SAM 3.1 checkpoint and prompts used, the grade parameters you chose, and anything you could not fully fix.
</output>

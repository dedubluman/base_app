You are a senior colorist and video-processing engineer working in a Linux shell with ffmpeg and Python (OpenCV, NumPy). Your task is to add a tracked oval spotlight grade to a short underwater clip and render the final video file.

<source_video>
Path: /root/.claude/uploads/d4d55e81-869f-56b7-9505-cd616068140e/31142cc8-player_export.mov
Specs: H.264, 1232×1078, 29.97 fps (30000/1001), yuv420p, BT.709, ~6.15 s, AAC audio.
Content: A freediver films themselves with a selfie stick, shot from below. The stick runs from the bottom center of the frame up to the diver's hand. The diver is backlit by the sun, which sits just above or behind them, so they read almost as a silhouette. Behind them are light shafts, the bright water surface and dark green-teal rock. The diver moves across the frame during the clip.
</source_video>

<creative_goal>
The look should feel cinematic and claustrophobic: a lone diver in a soft pool of light, surrounded by deep, heavy blue water. The effect should feel natural. A viewer should not notice a mask, only that their eye is drawn to the diver.
</creative_goal>

Here are the rules for this task:

1. **Tracked oval mask.** Find the diver on every frame. Use a person detector (e.g., YOLO or MediaPipe) seeded on frame 0, or an OpenCV tracker (CSRT) initialized from a detected or contrast-based bounding box. Smooth the per-frame center and size over time (Savitzky–Golay or a forward-backward EMA) so the oval glides and never jitters. If tracking fails on a frame, interpolate from neighboring frames. Do not snap to a default position.
2. **Mask shape.** Use a vertical ellipse that covers the diver's head and body. Extend it downward so it also covers the upper part of the selfie stick, roughly the top third to half of its visible length. Feather the edge heavily: blur the mask with a Gaussian whose sigma is about 8–12% of the frame's short side, so the transition has no visible edge.
3. **Inside the mask, light the diver.** Lift shadows and low midtones by about 1–1.5 stops with a soft luminance curve, not a flat gain. Keep highlights where they are: the sun hotspot and light shafts must not clip any further than in the source. The lift must not create visible noise or banding. Apply a light temporal or spatial denoise to the lifted region only if needed.
4. **Skin tone.** Make the diver's skin read clearly as slightly warm: a gentle shift toward orange/amber in the shadows and midtones of the masked region (e.g., +R, slightly −B). Keep it subtle. The skin should look like warm human skin under water, not orange or sunburned. Do not warm the surrounding water inside the oval as strongly as the skin. Use a skin-tone or luminance qualifier if that helps keep the warmth on the diver.
5. **Outside the mask, claustrophobic deep blue.** Darken by about 40–55% and pull the hue from green-teal toward deep navy/cinematic blue. Lower saturation slightly and add a gentle shadow crush. Light shafts can stay faintly visible but must be subdued. Nothing outside the oval should compete with the diver.
6. **Image quality.** Process in float32 (or at least 16-bit) and convert to 8-bit only once, at the end. Add very light dithering or fine grain before quantizing so the dark blue gradients do not band.
7. **Output.** Match the source: same resolution and fps (30000/1001) and the same duration and frame count. Use H.264 with `-crf 16 -preset slow -pix_fmt yuv420p` and BT.709 color tags (`-colorspace bt709 -color_primaries bt709 -color_trc bt709`). Copy the original audio unchanged with `-c:a copy`. Save as `output/diver_spotlight.mov` in the working directory.
8. **Scope.** Keep all intermediate files (frames, masks, scripts) in a scratch directory. Do not change the source file. Do not add features nobody asked for (titles, LUT packs, stabilization, reframing).

<verification>
Do these checks before you render the final file:
1. Write a `mask_preview.mp4` that shows the smoothed ellipse outline and the tracked center drawn over the source. Watch it frame by frame (sample at least every 10th frame) and confirm the oval stays on the diver and covers the top of the stick through the whole clip.
2. Make a before/after contact sheet `compare.jpg` for frames 0, 45, 90, 135 and the last frame (source on top, graded below). Look at it and check:
   - the diver's face and body are clearly lighter than in the source, with readable detail;
   - the skin looks slightly warm and natural;
   - outside the oval is deep navy/blue and darker, with no hard mask edge;
   - the sun and light shafts are not blown out further than in the source;
   - the dark gradients show no banding or blocky noise.
3. If any check fails, adjust the parameters and check again. After that, render the final file.
4. Run ffprobe on the output and confirm that the resolution, fps, frame count, duration and audio stream match the source.
</verification>

Now process the source video according to the rules above. Think about how to make the tracking robust before you write code: the diver is a dark silhouette against a bright background and partly overlaps the sun.

When you finish, report in this order:
- the path of the final video, `compare.jpg` and `mask_preview.mp4`;
- the final parameter values you used (feather sigma, ellipse size and offset, shadow lift, warmth shift, outside darkening and hue shift);
- every frame where tracking was interpolated or looked uncertain;
- the ffprobe result next to the source specs.

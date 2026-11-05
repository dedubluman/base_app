You are a senior underwater colorist and computer-vision engineer working in Claude Code. Your job is to remove the hazy, washed-out silhouette look from a backlit freediving clip so the diver stands out from the water, the same way the reference reel makes a stingray stand out. You will use SAM 3.1 to build separate masks for the parts of the diver.

<reference_technique>
A reel titled "How to recover bad visibility" fixes a murky green clip in three steps:
1. Curves: set a firm black point and use an S-curve to cut the milky haze and bring back depth.
2. White balance: remove the green cast and move the water toward a clean cyan-blue.
3. Color separation: warm and brighten the subject while the water stays cool, so the subject separates from the background by hue as well as brightness.
In the result, the green fog is gone, the water is deep blue, the reef has dark contrast, and the stingray reads clearly.
</reference_technique>

<input>
- Source video: /root/.claude/uploads/67700377-7282-5e45-b97d-9eddb5410356/b3ac646e-player_export.mov
  1232x1078, 29.97 fps, about 6.15 s, has an audio track.
- Scene: a selfie-stick shot looking up at a snorkeler (mask, snorkel, swimsuit, fins). The sun is behind the diver, so the body is a dark, low-contrast silhouette. The whole frame has a heavy green-teal cast. Sun rays and the water surface are at the top.
</input>

<rules>
1. Use SAM 3.1 locally on the GPU with weights from Hugging Face (the official SAM 3.1 checkpoint). Before you write any SAM code, read the installed package's README and example notebooks and use its actual video-predictor API. Never guess function names. If the GPU or the weights are unavailable, stop and report exactly what is missing. Do not fall back to another model without asking.
2. Use SAM 3.1's text and concept prompts, with point prompts where needed, to build these masks and track them across every frame:
   - Separate skin masks for face, arms (left and right) and legs (left and right). These are all skin, but each one is tracked on its own so a weak track can be repaired without touching the others.
   - A full-diver mask (body, swimsuit, mask, snorkel, fins). Use it only for the local exposure lift and to protect the diver from the background grade.
   - Leave the selfie stick out of every diver mask.
   Feather each mask edge by 2–4 px and smooth it over time (for example, a short temporal median or an EMA over the mask logits) so nothing flickers or leaves a halo around the edges.
3. Run the grade in this order, working in float (16- or 32-bit) linear-ish RGB and quantizing only at the end:
   a. Global curves (whole frame): set the black point, then apply a gentle S-curve to cut the haze. Keep the highlights in the sun and the rays, and do not clip them.
   b. Global white balance: neutralize the green cast and push the water toward a deep cyan-blue, as in the reference. Keep the sun rays.
   c. Diver lift, using the full-diver mask: raise the shadows and midtones to a "balanced" level. Skin detail and the body shape should read clearly, but a little of the backlit rim feel should stay. Do not make it look like it was lit from the front.
   d. Skin color, using the skin masks (skin = diver minus hair, swimsuit, fins and dive mask; the dive-mask lens and strap are never skin even when no text prompt finds them, so exclude dark, low-saturation pixels on the face):
      - Surface skin reference: light-warm #E8B896. This is the colour the skin would have at the surface, not the colour to paint.
      - Physical water absorption: estimate the water path to the diver per frame, d in metres (combine the red loss on the diver's skin in the source, with R/G calibrated on the clip itself: its 95th-percentile frame is the shallowest (~1 m) and its 5th-percentile frame the deepest (~5 m); add any other cue you find reliable; smooth it over time with a Gaussian of about 8 frames; report what you used). Attenuate the reference per channel with T = exp(-k*d), k = (0.40, 0.075, 0.035) per metre for R, G, B, and add veiling light: target = ref*T + water*(1-T)*0.45, where water is the median graded water colour in a ring around the diver. Red goes first, then orange, so deeper skin turns paler and cyan-ish and darker.
      - Subtle and realistic: shift the skin's a*/b* by 0.6 x (target - frame median skin a*/b*) and L by 0.15 x the L difference, as an offset, so every pixel keeps its own texture, shading and specular highlights. Never replace chroma or paint a flat colour.
      - Edges: feather the skin matte about 3 px wider than the diver matte so skin blends into the water at the silhouette edge; the diver must not look cut out or pasted on.
   e. Color separation: keep the water cool and the diver warm. Raise saturation a little on the diver and lower it a little in the far background.
4. Keep the original resolution, frame rate and frame count, and copy the audio stream through unchanged.
5. Do only what is listed here: no text overlays, no reframing, no stabilization, no sharpening beyond a light local contrast on the diver.
</rules>

<deliverable>
A single graded video:
/home/user/base_app/output/diver_graded.mp4
Encode it as H.264, CRF 16 or lower, yuv420p, Rec.709 tags, with the original audio.
</deliverable>

Now do the work. Inspect the source with ffprobe first, extract frames, build and track the masks, apply the grade and encode the result.

Before you finish, check your result against these criteria and fix anything that fails:
- Pull the same 4 frames (0%, 33%, 66%, 95%) from the source and from the output and look at them. The diver must be clearly more readable, and the water must be blue rather than green.
- Overlay the skin masks on those frames. They must cover the face, arms and legs, stay off the swimsuit and the stick, and show no flicker between neighboring frames.
- Measure the median skin colour per frame in the output and plot it against the depth estimate: it must get cooler (lower a*, lower b*) as d grows and stay smooth from frame to frame. Report the shallow and deep ends.
- Look at the face: the dive mask must stay dark glass, never skin-toned.
- Confirm that the output's resolution, frame rate, frame count and audio match the source.

When you finish, report the output path, the SAM 3.1 prompts you used for each mask, the final curve, white-balance and skin values, and anything that did not meet the criteria. If something is uncertain, such as a track that drifted on a few frames, list it rather than hiding it.

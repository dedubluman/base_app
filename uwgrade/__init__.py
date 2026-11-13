"""uwgrade: SAM 3.1 mask-based cinematic underwater grade (reference-photo driven).

Modules
    sam_diver      SAM 3.1 video tracking of the diver on every k-th frame (CPU port)
    sam_keyframes  SAM 3.1 concept prompts (rock / stone wall / water surface / shorts / person) on keyframes
    masks          interpolation to every frame, edge snapping, rock/water completion, part masks
    grade          node-based grade (global, water, rocks, subject, surface, finishing)
    render         decode -> grade -> ProRes 422 HQ master + H.264
    qc             contact sheets, reference comparison, temporal stability metrics
"""

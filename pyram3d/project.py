"""One-call output helper used at the end of build scripts."""
from __future__ import annotations

from pathlib import Path


def save(scene, path, formats=("glb", "gltf", "fbx"), preview: bool = True, views: bool = True,
         viewer: bool = True, frames: bool = True, size: int = 900) -> dict:
    """Export a scene in every format plus review images and an HTML viewer.

    save(scene, "projects/chair/exports/chair") writes:
      chair.glb, chair.gltf, chair.fbx      (models)
      chair_preview.png                     (3/4 beauty render)
      chair_views.png                       (front/right/back/3-4/top/left sheet)
      chair_<Animation>.png                 (filmstrip per animation)
      chair_viewer.html                     (interactive three.js page, model embedded)
    Returns {kind: Path}.
    """
    from .render import render, render_frames, render_views
    from .viewer import write_viewer
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.name
    out: dict = {}
    for fmt in formats:
        out[fmt] = scene.export(path.parent / f"{stem}.{fmt}")
    if preview:
        out["preview"] = path.parent / f"{stem}_preview.png"
        render(scene, out["preview"], width=size, height=int(size * 0.75))
    if views:
        out["views"] = path.parent / f"{stem}_views.png"
        render_views(scene, out["views"], size=max(size // 3, 200))
    if frames:
        for a in scene.animations:
            if a.channels:
                key = f"anim_{a.name}"
                out[key] = path.parent / f"{stem}_{a.name}.png"
                render_frames(scene, out[key], a, frames=6, size=max(size // 4, 180))
    if viewer:
        out["viewer"] = path.parent / f"{stem}_viewer.html"
        glb = out.get("glb")
        files = [p.name for k, p in out.items() if k in formats]
        write_viewer(glb if glb else scene, out["viewer"], title=scene.name, files=files)
    return out

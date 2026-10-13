#!/usr/bin/env python3
"""fakinflu — worker prep : extracteur de signal pour le remplacement de personnage.

Route legere (style Genjutsu), docs/plan-replace.md. On GARDE le mouvement et on jette le « qui » :
depth (Depth-Anything-V2) + decoupe personne (MediaPipe Selfie) + facemesh (MediaPipe FaceMesh, 478
points) + separation/pitch de la voix (Demucs + rubberband). Le generateur (Wan 2.2 Animate) consomme
la video de controle produite ici.

Handler GENERATEUR : progression et apercus lisibles en direct via /stream
(return_aggregate_stream = False).

Entree (contrat transfer.py) :
  {"run_id", "r2": bool,
   "inputs":  {"video": {b64|url, name, bytes, sha256}},
   "outputs": {<role>: {put_url, key, max_bytes} | {max_bytes}},
   "outputs_json": {put_url, key} | null,
   "params":  {"mode": "fast|advanced|bulletproof", "pitch_semitones": 3,
               "person_prompt": "person", "has_audio": bool}}

Sorties (roles) : control_video (.mp4), mask (.mp4, hors fast), face_landmarks (.json, hors advanced),
                  vocals (.wav, voix pitchee), audio_original (.m4a), preview (.jpg).
Modes : fast = facemesh sur l'original ; advanced = depth-cutout compose sur le fond d'origine ;
        bulletproof = facemesh par-dessus le depth-cutout (defaut).
"""
import json, os, shutil, subprocess, tempfile, time, traceback

import runpod

from transfer import Outputs, fetch_inputs, preview_event, version_event

MAX_SECONDS = int(os.environ.get("PREP_MAX_SECONDS", "30"))
DEPTH_MODEL = os.environ.get("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs")

# --- modeles charges UNE fois au cold start (lazy : seulement ce dont le mode a besoin) -------------
_depth = None
_face = None
_selfie = None


def depth_pipe():
    global _depth
    if _depth is None:
        import torch
        from transformers import pipeline
        dev = 0 if torch.cuda.is_available() else -1
        _depth = pipeline("depth-estimation", model=DEPTH_MODEL, device=dev)
    return _depth


def face_mesh():
    global _face
    if _face is None:
        import mediapipe as mp
        _face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2,
                                                refine_landmarks=True, min_detection_confidence=0.5)
    return _face


def selfie_seg():
    global _selfie
    if _selfie is None:
        import mediapipe as mp
        _selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    return _selfie


def _run(cmd, timeout=1800):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} a echoue : {r.stderr.decode()[-500:]}")
    return r


def has_audio_stream(path):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                            "stream=index", "-of", "csv=p=0", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return bool(r.stdout.strip())
    except Exception:
        return False


def prepare_audio(video, work, semitones):
    """Demucs separe la voix de la musique, rubberband la monte de `semitones` demi-tons.
    Renvoie (vocals_wav | None, audio_original_m4a | None)."""
    import numpy as np
    import soundfile as sf
    wav = os.path.join(work, "audio.wav")
    _run(["ffmpeg", "-y", "-v", "error", "-i", video, "-vn", "-ac", "2", "-ar", "44100", wav])
    # bande-son d'origine, gardee pour le remux final (etape export)
    audio_original = os.path.join(work, "audio_original.m4a")
    _run(["ffmpeg", "-y", "-v", "error", "-i", video, "-vn", "-c:a", "aac", "-b:a", "160k", audio_original])
    # separation voix / musique
    from demucs.api import Separator
    sep = Separator(model=DEMUCS_MODEL)
    _origin, stems = sep.separate_audio_file(wav)
    vocals = stems["vocals"].mean(0).cpu().numpy() if hasattr(stems["vocals"], "cpu") else np.asarray(stems["vocals"]).mean(0)
    sr = int(sep.samplerate)
    # pitch up : cacher le « qui » sans perdre les mots (le modele a besoin de la voix pour la bouche)
    import pyrubberband as pyrb
    shifted = pyrb.pitch_shift(vocals, sr, semitones)
    out = os.path.join(work, "vocals.wav")
    sf.write(out, shifted, sr)
    return out, audio_original


def colorize_depth(depth_img):
    """PIL (depth normalisee) -> image couleur BGR (colormap INFERNO), comme la « depth video » coloree."""
    import cv2
    import numpy as np
    arr = np.asarray(depth_img).astype("float32")
    if arr.ndim == 3:
        arr = arr[..., 0]
    mn, mx = float(arr.min()), float(arr.max())
    norm = (arr - mn) / (mx - mn + 1e-6)
    u8 = (norm * 255).astype("uint8")
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def process(video, work, params, emit_progress):
    """Deroule la chaine et renvoie {role: chemin} des sorties produites."""
    import cv2
    import numpy as np

    mode = params.get("mode", "bulletproof")
    want_depth = mode in ("advanced", "bulletproof")
    want_face = mode in ("fast", "bulletproof")
    out = {}

    # 1) audio (optionnel)
    if params.get("has_audio", True) and has_audio_stream(video):
        yield from emit_progress("audio : separation voix / pitch", 5)
        try:
            vocals, audio_original = prepare_audio(video, work, int(params.get("pitch_semitones", 3)))
            if vocals:
                out["vocals"] = vocals
            if audio_original:
                out["audio_original"] = audio_original
        except Exception as e:
            traceback.print_exc()
            yield {"type": "progress", "step": f"audio ignore ({type(e).__name__})"}

    # 2) frames : depth + cutout + facemesh
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cdir = os.path.join(work, "control"); os.makedirs(cdir, exist_ok=True)
    mdir = os.path.join(work, "mask"); os.makedirs(mdir, exist_ok=True)
    landmarks = []
    i = 0
    limit = int(MAX_SECONDS * fps) if fps else total
    while True:
        ok, frame = cap.read()  # BGR
        if not ok or (limit and i >= limit):
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canvas = frame.copy()

        if want_depth:
            from PIL import Image
            res = depth_pipe()(Image.fromarray(rgb))
            depth_color = colorize_depth(res["depth"])
            depth_color = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))
            seg = selfie_seg().process(rgb)
            person = (seg.segmentation_mask > 0.5) if seg.segmentation_mask is not None else np.zeros(frame.shape[:2], bool)
            # depth des personnes pose sur le fond d'origine (le modele ne trace plus un simple contour)
            canvas = frame.copy()
            canvas[person] = depth_color[person]
            cv2.imwrite(os.path.join(mdir, f"{i:06d}.png"), (person.astype("uint8") * 255))

        if want_face:
            import mediapipe as mp
            fm = face_mesh().process(rgb)
            faces = []
            if fm.multi_face_landmarks:
                for fl in fm.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=canvas, landmark_list=fl,
                        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                    faces.append([{"x": round(p.x, 5), "y": round(p.y, 5), "z": round(p.z, 5)} for p in fl.landmark])
            landmarks.append({"frame": i, "faces": faces})

        cv2.imwrite(os.path.join(cdir, f"{i:06d}.png"), canvas)
        i += 1
        if i % 15 == 0:
            yield from emit_progress(f"images {i}/{limit or total}", 10 + int(80 * i / max(1, limit or total)))
    cap.release()
    if i == 0:
        raise RuntimeError("aucune image lue dans la video")

    # 3) encodage H.264 (yuv420p, dimensions paires) des sequences de frames
    yield from emit_progress("encodage de la video de controle", 92)
    control = os.path.join(work, "control_video.mp4")
    _run(["ffmpeg", "-y", "-v", "error", "-framerate", f"{fps}", "-i", os.path.join(cdir, "%06d.png"),
          "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", control])
    out["control_video"] = control
    if want_depth:
        mask = os.path.join(work, "mask.mp4")
        _run(["ffmpeg", "-y", "-v", "error", "-framerate", f"{fps}", "-i", os.path.join(mdir, "%06d.png"),
              "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", mask])
        out["mask"] = mask
    if want_face:
        lm = os.path.join(work, "face_landmarks.json")
        with open(lm, "w") as f:
            json.dump({"fps": fps, "count": len(landmarks), "frames": landmarks}, f)
        out["face_landmarks"] = lm

    # apercu : premiere image de controle
    preview = os.path.join(work, "preview.jpg")
    _run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(cdir, "000000.png"), "-q:v", "4", preview])
    out["preview"] = preview
    return out


def handler(job):
    inp = job.get("input") or {}
    params = inp.get("params") or {}
    work = tempfile.mkdtemp(prefix="prep_")
    t0 = time.time()
    yield version_event("prep")  # toujours en premier : le client verifie la version de l'image
    try:
        outputs = Outputs(inp)
        paths, events = fetch_inputs(inp, os.path.join(work, "in"), outputs)
        yield from events
        if "video" not in paths:
            raise ValueError("entree « video » absente")

        produced = {}
        def emit_progress(step, pct):
            yield {"type": "progress", "step": step, "pct": pct}
        gen = process(paths["video"], work, params, emit_progress)
        try:
            while True:
                ev = next(gen)
                yield ev
        except StopIteration as stop:
            produced = stop.value or {}

        for role, path in produced.items():
            yield from outputs.emit(role, path, os.path.basename(path))
        outputs.write_summary()
        yield {"type": "done", "mode": params.get("mode"), "roles": list(produced),
               "timings": {"total_s": round(time.time() - t0, 2)}}
    except Exception as e:
        traceback.print_exc()
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": False})

# fakinflu — worker RunPod : prep (extracteur de signal, docs/plan-replace.md)
# Build par RunPod (integration GitHub) a chaque GitHub Release : make worker-release W=prep
# Poids BAKES (0 $ de stockage, aucun telechargement au cold start). Limite image RunPod : 80 Go.
ARG CUDA_TAG=12.4.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch

WORKDIR /app

# ffmpeg (encodage/audio), rubberband-cli (pitch via pyrubberband), libsndfile1 (soundfile),
# libgl1 + libglib2.0-0 (opencv-headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip git ca-certificates ffmpeg rubberband-cli libsndfile1 libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel \
 && pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Versions epinglees = builds reproductibles. numpy<2 : requis par mediapipe 0.10 et opencv.
RUN pip3 install \
      "numpy==1.26.4" \
      "transformers==4.46.3" \
      "huggingface_hub>=0.27" \
      "pillow==11.0.0" \
      "opencv-python-headless==4.10.0.84" \
      "mediapipe==0.10.14" \
      "demucs==4.0.1" \
      "pyrubberband==0.3.0" \
      "soundfile==0.12.1" \
      runpod

# Depth-Anything-V2 (petit modele, rapide) mis en cache HF au build.
ARG DEPTH_MODEL=depth-anything/Depth-Anything-V2-Small-hf
ENV DEPTH_MODEL=${DEPTH_MODEL}
RUN python3 -c "from transformers import pipeline; pipeline('depth-estimation', model='${DEPTH_MODEL}')"

# Poids Demucs htdemucs mis en cache torch au build ; modeles MediaPipe (face_mesh, selfie) fournis
# dans le paquet pip, instancies ici pour verifier leur presence (aucun telechargement au runtime).
RUN python3 -c "from demucs.api import Separator; Separator(model='htdemucs')" \
 && python3 -c "import mediapipe as mp; mp.solutions.face_mesh.FaceMesh(); mp.solutions.selfie_segmentation.SelfieSegmentation()"

# Plus aucun acces reseau au runtime pour Hugging Face (cache HF deja peuple).
ENV HF_HUB_OFFLINE=1

COPY handler.py transfer.py VERSION /app/

CMD ["python3", "-u", "handler.py"]

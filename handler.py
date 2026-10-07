"""
RunPod Serverless worker for MIDI-LLM (text -> MIDI). Same generation code as the
local tool: midillm_core.py (copy of C:\\AiT\\tools\\midigen\\midillm_core.py).

Weights: slseanwu/MIDI-LLM_Llama-3.2-1B (public, Llama 3.2 Community License).
  1) RunPod "Cached model" on the endpoint (fastest cold start, no download billing)
  2) otherwise downloaded once at start (to a network volume if one is attached)

Input:
  {"prompt": "...", "n_outputs": 2, "temperature": 1.0, "top_p": 0.98,
   "max_new_tokens": 2046, "seed": null}
  {"ping": true}  -> warm-up / health check
Output:
  {"outputs": [{"midi_b64", "events", "valid", "reason"}], "seconds", "device", "model"}
"""
import glob
import os
import time
import traceback

import midillm_core

REPO_ID = midillm_core.MODEL_ID
CACHED = "/runpod-volume/huggingface-cache/hub/models--slseanwu--MIDI-LLM_Llama-3.2-1B/snapshots/*"
MAX_OUTPUTS = 8


def model_path():
    hits = sorted((p for p in glob.glob(CACHED) if os.path.isfile(os.path.join(p, "config.json"))),
                  key=os.path.getmtime)
    if hits:
        print(f"[worker] cached model: {hits[-1]}", flush=True)
        return hits[-1]
    from huggingface_hub import snapshot_download
    base = "/runpod-volume/midillm" if os.access("/runpod-volume", os.W_OK) else "/tmp/midillm"
    print(f"[worker] downloading {REPO_ID} -> {base}", flush=True)
    return snapshot_download(REPO_ID, local_dir=base)


_path = None


def handler(job):
    global _path
    inp = job.get("input") or {}
    try:
        if _path is None:
            t = time.time()
            _path = model_path()
            midillm_core.load(_path)
            print(f"[worker] model ready in {time.time() - t:.1f}s", flush=True)
        if inp.get("ping"):
            tok, model, dev = midillm_core.load(_path)
            return {"ok": True, "model": REPO_ID, "device": dev}
        prompt = (inp.get("prompt") or "").strip()
        if not prompt:
            return {"error": "prompt is required"}
        n = max(1, min(MAX_OUTPUTS, int(inp.get("n_outputs", 2))))
        res = midillm_core.generate(prompt, n_outputs=n,
                                    temperature=float(inp.get("temperature", 1.0)),
                                    top_p=float(inp.get("top_p", 0.98)),
                                    max_new_tokens=max(16, min(2046, int(inp.get("max_new_tokens", 2046)))),
                                    seed=inp.get("seed"), model_id=_path)
        res["model"] = REPO_ID
        return res
    except Exception as e:  # report instead of crashing the worker
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})

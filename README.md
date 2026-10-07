# midillm-runpod-worker

RunPod Serverless (queue) worker for [MIDI-LLM](https://github.com/slSeanWU/MIDI-LLM):
text prompt -> multitrack MIDI, transformers backend.

- Image: `ghcr.io/mollylipsprod/midillm-worker:latest` (built by GitHub Actions on every push to `main`).
- Weights are not in the image: attach `slseanwu/MIDI-LLM_Llama-3.2-1B` as the endpoint's cached
  model, or the worker downloads them once at start. Weights are under the Llama 3.2 Community License.
- Input: `{"prompt": "...", "n_outputs": 2, "temperature": 1.0, "top_p": 0.98, "max_new_tokens": 2046, "seed": null}`,
  or `{"ping": true}`.
- Output: `{"outputs": [{"midi_b64", "events", "valid", "reason"}], "seconds", "device", "model"}`.

Generation follows `generate_transformers.py` from the MIDI-LLM repo, with sampling restricted to
MIDI tokens and the event stream trimmed to whole (time, duration, note) triplets.

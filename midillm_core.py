"""
MIDI-LLM (slSeanWU/MIDI-LLM, weights slseanwu/MIDI-LLM_Llama-3.2-1B) text -> MIDI,
transformers backend. Shared by the local WSL runner (compose.py --backend local)
and the RunPod worker (midillm_runpod/worker/handler.py) so both run the same code.

Follows generate_transformers.py: system prefix + prompt + trailing space, MIDI BOS
(55026 + 128256) appended, sampling with temperature/top_p, tokens shifted back by
128256 and decoded with anticipation.convert.events_to_midi. Differences, on purpose:
  * sampling is restricted to MIDI tokens (the vLLM script does this via
    allowed_token_ids; the transformers script does not, so text/pad tokens could
    leak into the event stream)
  * the stream is cut at the first non-MIDI token and trimmed to whole
    (time, duration, note) triplets before decoding
  * no audio synthesis
"""
import base64
import io
import time

AMT_GPT2_BOS_ID = 55026
LLAMA_VOCAB_SIZE = 128256
SYSTEM_PROMPT = "You are a world-class composer. Please compose some music according to the following description: "
MODEL_ID = "slseanwu/MIDI-LLM_Llama-3.2-1B"

_cache = {}


def load(model_id=MODEL_ID, device=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    key = (model_id, device)
    if key in _cache:
        return _cache[key]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, pad_token="<|eot_id|>")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    _cache[key] = (tok, model, device)
    return _cache[key]


def _midi_only_processor(eos_ids):
    import torch
    from transformers import LogitsProcessor

    class MidiOnly(LogitsProcessor):
        def __call__(self, input_ids, scores):
            mask = torch.full_like(scores, float("-inf"))
            mask[:, LLAMA_VOCAB_SIZE:LLAMA_VOCAB_SIZE + AMT_GPT2_BOS_ID] = 0
            for e in eos_ids:
                if e is not None and e < scores.shape[1]:
                    mask[:, e] = 0
            return scores + mask

    return MidiOnly()


def tokens_to_midi_bytes(tokens):
    from anticipation.convert import events_to_midi
    clean = []
    for t in tokens:
        if not (0 <= t < AMT_GPT2_BOS_ID):
            break
        clean.append(int(t))
    clean = clean[: len(clean) - len(clean) % 3]
    if not clean:
        return None, 0
    mid = events_to_midi(clean)
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue(), len(clean) // 3


def excessive_simultaneous(tokens, limit=64):
    from collections import Counter
    times = Counter(tokens[0::3])
    return any(c > limit for c in times.values())


def generate(prompt, n_outputs=2, temperature=1.0, top_p=0.98, max_new_tokens=2046, seed=None,
             model_id=MODEL_ID, device=None):
    """Returns {"outputs": [{"midi_b64", "events", "valid", "reason"}], "seconds", "device"}."""
    import torch
    from transformers import LogitsProcessorList
    tok, model, dev = load(model_id, device)
    if seed is not None:
        torch.manual_seed(int(seed))
    ids = tok(SYSTEM_PROMPT + prompt + " ", return_tensors="pt", padding=False)["input_ids"]
    ids = torch.cat([ids, torch.tensor([[AMT_GPT2_BOS_ID + LLAMA_VOCAB_SIZE]])], dim=1).to(dev)
    eos = [tok.eos_token_id, tok.pad_token_id]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(input_ids=ids, do_sample=True, max_new_tokens=int(max_new_tokens),
                             temperature=float(temperature), top_p=float(top_p),
                             num_return_sequences=int(n_outputs), pad_token_id=tok.pad_token_id,
                             logits_processor=LogitsProcessorList([_midi_only_processor(eos)]))
    seconds = time.time() - t0
    results = []
    for seq in (out[:, ids.shape[1]:] - LLAMA_VOCAB_SIZE).cpu().tolist():
        midi, events = tokens_to_midi_bytes(seq)
        if midi is None:
            results.append({"midi_b64": None, "events": 0, "valid": False, "reason": "no MIDI events"})
        elif excessive_simultaneous(seq[: events * 3]):
            results.append({"midi_b64": None, "events": events, "valid": False,
                            "reason": "excessive simultaneous notes (authors' validity check)"})
        else:
            results.append({"midi_b64": base64.b64encode(midi).decode("ascii"), "events": events,
                            "valid": True, "reason": None})
    return {"outputs": results, "seconds": round(seconds, 1), "device": dev}

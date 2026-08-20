"""
Text chunking and the Ollama processing pipeline (synchronous; run it in a worker thread).

Operations, their options and prompts are defined in config.json:
  operations.<id>.options         -> widgets in the GUI (text / combo / spinbox / checkbox)
  operations.<id>.prompts         -> translation prompts (system_first / user_first / *_continuation)
  operations.<id>.sub_operations  -> checkbox-driven tasks that are merged into ONE prompt per chunk
"""
import json
import os
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

from config import resource_dir

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
UNWANTED_PREFIXES = (
    "here is the translation:", "here's the translation:", "translation:", "translated text:",
    "here is the complete translation:", "complete translation:", "here is the processed text:",
    "processed text:", "here is the rewritten text:", "rewritten text:", "here is the text:", "output:",
)


def load_operations(path: Optional[str] = None) -> Dict:
    """config.json next to the app (user-editable) or the bundled copy"""
    candidates = [path] if path else []
    from config import app_dir
    candidates += [os.path.join(app_dir(), "config.json"), os.path.join(resource_dir(), "config.json")]
    for c in candidates:
        if c and os.path.isfile(c):
            with open(c, encoding="utf-8") as fh:
                return json.load(fh)
    raise FileNotFoundError("config.json (operation definitions) not found")


def read_text(path: str) -> str:
    """UTF-8 (with BOM) first, then common 8-bit fallbacks; never crashes on odd bytes"""
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig",):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    for enc in ("cp1250", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# ================================================================================ chunking
class TextChunker:
    SENTENCE_END = re.compile(r"[.!?…][\"'»”’)\]]*\s+")
    PARAGRAPH = re.compile(r"\n\s*\n")

    @classmethod
    def chunk_text(cls, text: str, max_chars: int, overlap: int) -> List[Tuple[str, str, bool]]:
        """-> [(chunk, previous_context, is_first)]. max_chars <= 0 means the whole text is one chunk."""
        text = text.replace("\r\n", "\n")
        if max_chars <= 0 or len(text) <= max_chars:
            return [(text.strip(), "", True)] if text.strip() else []
        chunks = []
        pos = 0
        prev_tail = ""
        n = len(text)
        while pos < n:
            end = min(pos + max_chars, n)
            if end < n:
                end = cls._best_break(text, pos, end)
            piece = text[pos:end].strip()
            if piece:
                chunks.append((piece, prev_tail, pos == 0))
                prev_tail = piece[-overlap:] if overlap > 0 else ""
            pos = end
        return chunks

    @classmethod
    def _best_break(cls, text: str, start: int, end: int) -> int:
        """Prefer a paragraph break, then a sentence end, then a space - searching back from `end`"""
        window_start = max(start + 1, end - max(300, (end - start) // 3))
        window = text[window_start:end]
        for pattern in (cls.PARAGRAPH, cls.SENTENCE_END):
            last = None
            for m in pattern.finditer(window):
                last = m
            if last:
                return window_start + last.end()
        space = text.rfind(" ", window_start, end)
        if space > start:
            return space + 1
        return end


def deduplicate_paragraphs(text: str) -> str:
    seen, out = set(), []
    for para in text.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        key = " ".join(p.lower().split())
        if key in seen and len(key) > 40:      # only treat substantial repeats as duplicates
            continue
        seen.add(key)
        out.append(p)
    return "\n\n".join(out)


def clean_model_output(text: str, strip_thinking: bool = True) -> str:
    if strip_thinking:
        text = THINK_RE.sub("", text)
    text = text.strip()
    low = text.lower()
    for prefix in UNWANTED_PREFIXES:
        if low.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    # a fenced block wrapping the whole answer
    m = re.fullmatch(r"```[a-zA-Z]*\n(.*?)\n```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # quotes wrapping the whole answer (the prompts quote the input)
    if len(text) > 2 and text[0] == text[-1] and text[0] in "\"“„":
        text = text[1:-1].strip()
    return text


class ProcessingStopped(Exception):
    pass


class OllamaError(Exception):
    pass


# ================================================================================ processor
class OllamaProcessor:
    def __init__(self, settings: Dict, operations: Dict, log: Callable[[str, str], None],
                 status: Callable[[str], None], progress: Callable[[int, int, str], None],
                 should_stop: Callable[[], bool]):
        self.s = settings
        self.ops = operations.get("operations", operations)
        self.log = log
        self.status = status
        self.progress = progress
        self.should_stop = should_stop
        self.client = None
        self.chunks_done = 0
        self.chars_in = 0
        self.chars_out = 0

    # ------------------------------------------------------------------ server
    def connect(self):
        import ollama
        timeout = float(self.s.get("timeout") or 0) or None
        self.client = ollama.Client(host=self.s["host"], timeout=timeout)

    @staticmethod
    def list_models(host: str, timeout: float = 5.0) -> List[str]:
        import ollama
        resp = ollama.Client(host=host, timeout=timeout).list()
        models = getattr(resp, "models", None) or (resp.get("models") if isinstance(resp, dict) else [])
        names = []
        for m in models:
            name = getattr(m, "model", None) or (m.get("model") or m.get("name") if isinstance(m, dict) else None)
            if name:
                names.append(name)
        return sorted(names)

    @staticmethod
    def server_version(host: str, timeout: float = 5.0) -> str:
        import urllib.request
        with urllib.request.urlopen(host.rstrip("/") + "/api/version", timeout=timeout) as r:
            return json.loads(r.read().decode()).get("version", "?")

    # ------------------------------------------------------------------ LLM call
    def _num_ctx(self, prompt_chars: int) -> int:
        manual = int(self.s.get("num_ctx") or 0)
        if manual > 0:
            return manual
        # ~3 chars per token for prompt, the answer can be as long as the input again; round up to 1k
        need = int(prompt_chars / 3 * 2.2) + 512
        return max(2048, ((need + 1023) // 1024) * 1024)

    def chat(self, model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
        import ollama
        options = {"temperature": float(temperature), "top_p": float(self.s.get("top_p", 0.9)),
                   "num_predict": int(self.s.get("num_predict", -1)),
                   "num_ctx": self._num_ctx(len(system_prompt) + len(user_prompt))}
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        parts: List[str] = []
        try:
            stream = self.client.chat(model=model, messages=messages, stream=True, options=options,
                                      keep_alive=self.s.get("keep_alive") or None)
            for part in stream:
                if self.should_stop():
                    try:
                        stream.close()
                    except Exception:
                        pass
                    raise ProcessingStopped()
                msg = getattr(part, "message", None) or (part.get("message") if isinstance(part, dict) else None)
                content = getattr(msg, "content", None) if msg is not None and not isinstance(msg, dict) else \
                    (msg or {}).get("content")
                if content:
                    parts.append(content)
        except ProcessingStopped:
            raise
        except ollama.ResponseError as exc:
            raise OllamaError(f"Ollama error ({exc.status_code}): {exc.error}") from exc
        except Exception as exc:  # connection refused, timeouts, ...
            raise OllamaError(f"Cannot talk to Ollama at {self.s['host']}: {exc}") from exc
        return clean_model_output("".join(parts), bool(self.s.get("strip_thinking", True)))

    # ------------------------------------------------------------------ operations
    def _chunks(self, text: str):
        if self.s.get("whole_file"):
            return TextChunker.chunk_text(text, 0, 0)
        return TextChunker.chunk_text(text, int(self.s["chunk_size"]), int(self.s["overlap"]))

    def run_translation(self, text: str, op_id: str, values: Dict) -> str:
        op = self.ops[op_id]
        prompts = op["prompts"]
        src, tgt = values.get("source_language", "English"), values.get("target_language", "Czech")
        model, temperature = values["model"], float(values.get("temperature", 0.3))
        chunks = self._chunks(text)
        self.status(f"Translating {src} → {tgt} with {model} ({len(chunks)} chunks)")
        out: List[str] = []
        for i, (chunk, context, first) in enumerate(chunks):
            if first or not context:
                sys_p = prompts["system_first"].format(src_lang=src, target_lang=tgt)
                usr_p = prompts["user_first"].format(src_lang=src, target_lang=tgt, chunk=chunk)
            else:
                # hand the model the END of the previous *translation* so terminology stays consistent
                snippet = (out[-1][-int(self.s["overlap"]):] if out else context)
                sys_p = prompts["system_continuation"].format(src_lang=src, target_lang=tgt)
                usr_p = prompts["user_continuation"].format(src_lang=src, target_lang=tgt,
                                                            context_snippet=snippet, chunk=chunk)
            result = self.chat(model, sys_p, usr_p, temperature)
            out.append(result)
            self._account(chunk, result)
            self.progress(i + 1, len(chunks), f"Translating ({src} → {tgt})")
            self.partial("\n\n".join(out))
        result = "\n\n".join(out)
        return deduplicate_paragraphs(result) if self.s.get("deduplicate", True) else result

    def build_combined_prompt(self, op_id: str, enabled: List[str]) -> Tuple[str, str]:
        """One prompt that performs every enabled sub-operation in a single pass"""
        subs = self.ops[op_id].get("sub_operations", {})
        lines = ["You are a professional editor. Perform ALL of the following tasks on the text in a SINGLE pass:"]
        names = []
        for i, sid in enumerate(enabled, 1):
            cfg = subs.get(sid, {})
            instruction = cfg.get("task") or self._core_instruction(cfg.get("system", sid))
            lines.append(f"{i}. {instruction}")
            names.append(cfg.get("name") or sid.replace("_", " "))
        lines += [
            "",
            "RULES:",
            "- Do NOT translate or change the language of the text.",
            "- Do NOT omit content, change the meaning or alter facts; keep every key point and detail.",
            "- Keep the paragraph structure of the input.",
            "- Output ONLY the processed text: no explanations, no notes, no introductory remarks.",
        ]
        user = "Apply these tasks: " + ", ".join(names) + ".\n\nText:\n\n{text}"
        return "\n".join(lines), user

    @staticmethod
    def _core_instruction(system_prompt: str) -> str:
        m = re.search(r"[Yy]our task is to (.+?)(?:\.|\n)", system_prompt)
        if m:
            return m.group(1).strip().rstrip(".") + "."
        return system_prompt.split("\n")[0]

    def run_combined(self, text: str, op_id: str, values: Dict, enabled: List[str]) -> str:
        op = self.ops[op_id]
        model, temperature = values["model"], float(values.get("temperature", 0.2))
        sys_p, usr_t = self.build_combined_prompt(op_id, enabled)
        chunks = self._chunks(text)
        self.status(f"{op.get('tab_name', op_id)}: {len(enabled)} task(s) with {model} ({len(chunks)} chunks)")
        out: List[str] = []
        for i, (chunk, _ctx, _first) in enumerate(chunks):
            result = self.chat(model, sys_p, usr_t.format(text=chunk), temperature)
            out.append(result)
            self._account(chunk, result)
            self.progress(i + 1, len(chunks), op.get("tab_name", op_id))
            self.partial("\n\n".join(out))
        result = "\n\n".join(out)
        return deduplicate_paragraphs(result) if self.s.get("deduplicate", True) else result

    def _account(self, chunk: str, result: str):
        self.chunks_done += 1
        self.chars_in += len(chunk)
        self.chars_out += len(result)
        ratio = len(result) / max(1, len(chunk))
        if ratio < 0.4 or ratio > 2.5:
            self.log("WARNING", f"chunk {self.chunks_done}: output is {ratio:.1f}× the input length - check the result")

    # ------------------------------------------------------------------ files
    def output_path(self, input_path: str) -> str:
        base = os.path.basename(input_path)
        stem, ext = os.path.splitext(base)
        ext = ext or ".txt"
        folder = self.s["output_dir"] if self.s.get("output_mode") == "custom" and self.s.get("output_dir") \
            else os.path.dirname(os.path.abspath(input_path))
        return os.path.join(folder, f"{stem}{self.s.get('suffix', '_processed')}{ext}")

    def partial(self, text: str):
        """Progress file so a crash / stop never loses finished chunks"""
        if self._partial_path:
            try:
                with open(self._partial_path, "w", encoding="utf-8") as fh:
                    fh.write(text)
            except OSError:
                pass

    def process_file(self, input_path: str, pipeline: List[Tuple[str, Dict, List[str]]]) -> Dict:
        """pipeline: [(op_id, values, enabled_sub_ops)] -> stats dict"""
        if self.client is None:
            self.connect()
        out_path = self.output_path(input_path)
        if os.path.exists(out_path) and not self.s.get("overwrite", False):
            raise OllamaError(f"Output exists, skipping (enable overwrite): {os.path.basename(out_path)}")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        text = read_text(input_path)
        if not text.strip():
            raise OllamaError("Input file is empty")
        self.log("INFO", f"{os.path.basename(input_path)}: {len(text):,} characters")
        stem, ext = os.path.splitext(out_path)
        self._partial_path = f"{stem}.partial{ext}"
        start = time.time()
        self.chunks_done = self.chars_in = self.chars_out = 0
        step_files = []
        for step, (op_id, values, enabled) in enumerate(pipeline, 1):
            if self.should_stop():
                raise ProcessingStopped()
            if op_id == "translation":
                text = self.run_translation(text, op_id, values)
                step_name = self.ops[op_id].get("step_name", "translated")
            else:
                text = self.run_combined(text, op_id, values, enabled)
                step_name = self.ops[op_id].get("step_name") or op_id
            if self.s.get("save_steps", True) and len(pipeline) > 1:
                sf = f"{stem}_step{step:02d}_{step_name}{ext}"
                with open(sf, "w", encoding="utf-8") as fh:
                    fh.write(text)
                step_files.append(sf)
                self.log("INFO", f"step {step} saved: {os.path.basename(sf)}")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        try:
            os.remove(self._partial_path)
        except OSError:
            pass
        self._partial_path = None
        elapsed = time.time() - start
        self.log("INFO", f"written {os.path.basename(out_path)} ({len(text):,} chars, {elapsed:.0f}s)")
        return {"output": out_path, "chunks": self.chunks_done, "chars_in": self.chars_in,
                "chars_out": self.chars_out, "elapsed": elapsed, "steps": step_files}

    _partial_path: Optional[str] = None

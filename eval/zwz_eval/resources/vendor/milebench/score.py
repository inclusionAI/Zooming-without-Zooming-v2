#!/usr/bin/env python
# -*- coding: utf-8 -*-
'MileBench \u5b98\u65b9\u7b97\u6cd5\u91cd\u7b97 - \u542b LLM \u62bd\u7b54\u6848 (\u7528 Qwen3.6-35B-A3B)\n   \u5bf9\u6240\u6709 multi-choice \u9898 -> \u8ba9 LLM \u62bd\u7b54\u6848\u5b57\u6bcd -> \u8ddf GT \u5b57\u6bcd\u6bd4\u5bf9\u7b97 acc\n   open-ended -> ROUGE-L F1 (\u4e0e\u4e4b\u524d\u4e00\u81f4)\n   Needle -> \u5b57\u9762\u5b50\u4e32 (\u4e0e\u4e4b\u524d\u4e00\u81f4)\n\n   \u8f93\u51fa: 4 \u7c7b\u522b (Realistic Temporal / Realistic Semantic / Diagnostic / Overall) macro-avg\n'
import os, sys, json, re, threading, time, glob, multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI  # pip install openai

def configured_models():
    """Select only explicitly requested model IDs; never infer a private roster."""
    models = os.environ.get("MILEBENCH_MODELS", "").split()
    if not models:
        raise ValueError("Set MILEBENCH_MODELS to the model IDs to score (space-separated).")
    return models


ALL_MODELS = configured_models()

# Output paths are caller-configurable, with relative public defaults.
EVAL_ROOT = os.environ.get("EVAL_ROOT_MILEBENCH", "outputs/milebench")
CACHE_DIR = os.environ.get("CACHE_DIR_MILEBENCH", EVAL_ROOT + "/llm_cache")
os.makedirs(CACHE_DIR, exist_ok=True)  # concurrent shard-safe

# Preserved upstream logic; see the corresponding protocol documentation.
# Preserved upstream logic; see the corresponding protocol documentation.
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
LLM_CACHE_PREFIX = EVAL_ROOT + "/llm_extract"
LLM_CACHE = "%s.shard%d.jsonl" % (LLM_CACHE_PREFIX, SHARD_ID)  # Preserved upstream logic; see the corresponding protocol documentation.

realistic_temporal = ["ActionLocalization","ActionPrediction","ActionSequence","CharacterOrder","CounterfactualInference","EgocentricNavigation","MovingAttribute","MovingDirection","ObjectExistence","ObjectInteraction","ObjectShuffle","SceneTransition","StateChange"]
realistic_semantic = ["ALFRED","CLEVR-Change","DocVQA","IEdit","MMCoQA","MultiModalQA","nuscenes","OCR-VQA","SlideVQA","Spot-the-Diff","TQA","WebQA","WikiVQA"]
diagnostic = ["TextNeedleInAHaystack","ImageNeedleInAHaystack","GPR1200"]

def get_category(task):
    if task in realistic_temporal: return "Realistic Temporal"
    if task in realistic_semantic: return "Realistic Semantic"
    if task in diagnostic: return "Diagnostic"
    return "Unknown"

_period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
_comma_strip  = re.compile(r"(\d)(\,)(\d)")
_punct = [";", r"/","[","]",'"',"{","}","(",")","=","+","\\","_","-",">","<","@","`",",","?","!"]

def process(text):
    if not text: return ""
    out = text
    for p in _punct:
        if (p+" " in text) or (" "+p in text) or re.search(_comma_strip, text):
            out = out.replace(p, "")
        else: out = out.replace(p, " ")
    out = _period_strip.sub("", out)
    out = out.strip("'").strip('"').strip().lower()
    return out

def _preprocess_option(opt):
    s = process(opt)
    for c in ["\\",".","^","$","*","+","?","{","}","[","]","|","(",")"]:
        s = s.replace(c, "\\"+c)
    return s

def char_for(idx):
    if idx < 26: return chr(idx+65)
    if idx < 52: return "A"+chr(idx+65-26)
    return "B"+chr(idx+65-52)

# ============================================================================
# Preserved upstream logic; see the corresponding protocol documentation.
# ============================================================================
def _lcs_len(a, b):
    if not a or not b: return 0
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        ai = a[i-1]; dp_i = dp[i]; dp_im1 = dp[i-1]
        for j in range(1, n+1):
            if ai == b[j-1]: dp_i[j] = dp_im1[j-1] + 1
            else: dp_i[j] = dp_im1[j] if dp_im1[j] >= dp_i[j-1] else dp_i[j-1]
    return dp[m][n]

def rouge_l_f1(pred, gt):
    pred_sents = [s for s in pred.split(".") if s.strip()]
    gt_sents = [s for s in gt.split(".") if s.strip()]
    if not pred_sents or not gt_sents: return 0.0
    ref_words = []
    for s in gt_sents: ref_words.extend(s.split())
    m = len(set(ref_words))
    ev_words = []
    for s in pred_sents: ev_words.extend(s.split())
    n = len(set(ev_words))
    if m == 0 or n == 0: return 0.0
    prev_union = set(); total_union_count = 0
    for ref_s in gt_sents:
        ref_words_one = ref_s.split()
        lcs_union = set(prev_union)
        for ev_s in pred_sents:
            ev_words_one = ev_s.split()
            m_l = len(ref_words_one); n_l = len(ev_words_one)
            dp = [[0]*(n_l+1) for _ in range(m_l+1)]
            for i in range(1, m_l+1):
                for j in range(1, n_l+1):
                    if ref_words_one[i-1] == ev_words_one[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = dp[i-1][j] if dp[i-1][j] >= dp[i][j-1] else dp[i][j-1]
            i = m_l; j = n_l; lcs_seq = []
            while i > 0 and j > 0:
                if ref_words_one[i-1] == ev_words_one[j-1]:
                    lcs_seq.append(ref_words_one[i-1]); i -= 1; j -= 1
                elif dp[i-1][j] >= dp[i][j-1]: i -= 1
                else: j -= 1
            lcs_seq.reverse()
            lcs_union = lcs_union.union(set(lcs_seq))
        new_count = len(lcs_union) - len(prev_union)
        total_union_count += new_count
        prev_union = lcs_union
    R = total_union_count / float(m); P = total_union_count / float(n)
    if R + P == 0: return 0.0
    return 2 * R * P / (R + P)

def judge_needle(gt, pred, is_needle=True):
    gt_ = process(gt); pred_ = process(pred)
    if is_needle: return 1 if gt_ in pred_.split() else 0
    else: return 1 if gt_ in pred_ else 0

# ============================================================================
# Preserved upstream logic; see the corresponding protocol documentation.
# ============================================================================
_clients = []
_client_index = 0
_client_lock = threading.Lock()
_judge_model = "milebench-extractor"

def init_client(base_url, model_name="milebench-extractor"):
    global _clients, _client_index, _judge_model
    # A single URL remains supported; all configured replicas share the same model.
    urls = [url.strip() for url in base_url.split(",")]
    if not urls or any(not url for url in urls):
        raise ValueError("JUDGE_URL must contain one or more nonempty base URLs")
    clients = [OpenAI(base_url=url, api_key=os.environ.get("JUDGE_API_KEY", "EMPTY"))
               for url in urls]
    with _client_lock:
        _clients = clients
        _client_index = 0
        _judge_model = model_name

def _next_client():
    global _client_index
    with _client_lock:
        if not _clients:
            raise RuntimeError("MileBench extractor clients are not initialized")
        client = _clients[_client_index % len(_clients)]
        _client_index += 1
        return client

def llm_extract_letter(question, choices_dict, model_answer):
    'question: \u539f\u9898; choices_dict: {A: txt, B: txt, ...}; model_answer: \u6a21\u578b\u539f\u6587\u56de\u7b54.\n       \u8fd4\u56de\u5355\u4e2a\u5927\u5199\u5b57\u6bcd (A/B/C/D/...).\n    '
    choices_str = "\n".join(["%s. %s" % (k, v) for k, v in sorted(choices_dict.items())])
    prompt = (
        '\u4f60\u662f\u4e00\u540d\u4e25\u683c\u7684\u9605\u5377\u52a9\u624b\u3002\u4efb\u52a1\u662f\u4ece\u6a21\u578b\u7684\u56de\u7b54\u4e2d\u63d0\u53d6\u51fa\u5b83\u9009\u4e2d\u7684\u9009\u9879\u5b57\u6bcd\u3002\n'
        '\u4e0b\u9762\u662f\u4e00\u9053\u591a\u9009\u9898:\n\n'
        '\u9898\u76ee:\n%s\n\n'
        '\u9009\u9879:\n%s\n\n'
        '\u6a21\u578b\u7684\u56de\u7b54:\n%s\n\n'
        '\u8bf7\u4ed4\u7ec6\u5206\u6790\u6a21\u578b\u7684\u56de\u7b54, \u5224\u65ad\u5b83\u6700\u7ec8\u9009\u4e86\u54ea\u4e2a\u9009\u9879\u3002\n'
        '\u53ea\u8f93\u51fa\u4e00\u4e2a**\u5927\u5199\u5b57\u6bcd**(A/B/C/D...) \u4f5c\u4e3a\u4f60\u7684\u7b54\u6848, \u4e0d\u8981\u4efb\u4f55\u5176\u4ed6\u5b57\u7b26\u3002'
    ) % (question[:500], choices_str, model_answer[:1000])
    try:
        # Route one independent request; transport retries stay on this replica.
        client = _next_client()
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=_judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                raw = (resp.choices[0].message.content or "").strip()
                # Preserved upstream logic; see the corresponding protocol documentation.
                m = re.search(r"\b([A-Z])\b", raw)
                if m: return m.group(1)
                m = re.search(r"([A-Z])", raw)
                if m: return m.group(1)
                return "?"
            except Exception as e:
                if attempt < 2:
                    time.sleep(1 + attempt * 2)
                    continue
                if os.environ.get("ZWZ_STRICT_ERRORS") == "1":
                    raise RuntimeError("MileBench extractor transport failed") from e
                return "?"
    except Exception:
        if os.environ.get("ZWZ_STRICT_ERRORS") == "1":
            raise
        return "?"

# ============================================================================
# Preserved upstream logic; see the corresponding protocol documentation.
# ============================================================================
_llm_cache = {}
_llm_lock = threading.Lock()

def load_llm_cache():
    # Preserved upstream logic; see the corresponding protocol documentation.
    for fpath in sorted(glob.glob("%s.shard*.jsonl" % LLM_CACHE_PREFIX)):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: rec = json.loads(line)
                except: continue
                key = (rec.get("model"), rec.get("sample_id"))
                _llm_cache[key] = rec.get("extracted_letter", "?")

def save_llm_record(rec):
    with _llm_lock:
        with open(LLM_CACHE, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ============================================================================
# Preserved upstream logic; see the corresponding protocol documentation.
# ============================================================================
def evaluate_subtask_with_llm(rows, model_name, num_workers=64):
    '\u5bf9 subtask \u6240\u6709\u6837\u672c\u6253\u5206.\n       multi-choice \u7528 LLM \u62bd\u5b57\u6bcd, open-ended \u7528 ROUGE-L, needle \u7528\u5b50\u4e32.\n       \u8fd4\u56de list of float (0-1) \u63d0\u53d6\u5b57\u6bcd\u540e\u7684 acc.\n    '
    # Preserved upstream logic; see the corresponding protocol documentation.
    multi_items = []
    for r in rows:
        meta = r.get("meta") or {}
        qtype = meta.get("question_type", "")
        task = r.get("task", "") or ""
        if "NeedleInAHaystack" in task or "MMCoQA" in task: continue
        if qtype != "multi-choice": continue
        multi_items.append(r)

    # Preserved upstream logic; see the corresponding protocol documentation.
    if multi_items:
        def proc_one(r):
            meta = r.get("meta") or {}
            choice_list = meta.get("choice_list") or []
            key = (model_name, r.get("sample_id"))
            with _llm_lock:
                cached = _llm_cache.get(key)
            if cached is not None:
                return key, cached
            choice_dict = {char_for(i): c for i, c in enumerate(choice_list)}
            letter = llm_extract_letter(
                r.get("query", ""),
                choice_dict,
                r.get("model_answer", "") or "",
            )
            rec = {"model": model_name, "sample_id": r.get("sample_id"),
                   "extracted_letter": letter}
            save_llm_record(rec)
            with _llm_lock:
                _llm_cache[key] = letter
            return key, letter
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futs = [pool.submit(proc_one, r) for r in multi_items]
            for f in as_completed(futs):
                f.result()

    # Preserved upstream logic; see the corresponding protocol documentation.
    scores = []
    for r in rows:
        meta = r.get("meta") or {}
        qtype = meta.get("question_type", "open-ended")
        gt = r.get("response", "") or ""
        pred = r.get("model_answer", "") or ""
        choice = meta.get("choice_list") or []
        task = r.get("task", "") or ""
        if "NeedleInAHaystack" in task:
            scores.append(judge_needle(gt, pred, True)); continue
        if "MMCoQA" in task:
            scores.append(judge_needle(gt, pred, False)); continue
        if qtype == "multi-choice":
            key = (model_name, r.get("sample_id"))
            letter = _llm_cache.get(key, "?")
            gt_char = char_for(choice.index(gt)) if gt in choice else "?"
            scores.append(1 if letter == gt_char else 0); continue
        if process(pred) == "":
            scores.append(0.0); continue
        scores.append(rouge_l_f1(process(pred), process(gt)))
    return scores

def worker(args):
    model, judge_url, judge_model = args
    init_client(judge_url, judge_model)
    mdir = "%s/%s/model_answer" % (EVAL_ROOT, model)
    if not os.path.isdir(mdir):
        return model, {}
    out = {}
    for sub in sorted(os.listdir(mdir)):
        if not sub.endswith("-test"): continue  # Preserved upstream logic; see the corresponding protocol documentation.
        fpath = None
        for fn in os.listdir("%s/%s" % (mdir, sub)):
            if fn.endswith(".jsonl"):
                fpath = "%s/%s/%s" % (mdir, sub, fn); break
        if not fpath: continue
        rows = []
        try:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    rows.append(json.loads(line))
        except: continue
        if not rows: continue
        task = rows[0].get("task","") or ""
        cate = get_category(task)
        scores = evaluate_subtask_with_llm(rows, model)
        mean = 100.0 * sum(scores) / len(scores)
        out[sub] = {"mean": mean, "task": task, "cate": cate, "n": len(scores)}
    return model, out

def print_summary(cols, cache):
    print("| benchmark | " + " | ".join(cols) + " |")
    print(("|---" * (len(cols)+1)) + "|")
    for cate_label in ["Realistic Temporal", "Realistic Semantic", "Diagnostic", "Overall"]:
        cells = []
        for m in cols:
            c = cache.get(m, {})
            means = [d["mean"] for d in c.values() if cate_label == "Overall" or d["cate"] == cate_label]
            if not means: cells.append("-"); continue
            cells.append("%.2f" % (sum(means)/len(means)))
        print("| MileBench-%s | %s |" % (cate_label, " | ".join(cells)))


if __name__ == "__main__":
    # Preserved upstream logic; see the corresponding protocol documentation.
    if os.environ.get("AGGREGATE") == "1":
        cache = {}
        for m in ALL_MODELS:
            f = "%s/%s_llm.json" % (CACHE_DIR, m)
            if not os.path.exists(f):
                print("[warn] missing %s" % f, file=sys.stderr)
                cache[m] = {}
                continue
            with open(f) as fp:
                cache[m] = json.load(fp)
        print_summary(ALL_MODELS, cache)
        sys.exit(0)

    # Preserved upstream logic; see the corresponding protocol documentation.
    cols = list(ALL_MODELS)
    if NUM_SHARDS > 1:
        n = len(cols)
        base, rem = divmod(n, NUM_SHARDS)
        sizes = [base + (1 if i < rem else 0) for i in range(NUM_SHARDS)]
        start = sum(sizes[:SHARD_ID])
        end = start + sizes[SHARD_ID]
        cols = cols[start:end]
        print("[shard %d/%d] cols=%s" % (SHARD_ID, NUM_SHARDS, json.dumps(cols, ensure_ascii=False)))

    judge_url = os.environ.get("JUDGE_URL", "http://127.0.0.1:8120/v1")
    judge_model = os.environ.get("JUDGE_MODEL", "milebench-extractor")

    load_llm_cache()
    print("[cache] loaded %d entries (own_file=%s)" % (len(_llm_cache), LLM_CACHE))

    # Preserved upstream logic; see the corresponding protocol documentation.
    cache = {}
    for m in cols:
        print("[start] %s" % m)
        t0 = time.time()
        _, c = worker((m, judge_url, judge_model))
        # Preserved upstream logic; see the corresponding protocol documentation.
        with open("%s/%s_llm.json" % (CACHE_DIR, m), "w") as f:
            json.dump(c, f)
        cache[m] = c
        print("[done] %s in %.1fs" % (m, time.time()-t0))
        sys.stdout.flush()

    # Preserved upstream logic; see the corresponding protocol documentation.
    if NUM_SHARDS == 1:
        print_summary(cols, cache)
    else:
        print("[shard %d] all done, details written to <model>_llm.json" % SHARD_ID)

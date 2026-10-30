"""Prospective S2-CGE-1 selector; no model setup, inference or label invention."""
from collections import Counter, defaultdict
import hashlib
import json
import math
import random
import re

class PreparationRefused(ValueError):
    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.details = details or {}


TOKENS = ("<|block_start|>", "<|block_end|>", "<|summary_start|>", "<|summary_end|>")
TOKEN_RE = re.compile("(" + "|".join(re.escape(x) for x in TOKENS) + ")")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_key(record):
    return digest(record["problem"]), digest(record["response"])


def stratum(record):
    difficulty=record.get("difficulty")
    if difficulty is None or (isinstance(difficulty,(int,float)) and not math.isfinite(difficulty)):
        difficulty="unknown"
    return (str(record.get("source") or "unknown"),
            str(record.get("domain") or "unknown"),str(difficulty))


def complete_target(response, final_answer):
    """Strict ordered complete nonempty block/summary pairs, then final answer."""
    parts = TOKEN_RE.split(response)
    markers = parts[1::2]
    if not markers or len(markers) % 4 or any(t != TOKENS[i % 4] for i,t in enumerate(markers)):
        return False
    # Between each opening and closing marker, a nonempty body is required.
    for index in range(1, len(parts), 2):
        if parts[index] in (TOKENS[0], TOKENS[2]) and not parts[index + 1].strip():
            return False
    tail = parts[-1].strip()
    return bool(final_answer and tail and final_answer in tail)


def tokenize_complete(record, tokenizer, cap):
    messages = [{"role":"user", "content":record["problem"]},
                {"role":"assistant", "content":record["response"]}]
    full = list(tokenizer.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=False, return_dict=True)["input_ids"])
    prompt = list(tokenizer.apply_chat_template(messages[:1], tokenize=True,
                add_generation_prompt=True, return_dict=True)["input_ids"])
    if not full or len(full) > cap:
        raise ValueError("truncated_ending")
    if full[:len(prompt)] != prompt or len(full) <= len(prompt):
        raise ValueError("prompt_boundary_or_prompt_only")
    # Requiring the whole target fit preserves every final-answer byte after
    # the real template/tokenizer, without relying on substring token search.
    return {"full_tokens":len(full), "retained_tokens":len(full),
            "supervised_tokens":len(full)-len(prompt),
            "input_ids_sha256":digest(json.dumps(full, separators=(",",":")))}


def independent_receipt_valid(label, approved_verifiers):
    return bool(label and label.get("verdict")=="correct"
        and label.get("verifier_kind")=="independent_deterministic"
        and label.get("verifier_sha256")
        and approved_verifiers.get(label.get("verifier_id"))==label.get("verifier_sha256")
        and label.get("verification_receipt_sha256")
        and label.get("_verified_receipt_sha256")==label.get("verification_receipt_sha256")
        and label.get("reference_sha256") and label.get("reference_location"))


def select_targets(records, labels, *, tokenizer, cap, seed, reserved_problem_hashes,
                   approved_verifiers, quota_policy, token_budget):
    """Select no more than 512 staged rows. All policy/data arrive explicitly.

    labels: externally reviewed deterministic verifier receipts, not teacher
    assertions. approved_verifiers maps verifier id to reviewed code digest.
    quota_policy keys are (split,source,domain,difficulty), values exact counts.
    No learning effect or pass threshold is supplied by this function.
    """
    if len(records) > 512 or not records or cap <= 0 or token_budget <= 0:
        raise ValueError("bounded nonempty pool/cap/budget required")
    if not quota_policy or not any(k[-1] == "unknown" for k in quota_policy):
        raise ValueError("explicit quotas including unknown difficulty required")
    if any(k[0] not in ("train", "screen") or not isinstance(n,int) or n < 0 for k,n in quota_policy.items()):
        raise ValueError("invalid quota policy")
    label_map = {}
    for label in labels:
        key = (label.get("problem_sha256"), label.get("response_sha256"))
        if key in label_map:
            raise ValueError("duplicate correctness label")
        label_map[key] = label
    refusals, candidates = [], defaultdict(list)
    for index, record in enumerate(records):
        reason = None
        if not all(isinstance(record.get(k),str) and record[k] for k in ("problem","response")):
            refusals.append({"index":index,"reason":"malformed_record"}); continue
        ph, rh = record_key(record)
        label = label_map.get((ph,rh))
        if ph in reserved_problem_hashes:
            reason = "reserved_evaluation_overlap"
        elif label is None:
            reason = "unknown_or_unjoinable_label"
        elif label.get("verdict") != "correct":
            reason = "wrong_or_unknown_label"
        elif not independent_receipt_valid(label,approved_verifiers):
            reason = "unreviewed_independence_or_provenance"
        elif not complete_target(record["response"],label.get("final_answer")):
            reason = "malformed_or_missing_final_answer"
        if reason is None:
            try:
                tokens = tokenize_complete(record,tokenizer,cap)
            except ValueError as e:
                reason = str(e)
        if reason:
            refusals.append({"index":index,"problem_sha256":ph,"response_sha256":rh,"reason":reason})
            continue
        candidates[ph].append((tokens["supervised_tokens"],rh,index,record,tokens))
    # Deduplicate problems BEFORE split. Cheapest independently correct complete
    # response wins, with deterministic digest/index tie-breakers.
    unique = []
    for ph, options in candidates.items():
        options.sort(key=lambda x:(x[0],x[1],x[2]))
        unique.append((ph,options[0]))
        for x in options[1:]:
            refusals.append({"index":x[2],"problem_sha256":ph,"response_sha256":x[1],"reason":"duplicate_problem"})
    # Split the same entire deduplicated staged problem pool for candidate and
    # future incumbent, independent of whether a response passes correctness.
    all_problems = sorted({digest(r["problem"]) for r in records
        if isinstance(r.get("problem"),str) and r["problem"]
        and digest(r["problem"]) not in reserved_problem_hashes})
    random.Random(seed).shuffle(all_problems)
    screen_count = (len(all_problems)+4)//5
    screen_ids = set(all_problems[:screen_count])
    pools = {"screen":[x for x in unique if x[0] in screen_ids],
             "train":[x for x in unique if x[0] not in screen_ids]}
    selected, used = {"train":[],"screen":[]}, Counter()
    for split, pool in pools.items():
        bins = defaultdict(list)
        for ph, item in pool:
            bins[stratum(item[3])].append((ph,item))
        for policy_key, count in quota_policy.items():
            if policy_key[0] != split: continue
            key = policy_key[1:]; available=bins.get(key,[])
            available.sort(key=lambda x:(x[1][0],x[0]))
            if len(available) < count:
                raise PreparationRefused("missing eligible population for quota: " + repr(policy_key), {"refusals":refusals,"eligible_unique_problems":len(unique)})
            for ph,item in available[:count]:
                if used[split]+item[0] > token_budget:
                    raise PreparationRefused("supervised token budget cannot satisfy exact quotas", {"refusals":refusals,"eligible_unique_problems":len(unique)})
                used[split] += item[0]
                selected[split].append({"record":item[3],"problem_sha256":ph,
                    "response_sha256":item[1],"tokens":item[4],"stratum":list(key)})
    if not selected["train"] or not selected["screen"]:
        raise PreparationRefused("both train and held-out screen must be populated", {"refusals":refusals,"eligible_unique_problems":len(unique)})
    return {"schema":"epyc.memento.cge_preparation.v1", "seed":seed,"cap":cap,
            "selected":selected,"supervised_tokens":dict(used),"refusals":refusals,
            "refusal_counts":dict(Counter(x["reason"] for x in refusals)),
            "eligible_unique_problems":len(unique),"screen_reserved_problems":screen_count}


def prepare_from_contract(path, expected_sha256, *, data_loader, data_dir, special_tokens):
    """No weights constructed. Missing genuine receipts refuse before tokenization."""
    from pathlib import Path
    raw = Path(path).read_bytes()
    if not expected_sha256 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("reviewed CGE contract pin required")
    config = json.loads(raw)
    if config.get("schema") != "epyc.memento.cge_contract.v1":
        raise ValueError("unknown preparation contract")
    def pinned_json(key):
        spec = config[key]
        content = Path(spec["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != spec["sha256"]:
            raise ValueError("input pin mismatch: " + key)
        return json.loads(content)
    labels = pinned_json("independent_labels")
    reserved_spec = pinned_json("reserved_evaluation_problems")
    if not isinstance(labels,list) or reserved_spec.get("schema")!="epyc.memento.reserved_eval.v1":
        raise ValueError("label receipt list and provenance-bound reserved manifest required")
    if reserved_spec.get("corpus_id")!="MATH-500" or reserved_spec.get("expected_records")!=500:
        raise ValueError("complete MATH-500 corpus identity required, not an evaluation sample")
    corpus=Path(reserved_spec["source_jsonl"]).read_bytes()
    if hashlib.sha256(corpus).hexdigest()!=reserved_spec["source_sha256"]:
        raise ValueError("reserved corpus content pin mismatch")
    question_rows=[json.loads(line) for line in corpus.decode("utf-8").splitlines() if line.strip()]
    field=reserved_spec["problem_field"]
    # Only problem text is retained; reserved answers never become labels.
    questions=[row[field] for row in question_rows]
    if len(questions)!=500 or any(not isinstance(x,str) or not x for x in questions):
        raise ValueError("incomplete or invalid reserved question corpus")
    reserved={digest(x) for x in questions}
    if len(reserved)!=500 or reserved!=set(reserved_spec["problem_sha256"]):
        raise ValueError("reserved manifest not derived from complete pinned corpus")
    if labels and not config.get("approved_verifiers"):
        raise ValueError("no reviewed independent deterministic verifier")
    for label in labels:
        label.pop("_verified_receipt_sha256",None)  # never trust a supplied internal mark
        if label.get("verdict")!="correct": continue
        try:
            location=Path(label["reference_location"])
            if hashlib.sha256(location.read_bytes()).hexdigest()!=label["reference_sha256"]:
                raise ValueError("reference pin")
            raw_receipt=Path(label["verification_receipt_path"]).read_bytes()
            receipt_hash=hashlib.sha256(raw_receipt).hexdigest()
            if receipt_hash!=label["verification_receipt_sha256"]:
                raise ValueError("verification receipt pin")
            receipt=json.loads(raw_receipt)
            if receipt.get("schema")!="epyc.independent_correctness_receipt.v1" or not receipt.get("run_id") or not receipt.get("producer_id"):
                raise ValueError("missing native independent verification identity")
            for key in ("problem_sha256","response_sha256","reference_sha256", "verifier_id",
                        "verifier_sha256","verifier_kind","verdict","final_answer"):
                if not label.get(key) or receipt.get(key)!=label[key]:
                    raise ValueError("verification receipt disagrees: "+key)
            label["_verified_receipt_sha256"]=receipt_hash
        except (KeyError,OSError,ValueError,TypeError) as error:
            label["_provenance_error"]=type(error).__name__
    # This authentic receipt schema is a prospective contract; no matching
    # genuine receipt has been found or created by this preparation.
    tokenizer_path = Path(config["tokenizer_directory"])
    if not tokenizer_path.is_dir():
        raise ValueError("local tokenizer directory required")
    # Pin every tokenizer/template file used, not only the model's name.
    for relative, expected in config["tokenizer_files"].items():
        target = (tokenizer_path / relative).resolve()
        if tokenizer_path.resolve() not in target.parents:
            raise ValueError("tokenizer manifest traversal")
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError("tokenizer/template pin mismatch")
    # AutoTokenizer can read optional token maps, vocab/model files and
    # standalone templates too; none may silently escape the declared seal.
    relevant = {str(x.relative_to(tokenizer_path)) for x in tokenizer_path.rglob("*") if x.is_file() and
        (x.name.startswith(("tokenizer", "vocab", "merges", "special_tokens", "added_tokens", "chat_template"))
         or "chat_templates" in x.relative_to(tokenizer_path).parts
         or x.name == "config.json" or x.suffix == ".model")}
    if relevant != set(config["tokenizer_files"]):
        raise ValueError("tokenizer/template file set incomplete or stale")
    required = {"tokenizer_config.json", "tokenizer.json"}
    if not required <= set(config["tokenizer_files"]):
        raise ValueError("complete tokenizer and template manifest required")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path),
            local_files_only=True, trust_remote_code=False, padding_side="right")
    missing = [x for x in special_tokens if x not in tokenizer.get_vocab()]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens":missing})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = data_loader(data_dir, max_samples=512)
    quotas = {tuple(row["key"]):row["count"] for row in config["quotas"]}
    if len(quotas) != len(config["quotas"]):
        raise ValueError("duplicate quota key")
    result = select_targets(records, labels, tokenizer=tokenizer, cap=config["cap"],
            seed=config["seed"], reserved_problem_hashes=set(reserved),
            approved_verifiers=config.get("approved_verifiers",{}), quota_policy=quotas,
            token_budget=config["supervised_token_budget_per_split"])
    result["incumbent"] = matched_incumbent(records, labels, tokenizer=tokenizer,
        cap=config["cap"], seed=config["seed"], reserved_problem_hashes=set(reserved),
        quota_policy=quotas, token_budget=config["supervised_token_budget_per_split"],
        approved_verifiers=config.get("approved_verifiers",{}))
    result["budget_semantics"] = "identical ceilings and quotas; realized token counts reported separately"
    result["acceptance_policy"] = None
    batch_size=config["preparation_batch_size"]
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or not 1<=batch_size<=512:
        raise ValueError("explicit bounded preparation batch size required")
    def padded(lengths):
        return sum(max(lengths[i:i+batch_size])*len(lengths[i:i+batch_size])
                   for i in range(0,len(lengths),batch_size))
    result["batch_size"]=batch_size
    result["padded_tokens"]={split:padded([row["tokens"]["retained_tokens"] for row in rows])
                              for split,rows in result["selected"].items()}
    result["incumbent"]["padded_tokens"]={split:padded([row["retained_tokens"] for row in rows])
                              for split,rows in result["incumbent"]["selected"].items()}
    result["contract_sha256"] = expected_sha256
    result["tokenizer_files"] = config["tokenizer_files"]
    result["staged_pool_sha256"] = digest(json.dumps(records,sort_keys=True,separators=(",",":")))
    result["producer_status"] = "preparation_only_not_training_or_learning_evidence"
    # Exact identities are ready for a later loader-consumption fixture. The
    # existing training loader is unchanged until matched-incumbent design and
    # the owner GPU/environment restriction are resolved.
    return result


def matched_incumbent(records, labels, *, tokenizer, cap, seed,
                      reserved_problem_hashes, quota_policy, token_budget, approved_verifiers):
    """Same raw problem split, seeded response choice, incumbent nonzero filter.

    Identical ceilings are a budget design, not equal realized token costs.
    Incorrect/unknown labels remain counts, never candidate eligibility.
    """
    label_map = {(x.get("problem_sha256"),x.get("response_sha256")):x for x in labels}
    grouped = defaultdict(list)
    for record in records:
        if all(isinstance(record.get(k),str) and record[k] for k in ("problem","response")):
            ph, rh = record_key(record)
            if ph not in reserved_problem_hashes: grouped[ph].append((rh,record))
    problem_ids = sorted(grouped); random.Random(seed).shuffle(problem_ids)
    screen = set(problem_ids[:(len(problem_ids)+4)//5])
    bins = defaultdict(list); refusals = []
    rng = random.Random(seed)
    for ph in problem_ids:
        options = sorted(grouped[ph],key=lambda x:x[0]); rng.shuffle(options)
        rh, record = options[0]  # seeded incumbent, never best-label selection
        messages = [{"role":"user","content":record["problem"]},
                    {"role":"assistant","content":record["response"]}]
        full = list(tokenizer.apply_chat_template(messages,tokenize=True,
                    add_generation_prompt=False,return_dict=True)["input_ids"])
        prompt = list(tokenizer.apply_chat_template(messages[:1],tokenize=True,
                    add_generation_prompt=True,return_dict=True)["input_ids"])
        retained = full[:cap]; supervised=max(0,len(retained)-min(len(prompt),len(retained)))
        if supervised == 0:
            refusals.append({"problem_sha256":ph,"response_sha256":rh,"reason":"prompt_only"}); continue
        label = label_map.get((ph,rh))
        independently_correct = independent_receipt_valid(label,approved_verifiers)
        final_answer = label.get("final_answer") if label else None
        item={"record":record,"problem_sha256":ph,"response_sha256":rh,
              "full_tokens":len(full),"retained_tokens":len(retained),
              "supervised_tokens":supervised,"truncated":len(full)>cap,
              "supplied_label_verdict":label.get("verdict") if label else "unknown",
              "independent_label_verdict":"correct" if independently_correct else "unknown",
              "complete_verified_target":independently_correct and len(full)<=cap
                    and complete_target(record["response"],final_answer)}
        bins[("screen" if ph in screen else "train",)+stratum(record)].append(item)
    selected={"train":[],"screen":[]}; used=Counter()
    for key, count in quota_policy.items():
        options=bins.get(key,[]); rng.shuffle(options)
        if len(options)<count:
            raise PreparationRefused("incumbent missing population for quota: "+repr(key),{"incumbent_refusals":refusals})
        for row in options[:count]:
            if used[key[0]]+row["supervised_tokens"]>token_budget:
                raise PreparationRefused("seeded incumbent cannot fit same token ceiling and exact quotas",{"incumbent_refusals":refusals})
            used[key[0]]+=row["supervised_tokens"];selected[key[0]].append(row)
    return {"selected":selected,"supervised_tokens":dict(used),"refusals":refusals,
            "same_per_split_token_ceiling":token_budget,"same_exact_quotas":True}


def loader_rows(report, *, tokenizer, expected_report_sha256, expected_contract_sha256, expected_cap):
    """Exact existing train_stage inputs, without changing either collator.

    Invoked only by a future reviewed training entrypoint. This function never
    constructs weights. Existing training remains separate in this draft.
    """
    if digest(json.dumps(report,sort_keys=True,separators=(",",":"))) != expected_report_sha256:
        raise ValueError("prepared report content pin mismatch")
    if report.get("contract_sha256")!=expected_contract_sha256 or report.get("cap")!=expected_cap:
        raise ValueError("prepared contract/cap mismatch")
    if not report.get("incumbent") or report.get("acceptance_policy") is not None:
        raise ValueError("unrecognized preparation policy")
    out={}; seen=set()
    for split in ("train","screen"):
        out[split]=[]
        for row in report["selected"][split]:
            record=row["record"];ph,rh=record_key(record)
            if (ph,rh)!=(row["problem_sha256"],row["response_sha256"]) or ph in seen:
                raise ValueError("selected loader identity changed or overlapped")
            seen.add(ph)
            if tokenize_complete(record,tokenizer,expected_cap)!=row["tokens"]:
                raise ValueError("actual loader tokenizer identity changed")
            out[split].append(record)
        if not out[split]: raise ValueError("empty loader split")
    return out["train"],out["screen"]


def write_preparation_attempt(destination, *, operation, contract_sha256):
    """Atomic native result/refusal envelope, not a ClaimTuple or learning claim."""
    from pathlib import Path
    import os
    import tempfile
    from datetime import datetime, timezone
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    envelope={"schema":"epyc.memento.cge_attempt.v1", "contract_sha256":contract_sha256,
              "started_at":datetime.now(timezone.utc).isoformat(),"scope":"preparation_only"}
    envelope["producer_files_sha256"]={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        for name in ("memento_cge.py","memento_sft.py")}
    try:
        report=operation()
        envelope.update(status="PREPARED",report=report,
            report_sha256=digest(json.dumps(report,sort_keys=True,separators=(",",":"))))
    except Exception as error:
        envelope.update(status="REFUSED",reason=str(error),error_type=type(error).__name__,
                        details=getattr(error,"details",{}))
    envelope["ended_at"]=datetime.now(timezone.utc).isoformat()
    raw=json.dumps(envelope,sort_keys=True,indent=2).encode()
    with tempfile.NamedTemporaryFile(dir=destination,prefix=".cge-",delete=False) as handle:
        tmp=Path(handle.name);handle.write(raw);handle.flush();os.fsync(handle.fileno())
    try:
        # link() is atomic and refuses an existing final record; never replace.
        os.link(tmp,destination/"cge-preparation.json")
    finally:
        tmp.unlink()
    return envelope

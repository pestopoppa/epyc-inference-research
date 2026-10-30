"""Independent synthetic fixture tests, prepared only; not executed.

Fixture labels/receipts below are artificial unit inputs, never research labels.
No tokenizer/model/data corpus is loaded.
"""
import json
import pytest
import memento_cge as cge


class FixtureTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        ids=[1]+[ord(x) for x in messages[0]["content"]]+[2]
        if len(messages)>1:
            ids += [ord(x) for x in messages[1]["content"]]+[3]
        return {"input_ids":ids}


def response(answer="42"):
    return "<|block_start|>work<|block_end|><|summary_start|>summary<|summary_end|>Answer: "+answer


def receipt(record, *, approved=True):
    ph,rh=cge.record_key(record)
    return {"fixture_only":True,"problem_sha256":ph,"response_sha256":rh,
        "verdict":"correct","verifier_kind":"independent_deterministic",
        "verifier_id":"fixture","verifier_sha256":"fixture-code" if approved else "unapproved-code",
        "reference_sha256":"fixture-reference","reference_location":"fixture-only",
        "verification_receipt_sha256":"fixture-receipt","_verified_receipt_sha256":"fixture-receipt",
        "final_answer":"42"}


def pool():
    return [{"problem":"fixture problem "+str(i),"response":response(),
             "source":"fixture-source","domain":"math","difficulty":None} for i in range(20)]


def select(records,labels,**changes):
    kwargs=dict(tokenizer=FixtureTokenizer(),cap=1000,seed=17,
        reserved_problem_hashes=set(),approved_verifiers={"fixture":"fixture-code"},
        quota_policy={(split,"fixture-source","math","unknown"):1 for split in ("train","screen")},
        token_budget=10000)
    kwargs.update(changes)
    return cge.select_targets(records,labels,**kwargs)


def test_duplicate_problem_never_crosses_splits_and_wrong_target_cannot_win():
    records=pool()
    wrong=dict(records[0],response="cheap wrong")
    records.append(wrong)
    labels=[receipt(x) for x in records[:-1]]+[dict(receipt(wrong),verdict="wrong")]
    got=select(records,labels)
    train={x["problem_sha256"] for x in got["selected"]["train"]}
    screen={x["problem_sha256"] for x in got["selected"]["screen"]}
    assert train and screen and train.isdisjoint(screen)
    assert got["refusal_counts"]["wrong_or_unknown_label"]==1
    assert all(x["record"]["response"]!="cheap wrong" for rows in got["selected"].values() for x in rows)


def test_unknown_labels_refuse_instead_of_borrowing_teacher_answer():
    with pytest.raises(cge.PreparationRefused) as error:
        select(pool(),[])
    assert len(error.value.details["refusals"])==20
    assert {x["reason"] for x in error.value.details["refusals"]}=={"unknown_or_unjoinable_label"}


def test_unapproved_receipt_cannot_count_as_verified_incumbent():
    records=pool(); labels=[receipt(x,approved=False) for x in records]
    got=cge.matched_incumbent(records,labels,tokenizer=FixtureTokenizer(),cap=1000,seed=17,
        reserved_problem_hashes=set(),approved_verifiers={"fixture":"fixture-code"},
        quota_policy={(split,"fixture-source","math","unknown"):1 for split in ("train","screen")},token_budget=10000)
    assert not any(x["complete_verified_target"] for rows in got["selected"].values() for x in rows)
    assert {x["independent_label_verdict"] for rows in got["selected"].values() for x in rows}=={"unknown"}


def test_reserved_overlap_is_never_selected():
    records=pool(); labels=[receipt(x) for x in records]
    reserved={cge.digest(records[0]["problem"])}
    got=select(records,labels,reserved_problem_hashes=reserved)
    assert got["refusal_counts"]["reserved_evaluation_overlap"]==1
    assert all(x["problem_sha256"] not in reserved for rows in got["selected"].values() for x in rows)


@pytest.mark.parametrize("text",["Answer: 42",response().replace("<|summary_end|>",""),
    response().replace("work",""),response().replace("Answer: 42","Answer: unknown")])
def test_malformed_or_missing_final_answer_is_not_complete(text):
    assert not cge.complete_target(text,"42")


def test_cap_that_cuts_answer_is_refused():
    r=pool()[0]
    with pytest.raises(ValueError,match="truncated_ending"):
        cge.tokenize_complete(r,FixtureTokenizer(),1)


def test_nan_difficulty_is_unknown():
    assert cge.stratum({"source":"a","domain":"b","difficulty":float("nan")})==("a","b","unknown")


def test_refusal_attempt_is_native_and_existing_output_is_immutable(tmp_path):
    def refuse(): raise cge.PreparationRefused("no genuine labels",{"refusals":[{"reason":"unknown"}]})
    envelope=cge.write_preparation_attempt(tmp_path,operation=refuse,contract_sha256="fixture-contract")
    assert envelope["status"]=="REFUSED" and envelope["details"]["refusals"]
    original=(tmp_path/"cge-preparation.json").read_bytes()
    with pytest.raises(FileExistsError):
        cge.write_preparation_attempt(tmp_path,operation=lambda:{"fixture_only":True},contract_sha256="fixture-contract")
    assert (tmp_path/"cge-preparation.json").read_bytes()==original


def test_actual_loader_rows_are_hash_bound_selected_dictionaries():
    records=pool(); labels=[receipt(x) for x in records]
    got=select(records,labels)
    got.update(contract_sha256="fixture-contract",incumbent={"fixture_only":True},acceptance_policy=None)
    pin=cge.digest(json.dumps(got,sort_keys=True,separators=(",",":")))
    train,screen=cge.loader_rows(got,tokenizer=FixtureTokenizer(),expected_report_sha256=pin,
        expected_contract_sha256="fixture-contract",expected_cap=1000)
    assert train==[x["record"] for x in got["selected"]["train"]]
    assert screen==[x["record"] for x in got["selected"]["screen"]]
    got["selected"]["train"][0]["record"]["response"]+=" tampered"
    with pytest.raises(ValueError,match="content pin mismatch"):
        cge.loader_rows(got,tokenizer=FixtureTokenizer(),expected_report_sha256=pin,
            expected_contract_sha256="fixture-contract",expected_cap=1000)

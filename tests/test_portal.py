from __future__ import annotations

import contextlib
import inspect
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from training_portal import main as app_main
from training_portal.service import Portal


@pytest.fixture()
def portal(tmp_path):
    instance = Portal(tmp_path / "data")
    instance.start_workers()
    yield instance
    instance.shutdown()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    instance = Portal(tmp_path / "data")
    monkeypatch.setattr(app_main, "portal", instance)
    with TestClient(app_main.app) as test_client:
        yield test_client, instance
    instance.shutdown()


CSV_SPRING = """batch,respondent_id,text
B1,R1,春天第一段访谈正文
B1,R2,春天第二段访谈正文
B1,R3,春天第三段访谈正文
B2,R4,另一个批次的正文
"""


def upload(instance: Portal, csv_text: str, name: str = "u.csv") -> dict:
    return instance.ingest_upload(name, csv_text.encode("utf-8"))


def wait_for(instance: Portal, job_id: int, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for job in instance.jobs():
            if job["id"] == job_id:
                if job["status"] in ("completed", "blocked", "failed"):
                    return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not settle")


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------
def test_upload_parses_headers_and_reports_counts(portal):
    result = upload(portal, CSV_SPRING)
    assert result["accepted"] == 4
    assert result["batches"] == ["B1", "B2"]
    assert portal.stats()["utterances"] == 4


def test_upload_accepts_chinese_headers(portal):
    csv_text = "访谈批次,受访者编号,文本\nB1,R1,中文表头也支持\n"
    assert upload(portal, csv_text)["accepted"] == 1


def test_upload_rejects_missing_column(portal):
    from training_portal.service import ValidationError

    with pytest.raises(ValidationError):
        upload(portal, "batch,text\nB1,缺一列\n")


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def test_generation_produces_csv_for_selected_batch(portal):
    upload(portal, CSV_SPRING)
    video = portal.create_job("B1")
    job = wait_for(portal, video)
    assert job["status"] == "completed"

    export = portal.exports()[0]
    handle, name, error = portal.open_download(export["id"])
    assert error is None and handle is not None
    content = handle.read().decode("utf-8")
    handle.close()
    assert "春天第一段访谈正文" in content
    assert "另一个批次的正文" not in content
    assert content.splitlines()[0] == "batch,respondent_id,text"


def test_generation_of_empty_batch_is_rejected(portal):
    from training_portal.service import ValidationError

    with pytest.raises(ValidationError):
        portal.create_job("does-not-exist")


# ---------------------------------------------------------------------------
# revocation
# ---------------------------------------------------------------------------
def test_revocation_deletes_rows_and_locks_existing_exports(portal):
    upload(portal, CSV_SPRING)
    job_id = portal.create_job("B1")
    wait_for(portal, job_id)
    export_id = portal.exports()[0]["id"]

    result = portal.revoke("R2")
    assert result["deleted_rows"] == 1
    assert result["affected_exports"] == 1

    handle, _name, error = portal.open_download(export_id)
    assert handle is None
    assert "撤回" in error
    assert not [e for e in portal.exports() if e["id"] == export_id and e["status"] == "ready"]


def test_revocation_keeps_unaffected_files_downloadable(portal):
    upload(portal, CSV_SPRING)
    job_b1 = portal.create_job("B1")
    job_b2 = portal.create_job("B2")
    wait_for(portal, job_b1)
    wait_for(portal, job_b2)
    exports = {e["batch"]: e for e in portal.exports()}

    portal.revoke("R1")

    handle, _name, error = portal.open_download(exports["B2"]["id"])
    assert error is None and handle is not None
    handle.close()


def test_revocation_blocks_later_upload_of_same_id(portal):
    upload(portal, CSV_SPRING)
    portal.revoke("R1")

    result = upload(portal, "batch,respondent_id,text\nB9,R1,重新上传同号文本\n")
    assert result["accepted"] == 0
    assert result["skipped_revoked"] == 1
    assert "B9" not in [b["name"] for b in portal.batches()]


def test_revocation_blocks_original_text_under_new_id(portal):
    upload(portal, CSV_SPRING)
    portal.revoke("R1")

    csv_text = "batch,respondent_id,text\nB9,R1-ALT,春天第一段访谈正文\n"
    result = upload(portal, csv_text)
    assert result["accepted"] == 0
    assert result["skipped_revoked"] == 1


def test_revocation_is_persistent_across_restart(tmp_path):
    data_dir = tmp_path / "data"
    first = Portal(data_dir)
    upload(first, CSV_SPRING)
    first.revoke("R2")
    first.shutdown()

    second = Portal(data_dir)
    try:
        handle, _name, error = second.open_download(1) if second.exports() else (None, "", "")
        assert second.stats()["revocations"] == 1
        result = upload(second, "batch,respondent_id,text\nB1,R2,换新文本再试\n")
        assert result["skipped_revoked"] == 1
        assert handle is None
    finally:
        second.shutdown()


@pytest.mark.parametrize("respondent", ["R1", "R2", "R3", "R4"])
def test_revocation_of_any_respondent_leaves_consistent_state(portal, respondent):
    upload(portal, CSV_SPRING)
    job = portal.create_job("B1")
    wait_for(portal, job)
    portal.revoke(respondent)

    in_batch = respondent in {"R1", "R2", "R3"}
    for export in portal.exports():
        if export["status"] == "ready":
            handle, _name, error = portal.open_download(export["id"])
            assert error is None
            handle.close()

    export_id = portal.exports()[0]["id"]
    handle, _n, error = portal.open_download(export_id)
    if in_batch:
        assert handle is None and "撤回" in error
    else:
        # A respondent from another batch must not take this file down.
        assert handle is not None and error is None
        handle.close()


# ---------------------------------------------------------------------------
# mid-generation revocation
# ---------------------------------------------------------------------------
def test_revocation_during_generation_prevents_publication(client):
    test_client, instance = client
    repair = "\n".join(f"B1,R{i},第{i}段较长的访谈正文内容用于演示生成过程中的撤回" for i in range(200))
    upload(instance, "batch,respondent_id,text\n" + repair)
    instance.db.set_settings({"pause_generation": "1"})

    r = test_client.post("/generate", data={"batch": "B1"})
    assert r.status_code == 200

    deadline = time.monotonic() + 5
    paused = None
    while time.monotonic() < deadline:
        paused = next((j for j in instance.jobs() if j["status"] == "paused"), None)
        if paused:
            break
        time.sleep(0.05)
    assert paused is not None, "job never reached the pause gate"

    r = test_client.post("/revoke", data={"respondent_id": "R37"})
    assert r.status_code == 200

    instance.db.set_settings({"pause_generation": "0"})
    deadline = time.monotonic() + 5
    settled = None
    while time.monotonic() < deadline:
        settled = instance.jobs()[0]
        if settled["status"] in ("blocked", "completed", "failed"):
            break
        time.sleep(0.05)
    assert settled["status"] == "blocked"

    assert instance.exports() == []
    assert "R37" in r.text or "撤回" in r.text


# ---------------------------------------------------------------------------
# delete failures, pending cleanup, restarts
# ---------------------------------------------------------------------------
def test_delete_failure_leaves_pending_and_blocks_download(portal):
    upload(portal, CSV_SPRING)
    job = portal.create_job("B1")
    wait_for(portal, job)
    export_id = portal.exports()[0]["id"]

    portal.db.set_settings({"delete_failures_remaining": 2})
    portal.revoke("R2")

    deadline = time.monotonic() + 5
    export = None
    while time.monotonic() < deadline:
        export = next(e for e in portal.exports() if e["id"] == export_id)
        if export["status"] == "pending_delete" and export["delete_attempts"] >= 1:
            break
        time.sleep(0.05)
    assert export["status"] == "pending_delete"
    assert export["last_delete_error"]
    assert export["delete_attempts"] >= 1

    handle, _name, error = portal.open_download(export_id)
    assert handle is None and error

    # The file still exists on disk but is not reachable under a download name
    # until the cleanup budget is exhausted.
    path = portal.export_dir / export["stored_name"]
    assert path.exists()

    portal.db.set_settings({"delete_failures_remaining": 0})
    portal.retry_cleanup_now()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        export = next(e for e in portal.exports() if e["id"] == export_id)
        if export["status"] == "deleted":
            break
        time.sleep(0.05)
    assert export["status"] == "deleted"
    assert not path.exists()


def test_restart_continues_pending_cleanup_and_never_reopens(tmp_path):
    data_dir = tmp_path / "data"
    first = Portal(data_dir)
    first.start_workers()
    try:
        upload(first, CSV_SPRING)
        job = first.create_job("B1")
        wait_for(first, job)
        export_id = first.exports()[0]["id"]
        first.db.set_settings({"delete_always_fail": "1"})
        first.revoke("R1")
        time.sleep(0.3)
        pending = next(e for e in first.exports() if e["id"] == export_id)
        assert pending["status"] == "pending_delete"
        assert (first.export_dir / pending["stored_name"]).exists()
        handle, _n, error = first.open_download(export_id)
        assert handle is None
    finally:
        first.shutdown()

    second = Portal(data_dir)
    second.start_workers()
    try:
        # The switch is still on after restart: cleanup retries and still fails,
        # and the file remains non-downloadable.
        time.sleep(0.4)
        row = next(e for e in second.exports() if e["id"] == export_id)
        assert row["status"] == "pending_delete"
        handle, _n, error = second.open_download(export_id)
        assert handle is None and error

        # Clearing the switch lets the retry succeed; the file is gone for good.
        second.db.set_settings({"delete_always_fail": "0"})
        second.retry_cleanup_now()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            row = next(e for e in second.exports() if e["id"] == export_id)
            if row["status"] == "deleted":
                break
            time.sleep(0.05)
        assert row["status"] == "deleted"
        assert not (second.export_dir / row["stored_name"]).exists()
    finally:
        second.shutdown()


def test_startup_sweeps_orphan_files_and_demotes_missing_files(tmp_path):
    data_dir = tmp_path / "data"
    first = Portal(data_dir)
    first.start_workers()
    try:
        upload(first, CSV_SPRING)
        job = first.create_job("B1")
        wait_for(first, job)
        stored = first.exports()[0]["stored_name"]
    finally:
        first.shutdown()

    # Simulate an interrupted delete that left the file behind and an
    # interrupted publication that left a temp file.
    assert (data_dir / "exports" / stored).unlink() is None
    (data_dir / "exports" / "orphan.csv").write_text("leaked", encoding="utf-8")
    (data_dir / "tmp" / "half-written.csv").write_text("leaked", encoding="utf-8")

    second = Portal(data_dir)
    second.start_workers()
    try:
        assert not (data_dir / "exports" / "orphan.csv").exists()
        assert not list((data_dir / "tmp").glob("*.csv"))
        row = second.exports()[0]
        assert row["status"] == "pending_delete"
        handle, _n, error = second.open_download(row["id"])
        assert handle is None
    finally:
        second.shutdown()


def test_restart_fails_interrupted_jobs_and_keeps_queued(tmp_path):
    data_dir = tmp_path / "data"
    first = Portal(data_dir)
    try:
        upload(first, CSV_SPRING)
        # Simulate a crash while the generator held a job mid-run.
        with first.db.transaction() as conn:
            conn.execute(
                "INSERT INTO jobs(batch, status, requested_at) VALUES ('B1', 'running', ?)",
                ("2026-01-01T00:00:00+00:00",),
            )
            conn.execute(
                "INSERT INTO jobs(batch, status, requested_at) VALUES ('B2', 'queued', ?)",
                ("2026-01-01T00:00:00+00:00",),
            )
    finally:
        first.shutdown()

    second = Portal(data_dir)
    second.start_workers()
    try:
        jobs = {job["id"]: job for job in second.jobs()}
        crashed = next(j for j in jobs.values() if j["batch"] == "B1")
        queued = next(j for j in jobs.values() if j["batch"] == "B2")
        assert crashed["status"] == "failed"
        assert "重启" in crashed["message"]

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            queued = next(j for j in second.jobs() if j["batch"] == "B2")
            if queued["status"] in ("completed", "blocked", "failed"):
                break
            time.sleep(0.05)
        assert queued["status"] == "completed"
    finally:
        second.shutdown()


def test_download_blocks_text_withdrawn_before_export(portal):
    """Text ingested under another id must take an export down after revocation.

    This is the adversarial ordering: the duplicate copy is exported while its
    owner still has consent, then the real owner withdraws.
    """
    upload(
        portal,
        "batch,respondent_id,text\n"
        "B1,R-A,同一段访谈原文\n"
        "B1,R-B,另一段访谈原文\n"
        "B1,R-OTHER,同一段访谈原文\n",
    )
    job = portal.create_job("B1")
    wait_for(portal, job)
    export_id = portal.exports()[0]["id"]

    handle, _n, error = portal.open_download(export_id)
    assert error is None
    handle.close()

    portal.revoke("R-A")

    handle, _n, error = portal.open_download(export_id)
    assert handle is None and error
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        row = next(e for e in portal.exports() if e["id"] == export_id)
        if row["status"] == "deleted":
            break
        time.sleep(0.05)
    assert row["status"] == "deleted"


# ---------------------------------------------------------------------------
# revoke/upload interleaving (the concurrency defect under repair)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _revoke_transaction_barrier(portal_instance):
    """Gate the single write transaction of ``revoke`` behind two events.

    ``entered`` fires when the revoker reaches its write transaction; the
    revoker then blocks on ``release``. A racing upload issued in between parks
    on SQLite's single-writer lock, reproducing the exact interleaving
    deterministically rather than by timing luck.
    """
    from training_portal.db import Database

    original = Database.transaction
    entered = threading.Event()
    release = threading.Event()

    @contextlib.contextmanager
    def gated(self):
        in_revoke = any(frame.function == "revoke" for frame in inspect.stack())
        if in_revoke and not getattr(threading.current_thread(), "_gate_used", False):
            threading.current_thread()._gate_used = True
            entered.set()
            assert release.wait(10), "revoker was never released"
        with original(self) as conn:
            yield conn

    Database.transaction = gated
    try:
        yield entered, release
    finally:
        Database.transaction = original


def test_same_id_upload_racing_revocation_never_leaks_its_text(portal):
    """The reported defect, forced into one deterministic ordering.

    A same-id CSV uploaded while the revocation's write transaction is open
    must (a) store nothing, (b) register every text it carried as withdrawn, so
    that neither a later same-id upload nor the same text under a new id can
    ever reach a training file.
    """
    from training_portal.service import text_hash

    upload(
        portal,
        "batch,respondent_id,text\n"
        "B1,R1,受访者 R1 的机密文本 alpha\n"
        "B1,R2,无关受访者 R2 的文本 beta\n",
    )
    job_id = portal.create_job("B1")
    wait_for(portal, job_id)
    affected_export = portal.exports()[0]["id"]

    racing_text = "撤回处理中才上传的同号新文本 delta"
    racing_csv = (
        "batch,respondent_id,text\n"
        "B1,R1,受访者 R1 的机密文本 alpha\n"
        f"B1,R1,{racing_text}\n"
    )

    with _revoke_transaction_barrier(portal) as (entered, release):
        result_box = {}

        def racing_upload():
            assert entered.wait(10)
            # Let this thread issue its BEGIN IMMEDIATE and block on the lock.
            time.sleep(0.3)
            result_box["result"] = portal.ingest_upload("race.csv", racing_csv.encode())

        uploader = threading.Thread(target=racing_upload)
        revoker = threading.Thread(target=portal.revoke, args=("R1",))
        uploader.start()
        revoker.start()
        time.sleep(0.2)
        release.set()
        revoker.join(10)
        uploader.join(10)
        assert not revoker.is_alive() and not uploader.is_alive()

    assert result_box["result"]["accepted"] == 0
    assert result_box["result"]["skipped_revoked"] == 2

    # Nothing from the racing upload survived into the corpus.
    assert portal.stats()["utterances"] == 1
    with portal.db.read() as conn:
        recorded = {
            row["text_hash"]
            for row in conn.execute("SELECT text_hash FROM revoked_text_hashes")
        }
    assert text_hash(racing_text) in recorded
    assert text_hash("受访者 R1 的机密文本 alpha") in recorded

    # The export published before the revoke is immediately non-downloadable.
    handle, _name, error = portal.open_download(affected_export)
    assert handle is None and error

    # Same-id re-upload and same-text-under-new-id are both refused afterwards.
    later = upload(portal, f"batch,respondent_id,text\nB1,R1,{racing_text}\n")
    assert later["accepted"] == 0 and later["skipped_revoked"] == 1
    swapped = upload(portal, f"batch,respondent_id,text\nB1,R9,{racing_text}\n")
    assert swapped["accepted"] == 0 and swapped["skipped_revoked"] == 1

    # Therefore a freshly generated batch cannot contain any withdrawn text.
    upload(portal, "batch,respondent_id,text\nB9,R8,干净的全新语料 eta\n")
    new_job = portal.create_job("B9")
    settled = wait_for(portal, new_job)
    assert settled["status"] == "completed"
    clean_export = next(e for e in portal.exports() if e["batch"] == "B9")
    handle, _name, error = portal.open_download(clean_export["id"])
    assert error is None and handle is not None
    body = handle.read().decode("utf-8")
    handle.close()
    assert racing_text not in body and "alpha" not in body and "eta" in body


def test_revoked_id_upload_registers_its_text_hashes(portal):
    """Rows filtered out for matching a revoked id still withdraw their text.

    Without this, text carried only by post-revocation same-id uploads could
    re-enter under a different respondent id.
    """
    from training_portal.service import text_hash

    upload(portal, "batch,respondent_id,text\nB1,R1,原始文本 alpha\n")
    portal.revoke("R1")
    secret = "撤回后才出现的同号文本 theta"

    result = upload(portal, f"batch,respondent_id,text\nB1,R1,{secret}\n")
    assert result["accepted"] == 0 and result["skipped_revoked"] == 1
    with portal.db.read() as conn:
        recorded = {
            row["text_hash"]
            for row in conn.execute("SELECT text_hash FROM revoked_text_hashes")
        }
    assert text_hash(secret) in recorded

    swapped = upload(portal, f"batch,respondent_id,text\nB1,R7,{secret}\n")
    assert swapped["accepted"] == 0 and swapped["skipped_revoked"] == 1


def test_revocation_while_job_paused_blocks_it_under_racing_upload(portal):
    """An in-flight generation at the pause gate plus a racing same-id upload:
    the job is blocked, nothing is published, and the upload stores nothing."""
    upload(
        portal,
        "batch,respondent_id,text\n"
        "B1,R1,机密文本 alpha\nB1,R2,无关文本 beta\n",
    )
    portal.db.set_settings({"pause_generation": "1"})
    job_id = portal.create_job("B1")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = next(j for j in portal.jobs() if j["id"] == job_id)
        if job["status"] == "paused":
            break
        time.sleep(0.05)
    assert job["status"] == "paused"
    exports_before = len(portal.exports())

    racing_csv = "batch,respondent_id,text\nB1,R1,在途期间同号新文本 delta\n"
    with _revoke_transaction_barrier(portal) as (entered, release):
        result_box = {}

        def racing_upload():
            assert entered.wait(10)
            time.sleep(0.3)
            result_box["result"] = portal.ingest_upload("race.csv", racing_csv.encode())

        uploader = threading.Thread(target=racing_upload)
        revoker = threading.Thread(target=portal.revoke, args=("R1",))
        uploader.start()
        revoker.start()
        time.sleep(0.2)
        release.set()
        revoker.join(10)
        uploader.join(10)

    portal.db.set_settings({"pause_generation": "0"})
    settled = wait_for(portal, job_id)
    assert settled["status"] == "blocked"
    assert result_box["result"]["accepted"] == 0
    assert len(portal.exports()) == exports_before
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and list(portal.tmp_dir.glob("*.csv")):
        time.sleep(0.05)
    assert not list(portal.tmp_dir.glob("*.csv"))

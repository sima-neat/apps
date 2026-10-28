"""Storage pressure, LRU order, locks and filesystem boundaries for model caching."""

import fcntl
import os
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/ci"))
import cache_test_models as cache
import model_cache_storage as storage


@pytest.fixture
def budget(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMANEAT_APPS_MODEL_CACHE_MAX_BYTES", "100")
    monkeypatch.setenv("SIMANEAT_APPS_MODEL_CACHE_RESERVE_BYTES", "20")
    root = tmp_path / "cache"
    root.mkdir()
    destination = tmp_path / "runtime"
    destination.mkdir()
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: usage(1000))
    return storage.CacheStorage(root, destination)


def usage(free):
    return namedtuple("Usage", "total used free")(2000, 2000 - free, free)


def entry(budget, key, size, used):
    path = budget.root / (key * 64)
    path.mkdir()
    (path / "model.tar.gz").write_bytes(b"x" * size)
    (path / "metadata.json").write_bytes(b"")
    os.utime(path / "metadata.json", (used, used))
    return path


def test_evicts_oldest_until_budget_fits(budget):
    old = entry(budget, "a", 40, 1)
    recent = entry(budget, "b", 40, 2)
    assert budget.ensure(30, persistent=True)
    assert not old.exists() and recent.exists()
    assert budget.summary()["evicted_bytes"] == 40


def test_hit_refreshes_lru(budget):
    touched = entry(budget, "a", 40, 1)
    old = entry(budget, "b", 40, 2)
    budget.touch(touched)
    budget.ensure(30, persistent=True)
    assert touched.exists() and not old.exists()


def test_free_space_pressure_evicts_even_below_size_limit(budget, monkeypatch):
    old = entry(budget, "a", 30, 1)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: usage(15 if old.exists() else 45))
    budget.ensure(20)
    assert not old.exists()


def test_runtime_copy_space_failure_is_explicit(budget):
    with pytest.raises(RuntimeError, match="free-space reserve"):
        budget.ensure(990)


def test_oversized_model_bypasses_cache_without_deleting_entries(budget):
    existing = entry(budget, "a", 40, 1)
    assert not budget.ensure(101, persistent=True)
    assert existing.exists()


def test_locked_entry_is_not_evicted_or_lock_file_unlinked(budget):
    active = entry(budget, "a", 80, 1)
    lock_path = budget.root / (active.name + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        inode = lock_path.stat().st_ino
        assert not budget.ensure(30, persistent=True)
        assert active.exists() and lock_path.stat().st_ino == inode
        budget.maximum = 50
        with pytest.raises(RuntimeError, match="active entries"):
            budget.ensure()


def test_symlink_entry_cannot_delete_external_data(budget, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious").write_text("keep")
    (budget.root / ("a" * 64)).symlink_to(outside)
    with pytest.raises(ValueError, match="Unsafe"):
        budget.ensure()
    assert (outside / "precious").read_text() == "keep"


def test_symlink_file_is_rejected(budget, tmp_path):
    path = entry(budget, "a", 40, 1)
    (path / "link").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="Unsafe"):
        budget.ensure()


def test_abandoned_staging_cleanup_preserves_active_and_unrelated_files(budget):
    abandoned = budget.root / (".stage-" + "a" * 64 + "-tmp")
    active = budget.root / (".stage-" + "b" * 64 + "-tmp")
    for path in (abandoned, active):
        path.mkdir()
        (path / "model.tar.gz").write_bytes(b"partial")
    unrelated = budget.root / "operator-file"
    unrelated.write_text("keep")
    with (budget.root / ("b" * 64 + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        budget.cleanup_staging()
    assert not abandoned.exists() and active.exists() and unrelated.exists()


@pytest.mark.parametrize("value", ["0", "-1", "bad"])
def test_invalid_limits_fail(value, monkeypatch):
    monkeypatch.setenv("LIMIT", value)
    with pytest.raises(ValueError):
        storage.limit("LIMIT", 1)


def test_downloader_stops_under_space_pressure(tmp_path):
    class Full:
        def ensure(self):
            raise RuntimeError("free-space reserve")

    with pytest.raises(RuntimeError, match="free-space reserve"):
        cache.run_downloader([sys.executable, "-c", "import time; time.sleep(60)"], storage=Full())


def test_downloader_propagates_download_failure(budget):
    with pytest.raises(subprocess.CalledProcessError):
        cache.run_downloader([sys.executable, "-c", "raise SystemExit(7)"], storage=budget)

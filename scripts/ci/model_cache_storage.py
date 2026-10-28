"""Bound persistent model storage; callers hold the cache-wide preparation lock."""

import fcntl
import os
import re
import shutil
import stat

GIB = 1024**3


def limit(name, default):
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be a positive byte count")
    return value


class CacheStorage:
    def __init__(self, root, destination):
        self.root = root
        self.destination = destination
        self.maximum = limit("SIMANEAT_APPS_MODEL_CACHE_MAX_BYTES", 8 * GIB)
        self.reserve = limit("SIMANEAT_APPS_MODEL_CACHE_RESERVE_BYTES", 20 * GIB)
        self.evicted_bytes = 0
        self.evicted_entries = 0

    def entries(self):
        result = []
        for entry in self.root.iterdir():
            if not re.fullmatch(r"[0-9a-f]{64}", entry.name):
                continue
            if not stat.S_ISDIR(entry.lstat().st_mode):
                raise ValueError(f"Unsafe model cache entry: {entry.name}")
            files = list(entry.iterdir())
            if any(not stat.S_ISREG(p.lstat().st_mode) for p in files):
                raise ValueError(f"Unsafe files in model cache entry: {entry.name}")
            size = sum(p.stat().st_size for p in files)
            metadata = entry / "metadata.json"
            used = metadata.stat().st_mtime if metadata.exists() else entry.stat().st_mtime
            result.append((used, entry, size))
        return sorted(result)

    def cleanup_staging(self):
        for stage in self.root.glob(".stage-*"):
            match = re.fullmatch(r"\.stage-([0-9a-f]{64})-.+", stage.name)
            if not match:
                continue  # Old workers did not record their entry lock in the name.
            if not stat.S_ISDIR(stage.lstat().st_mode):
                raise ValueError("Unsafe cache staging directory")
            with (self.root / (match[1] + ".lock")).open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if any(not stat.S_ISREG(p.lstat().st_mode) for p in stage.iterdir()):
                    raise ValueError("Unsafe cache staging files")
                self.evicted_bytes += sum(p.stat().st_size for p in stage.iterdir())
                shutil.rmtree(stage)

    def staging_bytes(self):
        total = 0
        for stage in self.root.glob(".stage-*"):
            if not stat.S_ISDIR(stage.lstat().st_mode):
                raise ValueError("Unsafe cache staging directory")
            for path in stage.iterdir():
                if not stat.S_ISREG(path.lstat().st_mode):
                    raise ValueError("Unsafe cache staging files")
                total += path.stat().st_size
        return total

    def free(self):
        return min(shutil.disk_usage(p).free for p in (self.root, self.destination))

    def ensure(self, incoming=0, *, persistent=False):
        entries = self.entries()
        total = sum(size for _, _, size in entries) + self.staging_bytes()
        if persistent and incoming > self.maximum:
            return False
        for _, entry, size in entries:
            if total + (incoming if persistent else 0) <= self.maximum and self.free() >= self.reserve + incoming:
                break
            # Never unlink lock files: all processes must lock the same inode.
            with (self.root / (entry.name + ".lock")).open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                shutil.rmtree(entry)
                total -= size
                self.evicted_bytes += size
                self.evicted_entries += 1
        fits = total + (incoming if persistent else 0) <= self.maximum
        space = self.free() >= self.reserve + incoming
        if not persistent and (not fits or not space):
            raise RuntimeError(
                "Insufficient model storage: cannot retain free-space reserve "
                f"({self.reserve} bytes), requested={incoming}, free={self.free()}, "
                f"cache={total}, limit={self.maximum}; active entries are never evicted"
            )
        return fits and space

    def touch(self, entry):
        os.utime(entry / "metadata.json", None)

    def summary(self):
        return {
            "cache_bytes": sum(s for _, _, s in self.entries()) + self.staging_bytes(),
            "cache_limit_bytes": self.maximum,
            "free_bytes": self.free(),
            "filesystem_bytes": shutil.disk_usage(self.root).total,
            "reserve_bytes": self.reserve,
            "evicted_bytes": self.evicted_bytes,
            "evicted_entries": self.evicted_entries,
        }

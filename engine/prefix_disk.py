"""Local, bounded, checksummed prefix bundles. No GPU or distributed operations here.

A bundle contains one prompt's compressed KV plus its small boundary snapshots. Sharing
the large tensors within a bundle avoids writing the entire KV once per 2K boundary.
SQLite publishes only completed, fsynced files. Orphans after a crash are recoverable and
removed on startup. Only this module's UUID-named blobs are ever removed.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import os
import pickle
from pathlib import Path
import sqlite3
import struct
import tempfile
import time
import uuid

import torch
from engine.prefix_media import media_prefix

VERSION = 2


def token_hashes(tokens, lengths, media=()):
    wanted = set(lengths)
    h, result = hashlib.sha256(), {}
    images = {start: (end, digest) for start, end, digest in media}
    for n, token in enumerate(tokens, 1):
        if n - 1 in images:
            end, digest = images[n - 1]
            h.update(b'\x00image\x00' + struct.pack('<qq', n - 1, end) + digest.encode())
        h.update(struct.pack('<q', int(token)))
        if n in wanted:
            result[n] = h.hexdigest()
    return result


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True).contiguous()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {k: device_tree(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(device_tree(v, device) for v in value)
    return value


def file_checksum(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class PrefixDisk:
    def __init__(self, root, namespace, budget_bytes):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.namespace = namespace
        self.budget = int(budget_bytes)
        if self.budget <= 0:
            raise ValueError('prefix disk budget must be positive')
        self.db_path = self.root / 'index.sqlite3'
        with self._db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS bundles (
                    id TEXT PRIMARY KEY, namespace TEXT NOT NULL, checksum TEXT NOT NULL,
                    size INTEGER NOT NULL, used REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS prefixes (
                    bundle TEXT NOT NULL REFERENCES bundles(id) ON DELETE CASCADE,
                    n INTEGER NOT NULL, hash TEXT NOT NULL, route TEXT NOT NULL,
                    PRIMARY KEY(bundle, n));
                CREATE INDEX IF NOT EXISTS prefix_match ON prefixes(n, hash);
            ''')
        os.chmod(self.db_path, 0o600)
        # The engine owns one writer per rank and joins it before a new lookup. Do not
        # share this directory between independent server processes.
        with self._db() as db:
            known = {r[0] for r in db.execute('SELECT id FROM bundles')}
        for path in self.root.glob('*.pt'):
            if len(path.stem) == 32 and all(c in '0123456789abcdef' for c in path.stem):
                if path.stem not in known:
                    path.unlink(missing_ok=True)
        for path in self.root.glob('.writing-*'):
            if path.is_file() and not path.is_symlink():
                path.unlink(missing_ok=True)
        self.evict()

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, bundle_id, payload):
        if uuid.UUID(hex=bundle_id).hex != bundle_id:
            raise ValueError('invalid bundle ID')
        snapshots = payload['snapshots']
        hashes = token_hashes(payload['ids'], snapshots, payload.get('media', ()))
        fd, temporary = tempfile.mkstemp(prefix='.writing-', dir=self.root)
        destination = self.root / (bundle_id + '.pt')
        try:
            with os.fdopen(fd, 'wb') as f:
                torch.save({'version': VERSION, 'namespace': self.namespace,
                            'bundle_id': bundle_id, **payload}, f)
                f.flush()
                os.fsync(f.fileno())
            size = os.path.getsize(temporary)
            if size > self.budget:
                return {'saved': False, 'bytes': size, 'reason': 'over_budget'}
            checksum = file_checksum(temporary)
            os.replace(temporary, destination)
            dirfd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
            with self._db() as db:
                db.execute('INSERT INTO bundles VALUES (?, ?, ?, ?, ?)',
                           (bundle_id, self.namespace, checksum, size, time.time()))
                db.executemany('INSERT INTO prefixes VALUES (?, ?, ?, ?)',
                               [(bundle_id, n, hashes[n], snap.get('route', ''))
                                for n, snap in snapshots.items()])
            self.evict()
            return {'saved': True, 'bytes': size, 'boundaries': len(snapshots)}
        finally:
            Path(temporary).unlink(missing_ok=True)

    def serving(self, hashes, route=None):
        """For each ``(n, token hash)``, the bundles that could serve it, most recently used first.

        Only bundles whose file is present at its indexed size count: the index alone must not
        make a future restore a silent miss when a fresh write would have worked. Nothing is
        touched here. On the pair a skip needs the same bundle on BOTH ranks (restore uses
        only bundle ids common to every rank), so the caller compares this across ranks and
        then calls ``touch`` with the bundles it agreed on.
        """
        route_sql = '' if route is None else ' AND p.route=?'
        result, live = {}, {}
        with self._db() as db:
            for n, digest in hashes.items():
                params = [self.namespace, n, digest]
                if route is not None:
                    params.append(route)
                rows = db.execute(
                    'SELECT p.bundle, b.size FROM prefixes p JOIN bundles b ON b.id=p.bundle '
                    'WHERE b.namespace=? AND p.n=? AND p.hash=?' + route_sql +
                    ' ORDER BY b.used DESC', params).fetchall()
                bids = []
                for bid, size in rows:
                    if bid not in live:
                        try:
                            live[bid] = (self.root / (bid + '.pt')).stat().st_size == size
                        except OSError:
                            live[bid] = False
                    if live[bid] and bid not in bids:
                        bids.append(bid)
                result[n] = bids
        return result

    def touch(self, bundle_ids):
        """Mark bundles used now, so LRU keeps the ones a skipped save relies on."""
        with self._db() as db:
            db.executemany('UPDATE bundles SET used=? WHERE id=?',
                           [(time.time(), bid) for bid in dict.fromkeys(bundle_ids)])

    def candidates(self, tokens, route=None, media=()):
        with self._db() as db:
            rows = db.execute('''SELECT p.bundle, p.n, p.hash, p.route FROM prefixes p
                JOIN bundles b ON b.id=p.bundle WHERE b.namespace=? AND p.n<=?
                ORDER BY p.n DESC, b.used DESC''', (self.namespace, len(tokens))).fetchall()
        hashes = token_hashes(tokens, (row[1] for row in rows), media)
        return [(bid, n) for bid, n, digest, stored_route in rows
                if hashes.get(n) == digest and media_prefix(media, n) is not None
                and (route is None or route == stored_route)]

    def load(self, candidate, tokens, media=()):
        bid, n = candidate
        with self._db() as db:
            row = db.execute('SELECT checksum, size FROM bundles WHERE id=? AND namespace=?',
                             (bid, self.namespace)).fetchone()
        if row is None:
            return None
        path = self.root / (bid + '.pt')
        try:
            # Check before deserialization; weights_only forbids arbitrary pickle code.
            if path.stat().st_size != row[1] or file_checksum(path) != row[0]:
                return None
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if (payload['version'] != VERSION or payload['namespace'] != self.namespace
                    or payload['bundle_id'] != bid or n not in payload['snapshots']
                    or tuple(payload['ids'][:n]) != tuple(tokens[:n])
                    or tuple(payload['snapshots'][n]['ids']) != tuple(tokens[:n])
                    or media_prefix(media, n) is None
                    or media_prefix(payload.get('media', ()), n) != media_prefix(media, n)
                    or tuple(payload['snapshots'][n].get('media', ())) != media_prefix(media, n)):
                return None
            with self._db() as db:
                db.execute('UPDATE bundles SET used=? WHERE id=?', (time.time(), bid))
            return payload
        except (OSError, ValueError, RuntimeError, KeyError, EOFError, pickle.UnpicklingError):
            return None

    def evict(self):
        with self._db() as db:
            rows = db.execute('SELECT id, size FROM bundles ORDER BY used DESC').fetchall()
            used, expired = 0, []
            for bid, size in rows:
                used += size
                if used > self.budget:
                    expired.append(bid)
            db.executemany('DELETE FROM bundles WHERE id=?', [(bid,) for bid in expired])
        for bid in expired:
            (self.root / (bid + '.pt')).unlink(missing_ok=True)

from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def vec_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float16).tobytes()


def blob_to_vec(blob: bytes | None, dim: int) -> np.ndarray:
    if blob is None:
        return np.zeros((dim,), dtype=np.float32)
    arr = np.frombuffer(blob, dtype=np.float16)
    if arr.size != dim:
        return np.zeros((dim,), dtype=np.float32)
    return arr.astype(np.float32)


class SQLiteIndex:
    """Per-sample SQLite index: segment embeddings, BM25, and pair-turn embeddings."""

    def __init__(self, db_path: str | Path, emb_dim: int):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.emb_dim = int(emb_dim)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

    def _init_schema(self) -> None:
        c = self.conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS segments (
              segment_id TEXT PRIMARY KEY,
              conv_id TEXT,
              seg_idx INTEGER,
              session_idx INTEGER,
              session_label TEXT,
              start_order INTEGER,
              end_order INTEGER,
              start_time TEXT,
              end_time TEXT,
              text_user TEXT,
              text_assistant TEXT,
              text_full TEXT,
              emb_user BLOB,
              emb_assistant BLOB,
              emb_full BLOB
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS segment_turn_embeddings (
              segment_id TEXT,
              turn_idx INTEGER,
              role TEXT,
              turn_order INTEGER,
              turn_time TEXT,
              text_user TEXT,
              text_assistant TEXT,
              text_turn TEXT,
              emb_user BLOB,
              emb_assistant BLOB,
              emb_turn BLOB,
              PRIMARY KEY(segment_id, turn_idx)
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_turn_segment ON segment_turn_embeddings(segment_id)")
        c.execute("CREATE TABLE IF NOT EXISTS bm25_terms (term TEXT PRIMARY KEY, df INTEGER)")
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS bm25_postings (
              term TEXT,
              segment_id TEXT,
              tf INTEGER,
              doc_len INTEGER,
              PRIMARY KEY(term, segment_id)
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_bm25_postings_term ON bm25_postings(term)")
        c.execute("CREATE TABLE IF NOT EXISTS bm25_meta (key TEXT PRIMARY KEY, value REAL)")
        self.conn.commit()

    def insert_segment(self, rec: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO segments(
              segment_id, conv_id, seg_idx, session_idx, session_label,
              start_order, end_order, start_time, end_time,
              text_user, text_assistant, text_full,
              emb_user, emb_assistant, emb_full
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec["segment_id"], rec["conv_id"], int(rec["seg_idx"]), rec.get("session_idx"), rec.get("session_label"),
                rec.get("start_order"), rec.get("end_order"), rec.get("start_time"), rec.get("end_time"),
                rec.get("text_user", ""), rec.get("text_assistant", ""), rec.get("text_full", ""),
                vec_to_blob(rec["emb_user"]), vec_to_blob(rec["emb_assistant"]), vec_to_blob(rec["emb_full"]),
            ),
        )

    def delete_turns_for_segment(self, segment_id: str) -> None:
        self.conn.execute("DELETE FROM segment_turn_embeddings WHERE segment_id=?", (segment_id,))

    def insert_pair_turn(self, rec: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO segment_turn_embeddings(
              segment_id, turn_idx, role, turn_order, turn_time,
              text_user, text_assistant, text_turn,
              emb_user, emb_assistant, emb_turn
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec["segment_id"], int(rec["turn_idx"]), "user_assistant_pair",
                rec.get("turn_order"), rec.get("turn_time"), rec.get("text_user", ""),
                rec.get("text_assistant", ""), rec.get("text_turn", ""),
                vec_to_blob(rec["emb_user"]), vec_to_blob(rec["emb_assistant"]), vec_to_blob(rec["emb_turn"]),
            ),
        )

    def iter_segments_with_embeddings(self) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        rows = self.conn.execute("SELECT segment_id, emb_user, emb_assistant, emb_full FROM segments").fetchall()
        return [
            (
                str(sid),
                blob_to_vec(bu, self.emb_dim),
                blob_to_vec(ba, self.emb_dim),
                blob_to_vec(bf, self.emb_dim),
                blob_to_vec(bf, self.emb_dim),
            )
            for sid, bu, ba, bf in rows
        ]

    def get_segments_by_ids(self, segment_ids: Sequence[str]) -> List[Dict[str, Any]]:
        ids = [str(x) for x in segment_ids if str(x)]
        if not ids:
            return []
        qmarks = ",".join(["?"] * len(ids))
        rows = self.conn.execute(
            f"""
            SELECT segment_id, conv_id, seg_idx, session_idx, session_label,
                   start_order, end_order, start_time, end_time,
                   text_user, text_assistant, text_full
            FROM segments WHERE segment_id IN ({qmarks})
            """,
            ids,
        ).fetchall()
        return [
            {
                "segment_id": r[0], "conv_id": r[1], "seg_idx": r[2], "session_idx": r[3], "session_label": r[4],
                "start_order": r[5], "end_order": r[6], "start_time": r[7], "end_time": r[8],
                "text_user": r[9] or "", "text_assistant": r[10] or "", "text_full": r[11] or "",
            }
            for r in rows
        ]

    def get_pair_turns_by_segment_ids(self, segment_ids: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
        ids = [str(x) for x in segment_ids if str(x)]
        if not ids:
            return {}
        qmarks = ",".join(["?"] * len(ids))
        rows = self.conn.execute(
            f"""
            SELECT segment_id, turn_idx, role, turn_order, turn_time,
                   text_user, text_assistant, text_turn, emb_user, emb_assistant, emb_turn
            FROM segment_turn_embeddings
            WHERE segment_id IN ({qmarks})
            ORDER BY segment_id, turn_idx
            """,
            ids,
        ).fetchall()
        out: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            sid = str(r[0])
            out.setdefault(sid, []).append(
                {
                    "segment_id": sid,
                    "turn_idx": int(r[1]),
                    "role": r[2],
                    "turn_order": r[3],
                    "turn_time": r[4],
                    "text_user": r[5] or "",
                    "text_assistant": r[6] or "",
                    "text_turn": r[7] or "",
                    "emb_user": blob_to_vec(r[8], self.emb_dim),
                    "emb_assistant": blob_to_vec(r[9], self.emb_dim),
                    "emb_turn": blob_to_vec(r[10], self.emb_dim),
                }
            )
        return out

    @staticmethod
    def bm25_tokenize(text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_:/.-]*", (text or "").lower())
        return [t for t in tokens if len(t) >= 2]

    def rebuild_bm25_index(self, docs: Sequence[Tuple[str, str]]) -> None:
        self.conn.execute("DELETE FROM bm25_terms")
        self.conn.execute("DELETE FROM bm25_postings")
        self.conn.execute("DELETE FROM bm25_meta")
        if not docs:
            self.conn.execute("INSERT OR REPLACE INTO bm25_meta(key, value) VALUES('N', 0)")
            self.conn.execute("INSERT OR REPLACE INTO bm25_meta(key, value) VALUES('avgdl', 0)")
            return
        term_df: Dict[str, int] = {}
        postings: List[Tuple[str, str, int, int]] = []
        total_len = 0
        for sid, text in docs:
            toks = self.bm25_tokenize(text)
            doc_len = len(toks)
            total_len += doc_len
            tf: Dict[str, int] = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            for tok, n in tf.items():
                postings.append((tok, sid, int(n), int(doc_len)))
            for tok in tf:
                term_df[tok] = term_df.get(tok, 0) + 1
        n_docs = len(docs)
        avgdl = float(total_len) / float(n_docs) if n_docs else 0.0
        self.conn.execute("INSERT OR REPLACE INTO bm25_meta(key, value) VALUES('N', ?)", (float(n_docs),))
        self.conn.execute("INSERT OR REPLACE INTO bm25_meta(key, value) VALUES('avgdl', ?)", (float(avgdl),))
        self.conn.executemany("INSERT OR REPLACE INTO bm25_terms(term, df) VALUES(?, ?)", list(term_df.items()))
        self.conn.executemany("INSERT OR REPLACE INTO bm25_postings(term, segment_id, tf, doc_len) VALUES(?,?,?,?)", postings)

    def bm25_rank(self, query_text: str, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        q_terms = list(dict.fromkeys(self.bm25_tokenize(query_text)))
        if not q_terms:
            return []
        meta_rows = self.conn.execute("SELECT key, value FROM bm25_meta WHERE key IN ('N','avgdl')").fetchall()
        meta = {str(k): float(v) for k, v in meta_rows}
        n_docs = float(meta.get("N", 0.0))
        avgdl = float(meta.get("avgdl", 0.0))
        if n_docs <= 0 or avgdl <= 0:
            return []
        scores: Dict[str, float] = {}
        for term in q_terms:
            row = self.conn.execute("SELECT df FROM bm25_terms WHERE term=?", (term,)).fetchone()
            if not row:
                continue
            df = float(row[0] or 0.0)
            if df <= 0:
                continue
            idf = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))
            for sid, tf, dl in self.conn.execute("SELECT segment_id, tf, doc_len FROM bm25_postings WHERE term=?", (term,)).fetchall():
                tfv = float(tf or 0.0)
                dlv = float(dl or 0.0)
                denom = tfv + k1 * (1.0 - b + b * (dlv / avgdl))
                if tfv > 0 and dlv > 0 and denom > 0:
                    scores[str(sid)] = scores.get(str(sid), 0.0) + idf * (tfv * (k1 + 1.0) / denom)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

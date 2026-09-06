# -*- coding: utf-8 -*-
"""
database.py
===========
Module lưu trữ & truy xuất dữ liệu tư thế (SQLite) cho Analytics Dashboard.

Module này chịu trách nhiệm DUY NHẤT về persistence (lưu trữ), tách bạch
khỏi UI/analytics trong app.py:

  - init_db()          : tạo bảng `posture_logs` nếu chưa tồn tại.
  - save_log(...)      : ghi một mẫu log tư thế.
  - get_session_data() : trả về toàn bộ log dưới dạng pandas DataFrame.
  - clear_all_data()   : xóa toàn bộ lịch sử tư thế.

Clean Code:
  - Mỗi hàm đúng MỘT trách nhiệm, chỉ dùng thư viện chuẩn `sqlite3`.
  - Mọi connection đều được đóng một cách chắc chắn (try/finally).
  - Dữ liệu trả về theo cấu trúc chuẩn nên Analytics không phụ thuộc schema.
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd

from config import DATABASE_PATH

# Trạng thái tư thế (khớp với giá trị lưu trong cột `status`).
STATUS_GOOD = 'Good'
STATUS_BAD = 'Bad'


def init_db(db_path: str = DATABASE_PATH) -> None:
    """Tạo bảng `posture_logs` nếu chưa tồn tại (idempotent).

    Cột:
      - id         : khóa chính tự tăng.
      - timestamp  : thời điểm ghi log (ISO-8601).
      - status     : 'Good' hoặc 'Bad'.
      - score      : điểm tư thế 0-100.
      - error_type : loại lỗi (Turtle Neck, Lateral Tilt, Shoulder Imbalance, 'None').
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS posture_logs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT NOT NULL,
                status     TEXT NOT NULL CHECK(status IN ('Good', 'Bad')),
                score      INTEGER NOT NULL,
                error_type TEXT NOT NULL DEFAULT 'None'
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _current_timestamp() -> str:
    """Trả về timestamp ISO-8601 (dễ đọc và tương thích pd.to_datetime)."""
    return datetime.now().isoformat(timespec='seconds')


def save_log(
    status: str,
    score: int,
    error_type: str = 'None',
    timestamp: Optional[str] = None,
    db_path: str = DATABASE_PATH,
) -> None:
    """Ghi một mẫu log tư thế vào bảng `posture_logs`.

    Tham số:
      - status     : 'Good' hoặc 'Bad'.
      - score      : điểm tư thế 0-100.
      - error_type : tên lỗi, nhiều lỗi nối bằng " + ", mặc định 'None'.
      - timestamp  : (tùy chọn) chuỗi ISO-8601; nếu bỏ qua sẽ lấy thời điểm gọi.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            'INSERT INTO posture_logs (timestamp, status, score, error_type) '
            'VALUES (?, ?, ?, ?)',
            (timestamp or _current_timestamp(), status, int(score), error_type),
        )
        conn.commit()
    finally:
        conn.close()


def get_session_data(db_path: str = DATABASE_PATH) -> pd.DataFrame:
    """Trả về toàn bộ log dưới dạng pandas DataFrame, sắp xếp theo thời gian.

    Các cột: id, timestamp, status, score, error_type.
      - timestamp  được chuyển về kiểu datetime để vẽ biểu đồ thời gian.
      - Trả về DataFrame rỗng (đúng schema) nếu database chưa tồn tại/chưa có dữ liệu.
    """
    columns = ['id', 'timestamp', 'status', 'score', 'error_type']
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=columns)

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            'SELECT id, timestamp, status, score, error_type '
            'FROM posture_logs ORDER BY timestamp ASC',
            conn,
        )
    finally:
        conn.close()

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def clear_all_data(db_path: str = DATABASE_PATH) -> None:
    """Xóa toàn bộ lịch sử tư thế trong bảng `posture_logs`."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute('DELETE FROM posture_logs')
        conn.commit()
    finally:
        conn.close()
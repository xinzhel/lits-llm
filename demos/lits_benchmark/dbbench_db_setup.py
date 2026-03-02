"""MySQL Docker container management for DBBench.

Provides helpers to start/stop a MySQL 8 container and initialize it
with DBBench table data.  Designed to be called from the
``@register_resource("dbbench")`` loader or standalone scripts.

Usage::

    from demos.lits_benchmark.dbbench_db_setup import (
        start_mysql_container, stop_mysql_container, init_database,
    )
    from demos.lits_benchmark.dbbench import load_dbbench

    uri = start_mysql_container()
    entries = load_dbbench(database="wikisql")
    init_database(uri, entries, database="dbbench_wikisql")
    # ... run experiments ...
    stop_mysql_container()
"""

import logging
import subprocess
import time
from typing import Dict, List, Optional

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CONTAINER_NAME = "dbbench_mysql"
MYSQL_IMAGE = "mysql:8"
MYSQL_ROOT_PASSWORD = "password"
MYSQL_PORT = 3307  # avoid conflict with local MySQL on 3306


def _run(cmd: List[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run a shell command, logging it first."""
    logger.debug("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=check, **kwargs)


# ---------------------------------------------------------------------------
# Container lifecycle
# ---------------------------------------------------------------------------

def start_mysql_container(
    port: int = MYSQL_PORT,
    password: str = MYSQL_ROOT_PASSWORD,
    container_name: str = CONTAINER_NAME,
) -> str:
    """Start a MySQL 8 Docker container (or reuse if already running).

    Args:
        port: Host port to map to container's 3306.
        password: MySQL root password.
        container_name: Docker container name.

    Returns:
        SQLAlchemy-compatible URI: ``mysql+pymysql://root:<pw>@localhost:<port>``.
    """
    # Check if container already exists
    result = _run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
        check=False,
    )
    if result.returncode == 0:
        running = result.stdout.strip() == "true"
        if running:
            logger.info("Container '%s' already running on port %d", container_name, port)
            return f"mysql+pymysql://root:{password}@localhost:{port}"
        # exists but stopped — remove and recreate
        _run(["docker", "rm", "-f", container_name], check=False)

    logger.info("Starting MySQL 8 container '%s' on port %d ...", container_name, port)
    _run([
        "docker", "run", "-d",
        "--name", container_name,
        "-e", f"MYSQL_ROOT_PASSWORD={password}",
        "-p", f"{port}:3306",
        MYSQL_IMAGE,
    ])

    # Wait for MySQL to be ready
    uri = f"mysql+pymysql://root:{password}@localhost:{port}"
    _wait_for_mysql(uri, timeout=60)
    return uri


def stop_mysql_container(container_name: str = CONTAINER_NAME) -> None:
    """Stop and remove the MySQL Docker container."""
    logger.info("Stopping container '%s' ...", container_name)
    _run(["docker", "rm", "-f", container_name], check=False)


def _wait_for_mysql(uri: str, timeout: int = 60) -> None:
    """Block until MySQL accepts connections or timeout is reached."""
    deadline = time.time() + timeout
    engine = create_engine(uri)
    while time.time() < deadline:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("MySQL is ready.")
            engine.dispose()
            return
        except Exception:
            time.sleep(1)
    engine.dispose()
    raise TimeoutError(f"MySQL not ready after {timeout}s at {uri}")


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------

def init_database(
    base_uri: str,
    entries: List[Dict],
    database: str = "dbbench",
) -> str:
    """Create a MySQL database and populate it with DBBench table data.

    Args:
        base_uri: Base MySQL URI without database name
            (e.g. ``mysql+pymysql://root:password@localhost:3307``).
        entries: List of example dicts from ``load_dbbench()``, each with
            a ``table`` field.
        database: Name of the MySQL database to create.

    Returns:
        Full URI including the database name, ready for ``SQLDBClient``.
    """
    from .dbbench import build_init_sql

    engine = create_engine(base_uri)
    with engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{database}`"))
        conn.commit()
    engine.dispose()

    full_uri = f"{base_uri}/{database}"
    engine = create_engine(full_uri)

    # Deduplicate tables: multiple entries may share the same table_name
    seen_tables = set()
    with engine.connect() as conn:
        for entry in entries:
            stmts = build_init_sql(entry)
            for stmt in stmts:
                # Skip CREATE/INSERT for tables we've already initialized
                # (entries in the same database group share tables)
                table_name = _extract_table_name(stmt)
                if table_name and stmt.upper().startswith("CREATE"):
                    if table_name in seen_tables:
                        continue
                    seen_tables.add(table_name)
                elif table_name and stmt.upper().startswith("INSERT"):
                    if table_name in seen_tables and _table_has_data(conn, table_name):
                        continue

                conn.execute(text(stmt))
            conn.commit()

    engine.dispose()
    logger.info(
        "Initialized database '%s' with %d tables from %d entries",
        database, len(seen_tables), len(entries),
    )
    return full_uri


def _extract_table_name(sql: str) -> Optional[str]:
    """Extract the table name from a CREATE TABLE or INSERT INTO statement."""
    sql_upper = sql.upper().strip()
    if sql_upper.startswith("CREATE TABLE"):
        # CREATE TABLE IF NOT EXISTS `name` (...)
        parts = sql.split("`")
        if len(parts) >= 2:
            return parts[1]
    elif sql_upper.startswith("INSERT INTO"):
        parts = sql.split("`")
        if len(parts) >= 2:
            return parts[1]
    return None


def _table_has_data(conn, table_name: str) -> bool:
    """Check if a table already has rows (to avoid duplicate inserts)."""
    try:
        result = conn.execute(text(f"SELECT 1 FROM `{table_name}` LIMIT 1"))
        return result.fetchone() is not None
    except Exception:
        return False

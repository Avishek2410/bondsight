"""Alembic runs this to apply the numbered scripts in versions/ (see db.migrate)."""
from alembic import context

from wirebond_inspector.db import metadata

conn = context.config.attributes["connection"]
context.configure(connection=conn, target_metadata=metadata, render_as_batch=True)
with context.begin_transaction():
    context.run_migrations()

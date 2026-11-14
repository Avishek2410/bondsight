"""Let a person take back a wrongly drawn "missed" box, and speed up the "not decided" filter.

decisions.ref_id: a "withdraw" decision points at the "missed" decision it cancels.
Index on decisions.finding_id: the Review page asks "which defects have no decision yet" often.

Revision ID: 0002
Revises: 0001
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("decisions") as b:
        b.add_column(sa.Column("ref_id", sa.Integer, nullable=True))
    op.create_index("decisions_finding", "decisions", ["finding_id"])
    op.create_index("decisions_ref", "decisions", ["ref_id"])


def downgrade() -> None:
    op.drop_index("decisions_ref", "decisions")
    op.drop_index("decisions_finding", "decisions")
    with op.batch_alter_table("decisions") as b:
        b.drop_column("ref_id")

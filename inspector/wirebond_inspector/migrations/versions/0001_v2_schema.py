"""Inspector v2 tables: units, shots, findings, alerts, decisions.

Revision ID: 0001
Revises:
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def _unit_fk():
    return sa.Column("unit_id", sa.Integer, sa.ForeignKey("units.id", ondelete="CASCADE"), nullable=False)


def _box():
    return [sa.Column(n, sa.Float) for n in ("x0", "y0", "x1", "y1")]


def upgrade() -> None:
    op.create_table(
        "units",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("unit_no", sa.Integer, nullable=True),
        sa.Column("ts", sa.Float, nullable=False),
        sa.Column("source", sa.Text, nullable=False, server_default="camera"),
        sa.Column("verdict", sa.Text, nullable=False),
        sa.Column("label", sa.Text, nullable=True),
    )
    op.create_index("units_verdict", "units", ["verdict"])
    op.create_index("units_source", "units", ["source"])
    op.create_table(
        "shots",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True), _unit_fk(),
        sa.Column("camera", sa.Text), sa.Column("view", sa.Text), sa.Column("source", sa.Text),
        sa.Column("image_path", sa.Text), sa.Column("error", sa.Text),
        sa.Column("anomaly", sa.Float), sa.Column("anomaly_flag", sa.Boolean), sa.Column("heat_path", sa.Text),
        sa.Column("width", sa.Integer), sa.Column("height", sa.Integer), sa.Column("hints", sa.Text),
    )
    op.create_table(
        "findings",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True), _unit_fk(),
        sa.Column("camera", sa.Text), sa.Column("cls", sa.Text), sa.Column("score", sa.Float),
        *_box(), sa.Column("position", sa.Text),
    )
    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True), _unit_fk(),
        sa.Column("position", sa.Text), sa.Column("cls", sa.Text),
        sa.Column("count", sa.Integer), sa.Column("window", sa.Integer),
    )
    op.create_table(
        "decisions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True), _unit_fk(),
        sa.Column("finding_id", sa.Integer, sa.ForeignKey("findings.id", ondelete="CASCADE"), nullable=True),
        sa.Column("verdict", sa.Text, nullable=False),
        sa.Column("camera", sa.Text), sa.Column("cls", sa.Text), *_box(),
        sa.Column("operator", sa.Text, nullable=False),
        sa.Column("ts", sa.Float, nullable=False),
    )
    for t in ("shots", "findings", "alerts", "decisions"):
        op.create_index(f"{t}_unit", t, ["unit_id"])


def downgrade() -> None:
    for t in ("decisions", "alerts", "findings", "shots", "units"):
        op.drop_table(t)

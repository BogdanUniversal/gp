"""Added model

Revision ID: 2a21d8a94a4c
Revises: b75d79ad60e7
Create Date: 2025-05-31 15:27:30.301020

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '2a21d8a94a4c'
down_revision = 'b75d79ad60e7'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('Model',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('dataset_id', sa.UUID(), nullable=False),
    sa.Column('model_name', sa.TEXT(), nullable=False),
    sa.Column('parameters', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('resources_path', sa.TEXT(), nullable=True),
    sa.Column('train_date', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['dataset_id'], ['Dataset.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['User.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('Model')
    # ### end Alembic commands ###

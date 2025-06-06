"""Added dataset

Revision ID: b75d79ad60e7
Revises: 
Create Date: 2025-05-03 16:38:15.019642

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b75d79ad60e7'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('Dataset',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('file_name', sa.TEXT(), nullable=False),
    sa.Column('upload_date', sa.DateTime(), nullable=True),
    sa.Column('file_path', sa.TEXT(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['User.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('Dataset')
    # ### end Alembic commands ###

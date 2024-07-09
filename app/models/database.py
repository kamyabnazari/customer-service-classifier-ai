from sqlalchemy import MetaData, Table, Column, Integer, String, DateTime

metadata = MetaData()

query_history = Table(
    "query_history",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("question", String, nullable=False),
    Column("answer", String, nullable=False),
    Column("created_at", DateTime, nullable=False),
)

def create_tables(engine):
    metadata.create_all(bind=engine)
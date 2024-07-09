from sqlalchemy import MetaData, Table, Column, Integer, String, DateTime, Text, ForeignKey

metadata = MetaData()

predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(255), nullable=False),
    Column("request", Text, nullable=False),
    Column("response", Text, nullable=False),
    Column("created_at", DateTime, nullable=False),
    Column("status", String(50), nullable=False)
)

def create_tables(engine):
    metadata.create_all(bind=engine)
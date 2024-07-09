from pydantic import BaseModel
from datetime import datetime

class Prediction(BaseModel):
    id: int
    name: str
    request: str
    created_at: datetime
    response: str
    status: str

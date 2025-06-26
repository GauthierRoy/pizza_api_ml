# models.py
import datetime
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime

# Import the Base class from your database setup
from .database import Base


# we define your table model here, inheriting from the imported Base
class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.UTC))
    raw_request = Column(JSON)
    prediction_label = Column(String)
    prediction_value = Column(Integer)
    probability_of_success = Column(Float)

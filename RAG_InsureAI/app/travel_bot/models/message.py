from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from travel_bot.core.database import Base

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    role = Column(String)  # 'user', 'assistant', or 'system'
    content = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
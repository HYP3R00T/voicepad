from pydantic import BaseModel


class Config(BaseModel):
    timeout: int = 30

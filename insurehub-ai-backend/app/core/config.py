from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "InsureHub AI Backend"
    DATABASE_URL: str = "sqlite:///./insurehub_local.db" 
    OPENAI_API_KEY: str = "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"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
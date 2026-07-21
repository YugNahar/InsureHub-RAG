from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "InsureHub Travel Bot"
    # Persisted volume path (matches the rest of this app's /root/.insurehub
    # data dir) rather than a relative path — a relative sqlite:///./x.db
    # would resolve against the container's cwd (/app/app, bind-mounted from
    # the host source tree), writing a stray db file into the repo instead
    # of somewhere actually persisted across container recreation.
    DATABASE_URL: str = "sqlite:////root/.insurehub/travel_bot.db"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
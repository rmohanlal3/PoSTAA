"""
Application configuration using Pydantic settings
"""
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "Motivational Content Generator"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # API settings
    API_V1_PREFIX: str = "/api/v1"
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/motivational_db"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    
    # Google Cloud
    GCP_PROJECT_ID: str = "your-gcp-project-id"
    GCP_REGION: str = "us-central1"
    GCS_BUCKET_NAME: str = "motivational-content-bucket"
    GCS_VIDEO_BUCKET: str = "motivational-videos-bucket"
    
    # Firestore
    FIRESTORE_COLLECTION_USERS: str = "users"
    FIRESTORE_COLLECTION_CLIPS: str = "clips"
    FIRESTORE_COLLECTION_PREFERENCES: str = "preferences"
    
    # Pub/Sub
    PUBSUB_TOPIC_CONTENT_GENERATION: str = "content-generation"
    PUBSUB_TOPIC_VIDEO_PROCESSING: str = "video-processing"
    PUBSUB_SUBSCRIPTION_CONTENT: str = "content-generation-sub"
    
    # NVIDIA Riva settings
    RIVA_API_URL: str = "http://riva-service:50051"
    RIVA_VOICE_NAME: str = "English-US.Female-1"
    
    # NeMo settings
    NEMO_API_URL: str = "http://nemo-service:8080"
    NEMO_MODEL_NAME: str = "llama-2-13b"
    
    # Video generation
    VIDEO_GENERATION_API: str = "d-id"  # Options: d-id, synthesia, custom
    DID_API_KEY: str = ""
    DID_API_URL: str = "https://api.d-id.com"
    
    # Storage
    UPLOAD_MAX_SIZE: int = 100 * 1024 * 1024  # 100MB
    VIDEO_MAX_DURATION: int = 120  # seconds
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8081",
        "http://localhost:19006"
    ]
    
    # Redis (for caching)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_URL: str = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Feature flags
    ENABLE_PERSONALIZATION: bool = True
    ENABLE_ANALYTICS: bool = True
    ENABLE_PUSH_NOTIFICATIONS: bool = True
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_DAY: int = 1000
    
    # Content generation
    DAILY_CLIP_GENERATION_TIME: str = "06:00"  # UTC
    MAX_CONCURRENT_GENERATIONS: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

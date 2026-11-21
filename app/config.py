import os 
import certifi 
from dotenv import load_dotenv

# --- AUTOMATIC SSL FIX FOR CONDA/PYTHON ENVIRONMENTS ---
ssl_cert = os.environ.get('SSL_CERT_FILE')
if ssl_cert and not os.path.exists(ssl_cert):
    os.environ['SSL_CERT_FILE'] = certifi.where()
elif not ssl_cert:
    os.environ['SSL_CERT_FILE'] = certifi.where()
# --------------------------------------------------------

load_dotenv()
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    aws_access_key_id: str = ''
    aws_secret_access_key: str = ''
    aws_region: str = 'ap-southeast-2'
    bedrock_embed_model: str = 'amazon.titan-embed-text-v2:0'
    embed_dim: int = 1024

    pinecone_api_key: str = ''
    pinecone_index: str = 'healthguide-bedrock-1024'
    tavily_api_key: str = ''

    nvidia_api_key: str = ''
    nvidia_base_url: str = 'https://integrate.api.nvidia.com/v1'
    nvidia_chat_model: str = 'meta/muse-glimmer-30b'
    nvidia_vision_model: str = 'meta/muse-glimmer-30b'

    api_token: str = ''
    allowed_origins: str = 'http://localhost:8000,http://127.0.0.1:8000'
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


settings = Settings()

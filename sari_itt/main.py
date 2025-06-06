from fastapi import FastAPI
from app.api.routes import detection
from app.core.config import settings
from app.core.logging_config import logger

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Register routes
app.include_router(
    detection.router,
    prefix=settings.API_V1_STR,
    tags=["detection"]
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Cancer Cell Detection API") 
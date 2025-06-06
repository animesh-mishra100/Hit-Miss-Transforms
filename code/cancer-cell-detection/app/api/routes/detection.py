from fastapi import APIRouter, UploadFile, File
from ..controllers.detection_controller import DetectionController
from ...core.logging_config import logger

router = APIRouter()

@router.post("/detect")
async def detect_cancer_cells(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")
    return await DetectionController.process_image(file) 
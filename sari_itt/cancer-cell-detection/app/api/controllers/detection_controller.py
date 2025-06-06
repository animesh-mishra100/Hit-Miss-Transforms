from fastapi import UploadFile
import cv2
import numpy as np
from ...services.image_processing import ImageProcessor
from ...core.logging_config import logger

class DetectionController:
    @staticmethod
    async def process_image(file: UploadFile):
        try:
            # Read image file
            contents = await file.read()
            nparr = np.fromstring(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            # Process image
            processor = ImageProcessor()
            preprocessed = processor.preprocess_image(image)
            membranes = processor.detect_membranes(preprocessed)
            nuclei = processor.detect_nuclei(preprocessed)
            result = processor.filter_cancer_cells(membranes, nuclei)
            
            # Convert result to base64 for API response
            _, buffer = cv2.imencode('.png', result)
            img_str = base64.b64encode(buffer).decode()
            
            return {
                "status": "success",
                "result": img_str
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise 
#!/bin/bash

# Create project structure
mkdir -p cancer-cell-detection/app/{api/{routes,controllers},core,services,utils} \
         cancer-cell-detection/{tests,logs,static/uploads}

# Create Python package files
touch cancer-cell-detection/app/__init__.py
touch cancer-cell-detection/app/api/__init__.py
touch cancer-cell-detection/app/api/routes/__init__.py
touch cancer-cell-detection/app/api/controllers/__init__.py
touch cancer-cell-detection/app/core/__init__.py
touch cancer-cell-detection/app/services/__init__.py
touch cancer-cell-detection/app/utils/__init__.py
touch cancer-cell-detection/tests/__init__.py

# Create main application files
cat > cancer-cell-detection/requirements.txt << EOL
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
mpi4py==3.1.4
opencv-python==4.8.1.78
numpy==1.26.2
matplotlib==3.8.2
scipy==1.11.4
pydantic==2.5.2
pydantic-settings==2.1.0
python-dotenv==1.0.0
EOL

# Create .gitignore
cat > cancer-cell-detection/.gitignore << EOL
__pycache__/
*.py[cod]
*$py.class
*.so
.env
.venv
env/
venv/
ENV/
logs/
static/uploads/*
!static/uploads/.gitkeep
.DS_Store
EOL

# Create .env file
cat > cancer-cell-detection/.env << EOL
API_V1_STR=/api/v1
PROJECT_NAME="Cancer Cell Detection API"
EOL

# Make setup.sh executable
chmod +x setup.sh

echo "Project structure created successfully!"
echo "To get started:"
echo "1. cd cancer-cell-detection"
echo "2. python -m venv venv"
echo "3. source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
echo "4. pip install -r requirements.txt"
echo "5. uvicorn main:app --reload"
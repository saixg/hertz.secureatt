@echo off
echo Installing SecureAttend Backend Dependencies...
echo ================================================

echo.
echo Step 1: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 2: Installing core dependencies...
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install pydantic==2.5.0
pip install sqlmodel==0.0.14
pip install python-multipart==0.0.6
pip install websockets==12.0
pip install Pillow==10.1.0
pip install aiofiles==23.2.1

echo.
echo Step 3: Installing ML dependencies...
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install face-recognition==1.3.0

echo.
echo Step 4: Installing optional dependencies...
pip install alembic==1.12.1
pip install python-jose[cryptography]==3.3.0
pip install passlib[bcrypt]==1.7.4

echo.
echo ================================================
echo Installation completed!
echo.
echo You can now run: python main.py
echo.
pause
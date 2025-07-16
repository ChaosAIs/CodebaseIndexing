@echo off
echo Setting up fresh virtual environment...

if exist venv rmdir /s /q venv
python -m venv venv
call venv\Scripts\activate.bat

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo Virtual environment setup complete!
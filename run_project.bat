@echo off
echo Installing required packages...
python -m pip install -r requirements.txt
echo.
echo Checking the dataset...
python inspect_data.py
echo.
echo If the dataset check was successful, run train.py from the VS Code terminal.
pause

@echo off
python -m venv stock_bot_env
call stock_bot_env\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
echo Виртуальное окружение настроено. Для активации выполните: stock_bot_env\Scripts\activate.bat
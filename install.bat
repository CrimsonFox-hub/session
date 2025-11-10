@echo off
echo Установка телеграм-бота для анализа акций...
echo.

REM Проверяем наличие Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  Ошибка: Python не установлен или не добавлен в PATH
    pause
    exit /b 1
)

echo ✓ Python обнаружен

REM Создаем виртуальное окружение
echo Создание виртуального окружения...
python -m venv venv

REM Активируем виртуальное окружение
echo Активация виртуального окружения...
call venv\Scripts\activate.bat

REM Устанавливаем зависимости
echo Установка зависимостей...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo  Установка завершена!
echo.
echo Следующие шаги:
echo 1. Получите токен бота у @BotFather в Telegram
echo 2. Откройте файл .env и замените 'your_actual_bot_token_here' на ваш токен
echo 3. Запустите бота командой: run.bat
echo.
pause
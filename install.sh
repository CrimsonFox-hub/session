#!/bin/bash
echo "Установка телеграм-бота для анализа акций..."
echo

# Проверяем наличие Python
if ! command -v python3 &> /dev/null; then
    echo " Ошибка: Python3 не установлен"
    exit 1
fi

echo "✓ Python обнаружен"

# Создаем виртуальное окружение
echo "Создание виртуального окружения..."
python3 -m venv venv

# Активируем виртуальное окружение
echo "Активация виртуального окружения..."
source venv/bin/activate

# Устанавливаем зависимости
echo "Установка зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt

echo
echo " Установка завершена!"
echo
echo "Следующие шаги:"
echo "1. Получите токен бота у @BotFather в Telegram"
echo "2. Откройте файл .env и замените 'your_actual_bot_token_here' на ваш токен"
echo "3. Запустите бота командой: ./run.sh"
echo
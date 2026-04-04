import asyncio

from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ContentType, ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Document, Message

from config import settings
from application.services import FileProcessingService

MAX_TELEGRAM_MESSAGE_LEN = 4000

session = AiohttpSession(proxy=settings.proxy_url)
bot = Bot(session=session, token=settings.BOT_TOKEN)
dp = Dispatcher()
service = FileProcessingService()


async def send_html_in_chunks(
    message: Message,
    text: str,
    max_len: int = MAX_TELEGRAM_MESSAGE_LEN,
) -> None:
    """
    Отправляет длинный HTML-текст частями, чтобы не превысить лимит Telegram.
    Делит текст по строкам, а если строка слишком длинная — дополнительно режет её.
    """
    if len(text) <= max_len:
        await message.answer(text, parse_mode=ParseMode.HTML)
        return

    parts: list[str] = []
    current_part: list[str] = []
    current_len = 0

    for line in text.splitlines(keepends=True):
        line_len = len(line)

        if line_len > max_len:
            if current_part:
                parts.append("".join(current_part).rstrip())
                current_part = []
                current_len = 0

            start = 0
            while start < line_len:
                chunk = line[start : start + max_len]
                parts.append(chunk.rstrip())
                start += max_len

            continue

        if current_len + line_len > max_len:
            parts.append("".join(current_part).rstrip())
            current_part = [line]
            current_len = line_len
        else:
            current_part.append(line)
            current_len += line_len

    if current_part:
        parts.append("".join(current_part).rstrip())

    for part in parts:
        if part:
            await message.answer(part, parse_mode=ParseMode.HTML)


@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "Привет!\n"
        "Пожалуйста, загрузите файл с выгрузкой банковских операций.\n"
        "Поддерживаемые форматы: .xlsx или .csv"
    )


@dp.message(F.content_type == ContentType.DOCUMENT)
async def handle_document(message: Message):
    document: Document = message.document
    if not service.is_supported_file(document.file_name):
        await message.answer(
            "Неверный формат файла.\n"
            "Пожалуйста, загрузите файл в формате .xlsx или .csv"
        )
        return

    await message.answer("Файл обрабатывается. Подождите")

    tg_file = await bot.get_file(document.file_id)
    file_bytes = await bot.download_file(tg_file.file_path)

    try:
        result_text: list[str] = await service.process_file(
            filename=document.file_name, file_content=file_bytes.read()
        )
    except Exception as e:
        print(e)
        await message.answer("Ошибка при обработке файла. Попробуйте позже.")
        return

    for text in result_text:
        await send_html_in_chunks(message, text)

    await message.answer("Вы можете загрузить новый файл с банковской выгрузкой")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

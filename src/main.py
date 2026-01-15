import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message, Document
from aiogram.enums import ContentType, ParseMode

from config import settings
from application.services import FileProcessingService

bot = Bot(token=settings.BOT_TOKEN)
dp = Dispatcher()
service = FileProcessingService()


@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "Привет!\n"
        "Пожалуйста, загрузите файл с выгрузкой банковских операций.\n"
        "Поддерживаемые форматы: .xlsx или .csv"
    )
    print(1)


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
            filename=document.file_name,
            file_content=file_bytes.read()
        )
    except Exception as e:
        print(e)
        await message.answer("Ошибка при обработке файла. Попробуйте позже.")
        return

    for text in result_text:
        await message.answer(text, parse_mode=ParseMode.HTML)
    await message.answer(
        "Вы можете загрузить новый файл с банковской выгрузкой"
    )


async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
import aiohttp
from src.config import settings
from src.domain.file import BankStatementFile

class BackendClient:
    async def send_file(self, bank_file: BankStatementFile) -> str:
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field(
                name="file",
                value=bank_file.content,
                filename=bank_file.filename,
                content_type="application/octet-stream",
            )

            async with session.post(settings.BACKEND_URL, data=data) as resp:
                resp.raise_for_status()
                json_resp = await resp.json()

        # предполагаем, что бэкенд вернёт поле "message"
        return json_resp.get("message", "Файл успешно обработан")
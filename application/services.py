from domain.file import BankStatementFile, SUPPORTED_EXTENSIONS
from infrastructure.backend_client import BackendClient

class FileProcessingService:
    def __init__(self):
        self.backend = BackendClient()

    def is_supported_file(self, filename: str) -> bool:
        if not filename or '.' not in filename:
            return False
        return filename.split('.')[-1].lower() in SUPPORTED_EXTENSIONS

    async def process_file(self, filename: str, file_content: bytes) -> str:
        bank_file = BankStatementFile(
            filename=filename,
            content=file_content
        )

        if not bank_file.is_supported():
            raise ValueError("Unsupported file format")

        response = await self.backend.send_file(bank_file)
        return response
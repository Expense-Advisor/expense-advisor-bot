import asyncio

from src.domain.file import BankStatementFile, SUPPORTED_EXTENSIONS
from src.modules.financial_intelligence.pipeline.pipeline import FinancialIntelligencePipeline


class FileProcessingService:
    def __init__(self):
        pass

    def is_supported_file(self, filename: str) -> bool:
        if not filename or '.' not in filename:
            return False
        return filename.split('.')[-1].lower() in SUPPORTED_EXTENSIONS

    async def process_file(self, filename: str, file_content: bytes) -> list[str]:
        bank_file = BankStatementFile(
            filename=filename,
            content=file_content
        )

        if not bank_file.is_supported():
            raise ValueError("Unsupported file format")

        pipeline = FinancialIntelligencePipeline(content=bank_file.content)
        response: list[str] = await asyncio.to_thread(pipeline.run)

        return response
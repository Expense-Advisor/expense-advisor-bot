from dataclasses import dataclass

SUPPORTED_EXTENSIONS = {"csv", "xlsx"}

@dataclass
class BankStatementFile:
    filename: str
    content: bytes

    @property
    def extension(self) -> str:
        return self.filename.split('.')[-1].lower()

    def is_supported(self) -> bool:
        return self.extension in SUPPORTED_EXTENSIONS
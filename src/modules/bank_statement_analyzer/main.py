from src.modules.bank_statement_analyzer.pipeline import AnalyzerPipeline

if __name__ == '__main__':
    pipeline = AnalyzerPipeline('assets/12 июля 2025 - 11 января 2026.xlsx')
    print(pipeline.run())

from src.modules.financial_intelligence.pipeline.pipeline import FinancialIntelligencePipeline

if __name__ == '__main__':
    pipeline = FinancialIntelligencePipeline('assets/12 июля 2025 - 11 января 2026.xlsx')
    print(pipeline.run())

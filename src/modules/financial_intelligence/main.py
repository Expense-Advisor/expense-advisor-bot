from src.modules.financial_intelligence.pipeline.pipeline import (
    FinancialIntelligencePipeline,
)


def render_report(pages: list[str]):
    for page in pages:
        lines = page.split("\n")

        title = lines[0].replace("<b>", "").replace("</b>", "")
        print(f"\n{title}")
        print("-" * len(title))

        for line in lines[1:]:
            print(line)

        print("\n")


if __name__ == "__main__":
    with open("assets/12 июля 2025 - 11 января 2026.xlsx", "rb") as f:
        content = f.read()
    pipeline = FinancialIntelligencePipeline(content)
    print(render_report(pipeline.run()))

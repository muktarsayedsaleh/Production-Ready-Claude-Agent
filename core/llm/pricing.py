from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / 1_000_000


# Anthropic first-party API rates. Keep in sync with https://www.anthropic.com/pricing
PRICING: dict[str, ModelPricing] = {
    "claude-fable-5-1": ModelPricing(input_per_mtok=10.00, output_per_mtok=50.00),
    "claude-opus-5-5": ModelPricing(input_per_mtok=4.00, output_per_mtok=20.00),
    "claude-opus-5": ModelPricing(input_per_mtok=5.00, output_per_mtok=25.00),
    "claude-sonnet-5-5": ModelPricing(input_per_mtok=2.00, output_per_mtok=10.00),
    "claude-sonnet-5": ModelPricing(input_per_mtok=2.00, output_per_mtok=10.00),
    "claude-haiku-4-5": ModelPricing(input_per_mtok=1.00, output_per_mtok=5.00),
}


def pricing_for(model: str) -> ModelPricing:
    try:
        return PRICING[model]
    except KeyError:
        raise ValueError(f"No pricing configured for model {model!r}") from None

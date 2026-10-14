"""Invoice calculations."""

from dataclasses import dataclass

TAX_RATE = 0.08
BULK_DISCOUNT_THRESHOLD = 100


@dataclass
class LineItem:
    description: str
    unit_price: float
    quantity: int


def subtotal(items: list[LineItem]) -> float:
    """Sum price times quantity for every line item on the invoice."""
    return sum(i.unit_price * i.quantity for i in items)


def bulk_discount(items: list[LineItem]) -> float:
    """Apply a ten percent discount when the total quantity reaches the bulk threshold."""
    quantity = sum(i.quantity for i in items)
    return subtotal(items) * 0.10 if quantity >= BULK_DISCOUNT_THRESHOLD else 0.0


def invoice_total(items: list[LineItem]) -> float:
    """Compute the final amount due: subtotal minus discount, plus sales tax."""
    net = subtotal(items) - bulk_discount(items)
    return round(net * (1 + TAX_RATE), 2)

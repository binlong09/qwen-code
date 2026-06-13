from pricing import apply_discount


def cart_total(items, discount_percent=0):
    subtotal = sum(price for _, price in items)
    return apply_discount(subtotal, discount_percent)

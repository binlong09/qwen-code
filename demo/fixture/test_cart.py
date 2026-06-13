from cart import cart_total


def test_loyalty_discount():
    items = [("book", 30.0), ("pen", 20.0)]  # subtotal 50.00
    # a 10% loyalty discount should leave a 45.00 total
    assert cart_total(items, 10) == 45.0


if __name__ == "__main__":
    test_loyalty_discount()
    print("ok")

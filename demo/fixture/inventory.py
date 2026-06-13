def in_stock(sku, warehouse):
    return warehouse.get(sku, 0) > 0

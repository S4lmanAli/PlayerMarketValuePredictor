def format_value(val):
    if val >=1_000_000:
        return f"{val/1_000_000:.2f}M"
    elif val >=1_000:
        return f"{val/1_000:.0f}K"
    else: return str(round(val))
def get_total_line(path : str) -> int:
    total_line = 0
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            total_line += 1
    return total_line

from datetime import datetime

str2date = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M').timestamp()

def timestamp2date(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp)

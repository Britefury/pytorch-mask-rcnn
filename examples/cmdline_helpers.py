

def comma_separated_values(string_value, value_fn):
    vals = [x.strip() for x in string_value.split(',')]
    return [value_fn(x) for x in vals if x != '']

def colon_separated_values(string_value, value_fn):
    vals = [x.strip() for x in string_value.split(':')]
    return [value_fn(x) for x in vals if x != '']

def colon_separated_range(x):
    lower = upper = None
    if x != '':
        if ':' not in x:
            print('Invalid range format; should be float:float')
            return
        l, _, h = x.partition(':')
        try:
            lower = float(l)
            upper = float(h)
        except ValueError:
            print('Invalid range format; should be float:float')
            return

    return lower, upper

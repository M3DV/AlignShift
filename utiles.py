from torch._six import container_abcs
def as_triple(x, d_value=1): 
    if isinstance(x, container_abcs.Iterable):
        x = list(x)
        if len(x)==2:
            x = [d_value] + x
        return x
    else:
        return [d_value] + [x] * 2

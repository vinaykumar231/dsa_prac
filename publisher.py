def is_armstrong(n):
    new=str(n)
    power=len(new)
    total= 0

    for d in new:
        total +=int(d)**power

    return True



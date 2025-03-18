def maximum(list):

    n=len(list)
    max=list[0]
    for i in range(n):
        if list[i] > max:
            max=list[i]

    return max

print(maximum([2,44,5,5,22,434,666]))
    
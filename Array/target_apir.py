def target_pair(list, value):
    n=len(list)
    pair=[]
    for i in range(n-1):
        for j in range(i+1, n):
            if list[i]+ list[j]==value:
                pair.append((list[i], list[j]))

    return pair if pair else None

print(target_pair([2,4,5,7,75,5,3,5,7],9))

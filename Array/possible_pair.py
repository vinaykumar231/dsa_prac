def possible_pair(list):
    n=len(list)
    pair=[]
    for i in range(n-1):
        for j in range(i+1, n):
            list_pair=(list[i], list[j])
            pair.append(list_pair)
    return pair
print(possible_pair([2,4,5,7,8,9]))

def delete_dupicate(list):
    n=len(list)
    seen=[]
    for i in range(n):
        if list[i] not in seen:
            seen.append(list[i])

    return seen
print(delete_dupicate([3,4,5,6,3,9]))
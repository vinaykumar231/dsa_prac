def sort(list):
    n=len(list)
    for i in range(n):
        for j in range(n-1-i):
            if list[j] > list[j+1]:
                list[j] , list[j+1]= list[j+1],list[j]

    return list

print(sort([2,4,1,4,6,18]))



# bubble sort
def dupliacates(list):
    n=len(list)
    duplcate=0
    for i in range(n):
        for j in range(i+1,n):
            if list[i]==list[j]:
                duplcate +=1


    return duplcate

print(dupliacates([1,43,5,1,5]))

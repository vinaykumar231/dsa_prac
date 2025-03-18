def sumtarget(list, target):
    n=len(list)
    found_target=[]
    for i in range(n-1):
        for j in range(i+1, n):
            if list[i]+list[j]==target:
                found_target.append(([i,j]))

    return found_target

print(sumtarget([500,600,720,180,340,400],1100))

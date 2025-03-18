def Rotate_list_K_place(list, k):
    n=len(list)
    k= k%n
    print(k)

    rotatate_list=list[-k:]+ list[:-k]
    return rotatate_list
print(Rotate_list_K_place([2,3,4,5,7,8],4))
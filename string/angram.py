def sorting(str):
    str=list(str)
    n=len(str)
    for i in range(n-1):
        for j in range(n-1-i):
            if str[j] > str[j+1]:
                str[j], str[j+1] = str[j+1],  str[j]
    return "".join(str)



def is_anagram(str1, str2):
    sort_str1=sorting(str1)
    print(sort_str1)
    sort_str2=sorting(str2)
    print(sort_str2)
    if sort_str1 == sort_str2:
        return " anagram"
    else:
        return "not anagram"
   

print(is_anagram("slient", "listen"))

    




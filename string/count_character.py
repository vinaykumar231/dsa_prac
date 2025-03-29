def count_character(str ,value):
    count=0
    for s in str:
        if s == value:
            count +=1
    return count
    

print(count_character("hiiii","i"))



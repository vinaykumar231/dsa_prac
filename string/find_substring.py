def find_substring(str, value):
    seen=[]
    for s in str:
        seen.append(s)
    if value in seen:
        return True
    else:
        return False
    
print(find_substring("giii","m"))
    
print(find_substring("giii","g"))




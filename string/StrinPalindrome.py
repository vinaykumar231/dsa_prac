def StrinPalindrome(str):
    rev=""

    for s in str:
        rev= s + rev
    if rev != str:
        return {f"{str} is not string palindrome"}
    else:
        return {f"{str} is string palindrome"}
    

print(StrinPalindrome("madam"))
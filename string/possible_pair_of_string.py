def get_all_substrings(s):
    substrings = []
    for i in range(len(s)):  # Start index
        for j in range(i + 1, len(s) + 1):  # End index (exclusive)
            print(f"i={i}, j={j} → s[{i}:{j}] = {s[i:j]}")  # Debugging print
            substrings.append(s[i:j])  # Extract substring
    return substrings

# Example usage
print("Input: hiii")
print("All substrings:", get_all_substrings("hiii"))

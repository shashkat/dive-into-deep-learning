# print entries from dir function of an object, which have a particular substring
def DirSubsetToSubstring(obj, substr):
	entries_with_substr = []
	for entry in dir(obj):
		if substr in entry:
			entries_with_substr.append(entry)
	return entries_with_substr
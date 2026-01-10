# I believe convolutions can be helpful for textual data on in certain cases. For example, when
# we have data with character level meaning, like DNA data, then it can be useful to apply 
# convolutions of different sized kernels on it, and find out high level patterns from the 
# sequence data. It also helps that there are only 4 possible characters in DNA. 
# However, for language data, it might be more difficult, as any language is composed of words 
# of varying sizes, and the alphabet size is also generally quite big, hence how to "multiply"
# with a kernel is nontrivial. 
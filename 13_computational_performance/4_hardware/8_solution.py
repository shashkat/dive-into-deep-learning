# Not writing code for now, but the reason why it is generally faster to read forward through 
# memory than backward is how caching takes place. Generally, the cache line involves reading
# the data sequentially after the data being read in the memory. Hence, when we read data in 
# a forward manner, there are fewer cache misses, whereas when we read the data in a backward
# manner, there are more cache misses and hence more transfers of data from memory, which takes
# a bit longer than accessing data from cache.

# the difference between access times in reading forward through memory vs reading backward 
# through memory depends on the cpu and vendor because there are other factors at play here.
# For example, certain prefetchers can predict common access patterns like backwards easily
# and accordingly load the data in cache in a backwards fashion, reducing access times.
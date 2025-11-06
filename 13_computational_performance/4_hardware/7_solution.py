# The reason behind this is the fact that when we increase bit width, the architecture to handle 
# all the operations becomes more complex than simply linearly. The wiring complexity increases 
# and logic gates have to be scaled appropriately. Hence, the silicon requirements scale 
# quadratically with increase in bit-width. 

# NVIDIA added INT4 operations to their Turing GPUs because they found that with INT4 precision
# in inference, there was a significant speedup (50-60%), with minimal accuracy loss (1%). 
# Ref- https://developer.nvidia.com/blog/int4-for-ai-inference/
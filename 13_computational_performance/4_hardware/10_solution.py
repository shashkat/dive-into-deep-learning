# Not implementing checking packet overhead because I dont have ethernet connection as of now,
# and its more difficult to estimate packet overhead on wifi. 
# Difference between TCP (Transmission Control Protocol) and UDP (User Datagram Protocol) is 
# that UDP has less packet overhead (because it doesn't have error checking overhead), and hence
# is faster but less reliable. TCP has more overhead, but guarantees packet delivery, hence is 
# more reliable. Because of this difference, their use cases are different. Eg- TCP is better 
# for web browsing, email sending. UDP is more useful for video streaming etc, where speed is 
# more important and some packet losses are fine.
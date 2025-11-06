# The worst case scenario would be when the data we want to access is just behind where the 
# head (which actually reads the data) is in terms of angle on the disk, such that to access
# the data we would need to make almost a full rotation on the disk. In such a case, the time
# taken to access the data would be the time for one rotation, which would be 1/10000 mins, or 
# 6ms. This indicates that we can reduce the worst case time taken for data access by increasing
# the rotation speed of the disk. The rotation speed of the disk is limited by its radius, 
# because a larger radius leads to more centrifugal force on the edge of the disk, and more
# chances of it breaking. Hence, to allow for higher rotation speeds, the smaller diameter
# HDDs becoming more popular. Another benefit of smaller disk diameter is their lower power
# consumption at the same RPM.
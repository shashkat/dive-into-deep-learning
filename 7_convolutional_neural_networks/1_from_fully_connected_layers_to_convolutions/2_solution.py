# we might want to impose the assumptions of locality and translation invariance for audio if 
# say we are dealing with language audio. Words nearby a word may affect the meaning interpreted
# from them, but also, a word appearing in different faraway parts of the audio should refer to 
# the same thing.

# the convolution operations would be similar just in one dimension. H[a] = sum_i(V[i]*X[a]-i) + c

# yes we can treat it in the same way as computer vision by using a spectrogram of the audio. It
# is relevant as the locality and translation invariance (atleast along the horizontal direction)
# should hold for it as well, and its in an image form.
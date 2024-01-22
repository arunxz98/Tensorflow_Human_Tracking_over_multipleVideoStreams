from keras import backend as K


def cosine_distance(vecs):
    x, y = vecs

    x_norm = K.sum(x * x, axis=1)
    y_norm = K.sum(y * y, axis=1)
    dot_prod = K.batch_dot(x,y,axes=1)

    scalar_prod_norm = K.sqrt(x_norm) * K.sqrt(y_norm)
    scalar_prod_norm = K.expand_dims(scalar_prod_norm,axis=1)
    out = dot_prod / (scalar_prod_norm + K.epsilon())
    return out


def distance_output_shape(shapes):
    return (None,1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

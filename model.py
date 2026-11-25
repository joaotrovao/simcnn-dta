"""SimCNN-DTA architecture (Shim et al., 2021, Fig. 4)."""

from keras import Model, layers, ops, optimizers


class OuterProduct(layers.Layer):
    """k1_i (x) k2_j: (batch, n_drugs) x (batch, n_targets) -> (batch, n_drugs, n_targets, 1)."""

    def call(self, inputs):
        drug, target = inputs
        return ops.expand_dims(ops.einsum("bi,bj->bij", drug, target), -1)


def build_model(n_drugs, n_targets, pool_strides=1, learning_rate=1e-3):
    """Two conv + max-pool blocks, flatten, dropout, FC(128) ReLU, FC(1) linear.

    The paper specifies max-pooling of size 2 with stride 1; `pool_strides`
    allows stride 2 for inputs too large for that (see README).
    """
    drug = layers.Input((n_drugs,), name="drug_similarity")
    target = layers.Input((n_targets,), name="target_similarity")

    x = OuterProduct()([drug, target])
    x = layers.Conv2D(32, 5, activation="relu")(x)
    x = layers.MaxPooling2D(2, strides=pool_strides)(x)
    x = layers.Conv2D(18, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2, strides=pool_strides)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(1)(x)

    model = Model([drug, target], out, name="SimCNN_DTA")
    model.compile(optimizer=optimizers.Adam(learning_rate), loss="mse")
    return model

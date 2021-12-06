# SpatialTransformerFormer
Experiments using Spatial Transformers to achieve the principles of Transformers in a flexible and interpretable way.

The idea of the SpatialTransformerFormer is to leverage the attention capabilities of spatial transformers to allow a model to attain spatially meaningful latent states through a canvas-based self attention.

Moreover, to make the Spatial Transformers less erratic, a variational approach is used for deriving the affine transformations. The localization layer produces means and variances from which the affine transformation matric is drawn using the reparametrization trick.

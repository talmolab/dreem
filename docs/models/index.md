# DREEM Models

## User-facing models
There are two main model APIs users should interact with.

1. [`GlobalTrackingTransformer`](global_tracking_transformer.md) is the underlying model architecture we use for tracking. It is made up of a `VisualEncoder` and a `Transformer` `Encoder-Decoder`. Only more advanced users who have familiarity with python and pytorch should interact with this model. For others see below
2. [`GTRRunner`](gtr_runner.md) is a [`pytorch_lightning`](https://lightning.ai/) around the `GlobalTrackingTransformer`. It implements the basic routines you need for training, validation and testing. Most users will interact with this model.

## Model Parts
For advanced users who are interested in extending our model, we have modularized each component so that its easy to compose into your own custom model. The model parts are

1. [`VisualEncoder`](../reference/dreem/models/visual_encoder.md): A CNN backbone used for feature extraction.
2. [`Transformer`](../reference/dreem/models/transformer.md) which is composed of a:
    - [SpatioTemporal `Embedding`](../reference/dreem/models/embedding.md) which computes the spatial and temporal embedding of each detection.
    - [`TransformerEncoder`](../reference/dreem/models/transformer.md#dreem.models.transformer.TransformerEncoder): A stack of [`TransformerEncoderLayer`s](../reference/dreem/models/transformer.md#dreem.models.transformer.TransformerEncoderLayer)
    - [`TransformerDecoder`](../reference/dreem/models/transformer.md#dreem.models.transformer.TransformerDecoder): A stack of [`TransformerDecoderLayer`s](../reference/dreem/models/transformer.md#dreem.models.transformer.TransformerDecoderLayer)
3. An [`AttentionHead`](../reference/dreem/models/attention_head.md) which computes the association matrix from the transformer output.
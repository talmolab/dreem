# DREEM `io` module.

The `io` module contains classes for storing, manipulating and generating model inputs and outputs.

There are 6 main submodules:

1. [`Instance`](./instance.md) which represent detections. They are the main input into the model and contain 
    - GT Track ID
    - Predicted Track ID
    - the crop of the image
    - the location as a centroid `(x,y)`, bounding box `(y_1, x_1, y_2, x_2)` and (optionally) the pose `(1, n, 2)`
    - the features extracted by the visual encoder + spatiotemporal embedding
    - association score
2. [`Frame`](./frame.md) which stores the metadata for a single frame of a video. It contains:
    - The video index
    - The video name
    - The frame index
    - The list of [`Instance`s](./instance.md) that appear in that frame
    - The [`AssociationMatrix`](./asso_matrix.md) between the instances in that frame and all instances in the window
3. [`Config`](./config.md) which stores and parses the [configs](../configs/index.md) for training and/or
4. [`AssociationMatrix`](./asso_matrix.md) which stores the `GlobalTrackingTransformer` output. `AssociationMatrix[i,j]` represents the association score between the `i`th query instance and the `j`th instance in the window.
5. [`Track`](./track.md) which stores references to all instances belonging to the same trajectory
6. [`visualize`](./visualize.md) which contains util functions for generating videos annotated by the predicted tracks and scores.

import torch
from torchvision import functional as tvf


def pad_bbox(bbox, padding=16) -> torch.Tensor:
    """Pad bounding box coordinates.

    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format.
        padding: Padding to add to each side in pixels.

    Returns:
        Padded bounding box in [x1, y1, x2, y2] format.
    """
    x1, y1, x2, y2 = bbox
    y1, x1 = y1 - padding, x1 - padding
    y2, x2 = y2 + padding, x2 + padding
    return torch.Tensor([x1, y1, x2, y2])


def crop_bbox(img, bbox) -> torch.Tensor:
    """Crop an image to a bounding box.

    Args:
        img: Image as a tensor of shape (channels, height, width).
        bbox: Bounding box in [x1, y1, x2, y2] format.

    Returns:
        Cropped pixels as tensor of shape (channels, height, width).
    """
    # Crop to the bounding box.
    x1, y1, x2, y2 = bbox
    crop = tvf.crop(
        img,
        top=int(round(y1)),
        left=int(round(x1)),
        height=int(round(y2 - y1)),
        width=int(round(x2 - x1)),
    )

    return crop


def centroid_bbox(instance, anchors, crop_size) -> torch.Tensor:
    """Calculate bbox around instance centroid. This is useful for ensuring that
    crops are centered around each instance in the case of incorrect pose
    estimates

    Args:
        instance: a labeled instance in a frame
        anchors: indices of a given anchor point to use as the centroid
        crop_size: Integer specifying the crop height and width

    Returns:
        Bounding box in [x1, y1, x2, y2] format.
    """

    for anchor in anchors:
        cx, cy = instance[anchor].x, instance[anchor].y
        if not np.isnan(cx):
            break

    bbox = torch.Tensor(
        [
            -crop_size / 2 + cx,
            -crop_size / 2 + cy,
            crop_size / 2 + cx,
            crop_size / 2 + cy,
        ]
    )

    return bbox


def pose_bbox(instance, padding, im_shape) -> torch.Tensor:
    """Calculate bbox around instance pose.

    Args:
        instance: a labeled instance in a frame,
        padding: the amount to pad around the pose crop
        im_shape: the size of the original image in (w,h)

    Returns:
        Bounding box in [x1, y1, x2, y2] format.
    """

    w, h = im_shape

    points = torch.Tensor([[p.x, p.y] for p in instance.points])

    min_x = max(torch.nanmin(points[:, 0]) - padding, 0)
    min_y = max(torch.nanmin(points[:, 1]) - padding, 0)
    max_x = min(torch.nanmax(points[:, 0]) + padding, w)
    max_y = min(torch.nanmax(points[:, 1]) + padding, h)

    bbox = torch.Tensor([min_x, min_y, max_x, max_y])
    return bbox


def resize_and_pad(img, output_size):
    """Resize and pad an image to fit a square output size.

    Args:
        img: Image as a tensor of shape (channels, height, width).
        output_size: Integer size of height and width of output.

    Returns:
        The image zero padded to be of shape (channels, output_size, output_size).
    """
    # Figure out how to scale without breaking aspect ratio.
    img_height, img_width = img.shape[-2:]
    if img_width < img_height:  # taller
        crop_height = output_size
        scale = crop_height / img_height
        crop_width = int(img_width * scale)
    else:  # wider
        crop_width = output_size
        scale = crop_width / img_width
        crop_height = int(img_height * scale)

    # Scale without breaking aspect ratio.
    img = tvf.resize(img, size=[crop_height, crop_width])

    # Pad to square.
    img_height, img_width = img.shape[-2:]
    hp1 = int((output_size - img_width) / 2)
    vp1 = int((output_size - img_height) / 2)
    hp2 = output_size - (img_width + hp1)
    vp2 = output_size - (img_height + vp1)
    padding = (hp1, vp1, hp2, vp2)
    return tvf.pad(img, padding, 0, "constant")

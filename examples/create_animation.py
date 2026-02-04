import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display


def create_tracking_animation(
    video_path, metadata_df, fps=15, marker_size=15, max_frames=200, display_width=600
):
    """Create and display a tracking animation in the notebook."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create colormap for track IDs
    unique_ids = metadata_df["track_id"].unique()
    cmap = cm.get_cmap("tab10", len(unique_ids))
    id_to_color = {id_val: cmap(i) for i, id_val in enumerate(unique_ids)}

    # Setup figure
    fig_width = display_width / 100
    fig_height = fig_width * (height / width)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    frame_img = ax.imshow(np.zeros((height, width, 3), dtype=np.uint8))
    markers, texts = [], []

    frame_ids = sorted(metadata_df["frame_id"].unique())
    if max_frames and max_frames < len(frame_ids):
        frame_ids = frame_ids[:max_frames]
        print(f"Showing first {max_frames} frames")

    def update(frame_num):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            return []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_img.set_array(frame_rgb)

        for m in markers:
            m.remove()
        for t in texts:
            t.remove()
        markers.clear()
        texts.clear()

        frame_data = metadata_df[metadata_df["frame_id"] == frame_num]
        for _, row in frame_data.iterrows():
            x, y = row["centroid"]
            color = id_to_color[row["track_id"]]
            circle = Circle((x, y), marker_size, color=color, alpha=0.7)
            markers.append(ax.add_patch(circle))
            text = ax.text(
                x,
                y,
                str(row["track_id"]),
                color="white",
                fontsize=8,
                ha="center",
                va="center",
                fontweight="bold",
            )
            texts.append(text)

        frame_text = ax.text(
            10,
            20,
            f"Frame: {frame_num}",
            color="white",
            fontsize=8,
            backgroundcolor="black",
        )
        texts.append(frame_text)
        return [frame_img] + markers + texts

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    print(f"Creating animation with {len(frame_ids)} frames...")
    anim = FuncAnimation(fig, update, frames=frame_ids, blit=True)
    plt.close(fig)

    display(HTML(anim.to_html5_video()))
    cap.release()
    return anim

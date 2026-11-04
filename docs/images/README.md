# README illustrations

| Source file | README display file | Content |
| --- | --- | --- |
| `workflow.png` | `workflow-preview.jpg` | Head detection, smoothed ROI, eye and mouth models, normalization and event extraction (manuscript Figure 2) |
| `Loss.tif` | `loss.png` | Training and validation loss curves for four models (manuscript Figure 4) |
| `heatmap.tif` | `heatmap.png` | Mouth, head and eye validation confusion matrices |

Original figures are retained unchanged. The root README uses PNG versions of the TIFF figures for browser display and a resized JPEG workflow preview with a white background so that labels remain visible in dark themes. The original workflow PNG contains transparency.

The confusion-matrix image contains three panels, so its README caption describes all three rather than only mouth segmentation. Counts follow the displayed axes: true labels on rows, predicted labels on columns. Blank cells are not used to infer true negatives or frame-level accuracy.

To embed a figure from the root README:

```html
<p align="center">
  <a href="docs/images/loss.png"><img src="docs/images/loss.png" alt="Model training and validation losses" width="560"></a>
</p>
```

Display widths in the root README are 360 px for the portrait workflow, 560 px for training losses and 720 px for the wide confusion matrices. Only width is set, preserving each image's aspect ratio. Each figure links to its display file for a larger view; these HTML dimensions do not resample or modify the image files.

Use short filenames without spaces for additional images. PNG, JPEG, SVG and GIF are suitable for README display; convert TIFF sources to a browser-friendly format before embedding.

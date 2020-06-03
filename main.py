from data import *
from watershed import *

# Load data
imgs = loadImgs()
ws = watershedSegmentation(imgs)
# Apply watershed segmentation
imgs_final, imgs_marker = ws.segmentize()
# Save results
saveImgs(imgs_final, imgs_marker)
# Show results
showImgs(imgs_final)

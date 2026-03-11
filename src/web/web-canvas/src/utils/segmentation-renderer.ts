/**
 * Segmentation Renderer Utility
 *
 * Renders segmented images on-demand using HTML5 Canvas API
 * by applying polygon clipping paths from database coordinates.
 */

export interface SegmentationCoords {
  contours: number[][][];  // Array of polygons, each polygon is array of [x, y] points
  confidence: number;
  model_version: string;
}

/**
 * Creates a segmented (transparent background) image from original image
 * and segmentation coordinates.
 *
 * @param originalImageSrc - Source URL of the original fragment image
 * @param segmentationCoordsJson - JSON string containing contour coordinates
 * @returns Promise resolving to data URL of the segmented image
 */
export async function createSegmentedImage(
  originalImageSrc: string,
  segmentationCoordsJson: string
): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      try {
        // Parse segmentation coordinates
        const coordsData: SegmentationCoords = JSON.parse(segmentationCoordsJson);

        if (!coordsData.contours || coordsData.contours.length === 0) {
          reject(new Error('No contours found in segmentation data'));
          return;
        }

        // Use all valid contours (≥3 points) so every visible fragment piece is shown
        const validContours = coordsData.contours.filter(
          (c) => Array.isArray(c) && c.length >= 3
        );

        if (validContours.length === 0) {
          reject(new Error('Invalid contour data: no contours with at least 3 points'));
          return;
        }

        // Create canvas with same dimensions as original image
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');

        if (!ctx) {
          reject(new Error('Failed to get canvas 2D context'));
          return;
        }

        // Build a compound clipping path — one subpath per contour.
        // Multiple closed subpaths with the default "nonzero" winding rule
        // unions the regions: everything inside any contour is rendered.
        ctx.beginPath();
        validContours.forEach((contour) => {
          contour.forEach((point: number[], index: number) => {
            const [x, y] = point;
            if (index === 0) {
              ctx.moveTo(x, y);
            } else {
              ctx.lineTo(x, y);
            }
          });
          ctx.closePath();
        });

        // Apply compound clipping path and draw image
        ctx.clip();
        ctx.drawImage(img, 0, 0);

        // Convert to data URL (PNG for transparency support)
        const dataUrl = canvas.toDataURL('image/png');
        resolve(dataUrl);

      } catch (error) {
        reject(new Error(`Failed to render segmented image: ${error}`));
      }
    };

    img.onerror = () => {
      reject(new Error('Failed to load original image'));
    };

    // Start loading the image
    img.src = originalImageSrc;
  });
}

/**
 * Check if a fragment has valid segmentation coordinates
 */
export function hasValidSegmentation(segmentationCoordsJson: string | null): boolean {
  if (!segmentationCoordsJson) return false;

  try {
    const coordsData: SegmentationCoords = JSON.parse(segmentationCoordsJson);
    if (!coordsData.contours || !Array.isArray(coordsData.contours) || coordsData.contours.length === 0) {
      return false;
    }
    // Check that at least one contour has enough points
    return coordsData.contours.some(
      (c) => Array.isArray(c) && c.length >= 3
    );
  } catch {
    return false;
  }
}

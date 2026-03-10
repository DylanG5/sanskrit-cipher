/**
 * Hook for loading fragment images with on-demand segmentation
 */

import { useState, useEffect } from 'react';
import { getOrCreateSegmentedImage } from '../utils/segmentation-cache';

export interface UseFragmentImageOptions {
  fragmentId: string;
  imagePath: string;
  segmentationCoords?: string;
  showSegmented: boolean;
  isMirrored?: boolean;
}

/**
 * Custom hook to load fragment images with optional on-demand segmentation.
 *
 * When showSegmented is true and segmentationCoords are available,
 * generates a segmented version using Canvas API and IndexedDB caching.
 */
export function useFragmentImage({
  fragmentId,
  imagePath,
  segmentationCoords,
  showSegmented,
  isMirrored = false,
}: UseFragmentImageOptions): {
  image: HTMLImageElement | HTMLCanvasElement | null;
  isLoading: boolean;
  error: string | null;
} {
  const [image, setImage] = useState<HTMLImageElement | HTMLCanvasElement | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isCancelled = false;

    const loadImage = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // Determine which image source to use
        let imageSrc = imagePath;

        // If segmented view is requested and we have coordinates, generate on-demand
        if (showSegmented && segmentationCoords) {
          const segmentedDataUrl = await getOrCreateSegmentedImage(
            fragmentId,
            imagePath,
            segmentationCoords
          );

          // Use segmented version if generated successfully, otherwise fall back to original
          if (segmentedDataUrl) {
            imageSrc = segmentedDataUrl;
          }
        }

        // Load the image
        const img = new window.Image();
        img.crossOrigin = 'anonymous';

        img.onload = () => {
          if (!isCancelled) {
            if (isMirrored) {
              // Draw the image flipped onto an offscreen canvas so the Konva node
              // always has a positive scaleX — this avoids Transformer glitches
              // when rotating/resizing a negative-scale node.
              const canvas = document.createElement('canvas');
              canvas.width = img.naturalWidth;
              canvas.height = img.naturalHeight;
              const ctx = canvas.getContext('2d');
              if (ctx) {
                ctx.translate(img.naturalWidth, 0);
                ctx.scale(-1, 1);
                ctx.drawImage(img, 0, 0);
              }
              setImage(canvas);
            } else {
              setImage(img);
            }
            setIsLoading(false);
          }
        };

        img.onerror = () => {
          if (!isCancelled) {
            setError('Failed to load image');
            setIsLoading(false);
          }
        };

        img.src = imageSrc;

      } catch (err) {
        if (!isCancelled) {
          console.error('Error loading fragment image:', err);
          setError(String(err));
          setIsLoading(false);
        }
      }
    };

    loadImage();

    // Cleanup function to prevent state updates after unmount
    return () => {
      isCancelled = true;
    };
  }, [fragmentId, imagePath, segmentationCoords, showSegmented, isMirrored]);

  return { image, isLoading, error };
}

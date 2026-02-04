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
}: UseFragmentImageOptions): {
  image: HTMLImageElement | null;
  isLoading: boolean;
  error: string | null;
} {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
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
            setImage(img);
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
  }, [fragmentId, imagePath, segmentationCoords, showSegmented]);

  return { image, isLoading, error };
}

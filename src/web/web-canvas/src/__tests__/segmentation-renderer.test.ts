import { describe, it, expect, vi, beforeEach } from 'vitest';
import { hasValidSegmentation, createSegmentedImage } from '../utils/segmentation-renderer';

// -----------------------------------------------------------------------
// hasValidSegmentation  (pure function – no DOM required)
// -----------------------------------------------------------------------

describe('hasValidSegmentation', () => {
  it('returns false for null', () => {
    expect(hasValidSegmentation(null)).toBe(false);
  });

  it('returns false for empty string', () => {
    expect(hasValidSegmentation('')).toBe(false);
  });

  it('returns false for invalid JSON', () => {
    expect(hasValidSegmentation('not-json')).toBe(false);
  });

  it('returns false when no contours', () => {
    expect(hasValidSegmentation(JSON.stringify({ contours: [] }))).toBe(false);
  });

  it('returns false when contour has < 3 points', () => {
    expect(
      hasValidSegmentation(
        JSON.stringify({ contours: [[[0, 0], [1, 1]]], confidence: 0.9, model_version: '1.0' }),
      ),
    ).toBe(false);
  });

  it('returns true for valid segmentation', () => {
    const data = {
      contours: [[[0, 0], [100, 0], [100, 100], [0, 100]]],
      confidence: 0.9,
      model_version: '1.0',
    };
    expect(hasValidSegmentation(JSON.stringify(data))).toBe(true);
  });

  it('returns false when contours is not an array', () => {
    expect(hasValidSegmentation(JSON.stringify({ contours: 'bad' }))).toBe(false);
  });

  it('returns false when contours[0] is not an array', () => {
    expect(hasValidSegmentation(JSON.stringify({ contours: ['bad'] }))).toBe(false);
  });
});

// -----------------------------------------------------------------------
// createSegmentedImage  (needs Image + Canvas DOM stubs)
// -----------------------------------------------------------------------

describe('createSegmentedImage', () => {
  beforeEach(() => {
    // Minimal canvas context mock
    const ctxMock = {
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      closePath: vi.fn(),
      clip: vi.fn(),
      drawImage: vi.fn(),
    };

    // Mock HTMLCanvasElement
    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      if (tag === 'canvas') {
        return {
          width: 0,
          height: 0,
          getContext: () => ctxMock,
          toDataURL: () => 'data:image/png;base64,AAAA',
        } as unknown as HTMLCanvasElement;
      }
      return document.createElement(tag);
    });
  });

  it('resolves with data URL for valid input', async () => {
    // We need to stub Image so onload fires
    const origImage = globalThis.Image;
    const fakeImages: { onload?: () => void; crossOrigin?: string; src?: string }[] = [];

    globalThis.Image = class {
      onload?: () => void;
      onerror?: () => void;
      crossOrigin = '';
      width = 200;
      height = 200;
      set src(_: string) {
        fakeImages.push(this);
        // Fire onload asynchronously
        setTimeout(() => this.onload?.(), 0);
      }
    } as unknown as typeof Image;

    const coords = JSON.stringify({
      contours: [[[0, 0], [100, 0], [100, 100], [0, 100]]],
      confidence: 0.9,
      model_version: '1.0',
    });

    const result = await createSegmentedImage('http://example.com/img.png', coords);
    expect(result).toContain('data:image/png');

    globalThis.Image = origImage;
  });

  it('rejects when no contours', async () => {
    const origImage = globalThis.Image;
    globalThis.Image = class {
      onload?: () => void;
      onerror?: () => void;
      crossOrigin = '';
      width = 200;
      height = 200;
      set src(_: string) {
        setTimeout(() => this.onload?.(), 0);
      }
    } as unknown as typeof Image;

    await expect(
      createSegmentedImage('http://example.com/img.png', JSON.stringify({ contours: [] })),
    ).rejects.toThrow(/no contours/i);

    globalThis.Image = origImage;
  });

  it('rejects when contour has < 3 points', async () => {
    const origImage = globalThis.Image;
    globalThis.Image = class {
      onload?: () => void;
      onerror?: () => void;
      crossOrigin = '';
      width = 200;
      height = 200;
      set src(_: string) {
        setTimeout(() => this.onload?.(), 0);
      }
    } as unknown as typeof Image;

    const coords = JSON.stringify({
      contours: [[[0, 0], [1, 1]]],
      confidence: 0.9,
      model_version: '1.0',
    });

    await expect(createSegmentedImage('http://ex.com/img.png', coords)).rejects.toThrow(
      /at least 3 points/i,
    );

    globalThis.Image = origImage;
  });

  it('rejects when image fails to load', async () => {
    const origImage = globalThis.Image;
    globalThis.Image = class {
      onload?: () => void;
      onerror?: () => void;
      crossOrigin = '';
      width = 0;
      height = 0;
      set src(_: string) {
        setTimeout(() => this.onerror?.(), 0);
      }
    } as unknown as typeof Image;

    const coords = JSON.stringify({
      contours: [[[0, 0], [100, 0], [100, 100]]],
      confidence: 0.9,
      model_version: '1.0',
    });

    await expect(createSegmentedImage('bad-url', coords)).rejects.toThrow(/failed to load/i);

    globalThis.Image = origImage;
  });
});

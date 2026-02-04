import React, { useRef, useState, useEffect } from 'react';
import { Stage, Layer, Image as KonvaImage, Transformer, Line } from 'react-konva';
import { CanvasFragment } from '../types/fragment';
import Konva from 'konva';

interface CanvasProps {
  fragments: CanvasFragment[];
  onFragmentsChange: (fragments: CanvasFragment[]) => void;
  selectedFragmentIds: string[];
  onSelectionChange: (ids: string[]) => void;
  onEdgeMatch?: (fragmentId: string) => void;
  onFragmentDoubleClick?: (fragmentId: string) => void;
  isGridVisible?: boolean;
  gridScale?: number; // pixels per cm
}

interface FragmentImageProps {
  fragment: CanvasFragment;
  isSelected: boolean;
  onSelect: (e: any) => void;
  onChange: (newAttrs: Partial<CanvasFragment>) => void;
  onDoubleClick?: () => void;
  onDragStart?: () => void;
  onDragEnd?: () => void;
  onTransformStart?: () => void;
  onTransformEnd?: () => void;
}

const FragmentImage: React.FC<FragmentImageProps> = ({
  fragment,
  isSelected,
  onSelect,
  onChange,
  onDoubleClick,
  onDragStart,
  onDragEnd,
  onTransformStart,
  onTransformEnd,
}) => {
  const imageRef = useRef<Konva.Image>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    console.log('Loading image:', fragment.imagePath);
    const img = new window.Image();
    img.src = fragment.imagePath;
    img.onload = () => {
      console.log('Image loaded successfully:', fragment.imagePath);
      setImage(img);
    };
    img.onerror = (e) => {
      console.error('Error loading image:', fragment.imagePath, e);
    };
  }, [fragment.imagePath]);

  return (
    <>
      <KonvaImage
        ref={imageRef}
        id={fragment.id}
        name={fragment.name}
        image={image}
        x={fragment.x}
        y={fragment.y}
        width={fragment.width}
        height={fragment.height}
        rotation={fragment.rotation}
        scaleX={fragment.scaleX}
        scaleY={fragment.scaleY}
        draggable={!fragment.isLocked}
        onClick={(e) => onSelect(e.evt)}
        onTap={(e) => onSelect(e.evt)}
        onDblClick={() => {
          if (onDoubleClick) onDoubleClick();
        }}
        onDblTap={() => {
          if (onDoubleClick) onDoubleClick();
        }}
        onDragStart={() => {
          if (onDragStart) onDragStart();
        }}
        onDragEnd={(e) => {
          onChange({
            x: e.target.x(),
            y: e.target.y(),
          });
          if (onDragEnd) onDragEnd();
        }}
        onTransform={() => {
          if (onTransformStart) onTransformStart();
        }}
        onTransformEnd={() => {
          const node = imageRef.current;
          if (!node) return;

          const scaleX = node.scaleX();
          const scaleY = node.scaleY();

          // Reset scale and apply it to width/height
          node.scaleX(1);
          node.scaleY(1);

          const newWidth = Math.max(5, node.width() * scaleX);
          const newHeight = Math.max(5, node.height() * scaleY);

          // Check if dimensions actually changed (resize vs just transform/rotate)
          const widthChanged = Math.abs(newWidth - fragment.width) > 1;
          const heightChanged = Math.abs(newHeight - fragment.height) > 1;
          const wasResized = widthChanged || heightChanged;

          onChange({
            x: node.x(),
            y: node.y(),
            width: newWidth,
            height: newHeight,
            rotation: node.rotation(),
            hasBeenResized: wasResized ? true : fragment.hasBeenResized,
          });
          if (onTransformEnd) onTransformEnd();
        }}
        opacity={fragment.isLocked ? 0.7 : 1}
        shadowColor={isSelected ? 'blue' : undefined}
        shadowBlur={isSelected ? 10 : 0}
        shadowOpacity={isSelected ? 0.5 : 0}
      />
    </>
  );
};

const Canvas: React.FC<CanvasProps> = ({
  fragments,
  onFragmentsChange,
  selectedFragmentIds,
  onSelectionChange,
  onEdgeMatch,
  onFragmentDoubleClick,
  isGridVisible = false,
  gridScale = 25,
}) => {
  console.log('Canvas rendering with fragments:', fragments.length);

  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const layerRef = useRef<Konva.Layer>(null);
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });
  const [edgeMatchButtonPosition, setEdgeMatchButtonPosition] = useState<{ x: number; y: number; fragmentId: string } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const isTransformingRef = useRef(false);

  // Update stage size
  useEffect(() => {
    const updateSize = () => {
      const container = containerRef.current;
      if (container) {
        const newSize = {
          width: container.clientWidth,
          height: container.clientHeight,
        };
        console.log('Stage size updated:', newSize);
        setStageSize(newSize);
      }
    };

    // Use setTimeout to ensure DOM is ready
    const timer = setTimeout(updateSize, 0);
    window.addEventListener('resize', updateSize);
    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', updateSize);
    };
  }, []);

  // Update transformer
  useEffect(() => {
    const transformer = transformerRef.current;
    const layer = layerRef.current;
    if (!transformer || !layer) return;

    const selectedNodes = selectedFragmentIds
      .map((id) => {
        const fragment = fragments.find((f) => f.id === id);
        if (!fragment || fragment.isLocked) return null;
        return layer.findOne(`#${id}`);
      })
      .filter(Boolean) as Konva.Node[];

    transformer.nodes(selectedNodes);
    transformer.getLayer()?.batchDraw();
  }, [selectedFragmentIds, fragments]);

  // Update edge match button position when a single fragment is selected
  useEffect(() => {
    if (selectedFragmentIds.length === 1) {
      const selectedFragment = fragments.find((f) => f.id === selectedFragmentIds[0]);
      if (selectedFragment) {
        // Position button at the top-right corner of the fragment
        setEdgeMatchButtonPosition({
          x: selectedFragment.x + selectedFragment.width + 10,
          y: selectedFragment.y,
          fragmentId: selectedFragment.id,
        });
      }
    } else {
      setEdgeMatchButtonPosition(null);
    }
  }, [selectedFragmentIds, fragments]);

  const handleFragmentChange = (id: string, newAttrs: Partial<CanvasFragment>) => {
    const updatedFragments = fragments.map((f) =>
      f.id === id ? { ...f, ...newAttrs } : f
    );
    onFragmentsChange(updatedFragments);
  };

  const handleSelect = (id: string, e?: React.MouseEvent | React.TouchEvent) => {
    // Allow selecting locked fragments so they can be unlocked
    const metaPressed = e && ('ctrlKey' in e ? e.ctrlKey || e.metaKey : false);
    const shiftPressed = e && ('shiftKey' in e ? e.shiftKey : false);

    if (metaPressed || shiftPressed) {
      // Multi-select
      if (selectedFragmentIds.includes(id)) {
        onSelectionChange(selectedFragmentIds.filter((fid) => fid !== id));
      } else {
        onSelectionChange([...selectedFragmentIds, id]);
      }
    } else {
      // Single select
      onSelectionChange([id]);
    }
  };

  const handleStageClick = (e: Konva.KonvaEventObject<MouseEvent>) => {
    // Deselect when clicking on empty area
    if (e.target === e.target.getStage()) {
      onSelectionChange([]);
    }
  };

  // Generate grid lines
  const generateGridLines = () => {
    const lines: JSX.Element[] = [];
    const { width, height } = stageSize;

    if (!isGridVisible || width === 0 || height === 0) {
      return lines;
    }

    // Vertical lines
    for (let x = 0; x <= width; x += gridScale) {
      lines.push(
        <Line
          key={`v-${x}`}
          points={[x, 0, x, height]}
          stroke="#cbd5e1"
          strokeWidth={x % (gridScale * 5) === 0 ? 1.5 : 0.5}
          opacity={x % (gridScale * 5) === 0 ? 0.4 : 0.25}
          listening={false}
        />
      );
    }

    // Horizontal lines
    for (let y = 0; y <= height; y += gridScale) {
      lines.push(
        <Line
          key={`h-${y}`}
          points={[0, y, width, y]}
          stroke="#cbd5e1"
          strokeWidth={y % (gridScale * 5) === 0 ? 1.5 : 0.5}
          opacity={y % (gridScale * 5) === 0 ? 0.4 : 0.25}
          listening={false}
        />
      );
    }

    return lines;
  };

  console.log('Rendering stage with size:', stageSize);
  console.log('Fragments to render:', fragments.map(f => ({ id: f.id, x: f.x, y: f.y, width: f.width, height: f.height })));

  return (
    <div ref={containerRef} className="w-full h-full bg-gradient-to-br from-slate-100 via-slate-50 to-slate-100 relative overflow-hidden">
      <Stage
        ref={stageRef}
        width={stageSize.width}
        height={stageSize.height}
        onClick={handleStageClick}
        onTap={handleStageClick}
      >
        {/* Grid Layer */}
        <Layer listening={false}>
          {generateGridLines()}
        </Layer>

        {/* Fragments Layer */}
        <Layer ref={layerRef}>
          {fragments.map((fragment) => (
            <FragmentImage
              key={fragment.id}
              fragment={fragment}
              isSelected={selectedFragmentIds.includes(fragment.id)}
              onSelect={(e) => handleSelect(fragment.id, e)}
              onChange={(newAttrs) => handleFragmentChange(fragment.id, newAttrs)}
              onDoubleClick={() => {
                if (onFragmentDoubleClick) {
                  onFragmentDoubleClick(fragment.fragmentId);
                }
              }}
              onDragStart={() => setIsDragging(true)}
              onDragEnd={() => {
                // Small delay to allow position to update before showing button
                setTimeout(() => setIsDragging(false), 0);
              }}
              onTransformStart={() => {
                if (!isTransformingRef.current) {
                  isTransformingRef.current = true;
                  setIsDragging(true);
                }
              }}
              onTransformEnd={() => {
                isTransformingRef.current = false;
                // Small delay to allow position to update before showing button
                setTimeout(() => setIsDragging(false), 0);
              }}
            />
          ))}
          <Transformer
            ref={transformerRef}
            boundBoxFunc={(oldBox, newBox) => {
              // Limit resize
              if (newBox.width < 5 || newBox.height < 5) {
                return oldBox;
              }
              return newBox;
            }}
          />
        </Layer>
      </Stage>

      {/* Edge Match Button */}
      {/* {edgeMatchButtonPosition && !isDragging && (
        <button
          onClick={() => {
            if (onEdgeMatch) {
              onEdgeMatch(edgeMatchButtonPosition.fragmentId);
            }
          }}
          className="absolute px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg shadow-lg flex items-center gap-2 z-50 hover:scale-105 font-medium"
          style={{
            left: `${edgeMatchButtonPosition.x}px`,
            top: `${edgeMatchButtonPosition.y}px`,
            transition: 'background-color 0.2s, transform 0.2s',
          }}
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M11 4a2 2 0 114 0v1a1 1 0 001 1h3a1 1 0 011 1v3a1 1 0 01-1 1h-1a2 2 0 100 4h1a1 1 0 011 1v3a1 1 0 01-1 1h-3a1 1 0 01-1-1v-1a2 2 0 10-4 0v1a1 1 0 01-1 1H7a1 1 0 01-1-1v-3a1 1 0 00-1-1H4a2 2 0 110-4h1a1 1 0 001-1V7a1 1 0 011-1h3a1 1 0 001-1V4z"
            />
          </svg>
          Edge Match?
        </button>
      )} */}

      {/* Grid Scale Indicator */}
      {isGridVisible && (
        <div className="absolute bottom-4 left-4 bg-white/95 backdrop-blur-sm rounded-lg shadow-lg border-2 border-blue-400 p-3 z-20 max-w-sm">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1v-4zM14 5a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM14 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
            </svg>
            <div className="flex-1">
              <div className="text-xs font-bold text-slate-800 mb-2">Grid Scale Reference</div>

              {/* Small square explanation */}
              <div className="flex items-center gap-2 mb-2">
                <div
                  className="border-2 border-slate-400 bg-slate-50 flex-shrink-0"
                  style={{ width: `${gridScale}px`, height: `${gridScale}px` }}
                />
                <div className="text-xs">
                  <div className="font-semibold text-blue-600">Small square (thin lines)</div>
                  <div className="text-slate-600">{gridScale}px = <span className="font-bold">1 cm × 1 cm</span></div>
                </div>
              </div>

              {/* Large square explanation */}
              <div className="flex items-center gap-2">
                <div
                  className="border-[3px] border-slate-600 bg-slate-50 flex-shrink-0 relative"
                  style={{ width: `${gridScale}px`, height: `${gridScale}px` }}
                >
                  {/* Mini 5x5 grid inside */}
                  {[...Array(4)].map((_, i) => (
                    <div
                      key={`v${i}`}
                      className="absolute top-0 bottom-0 border-l border-slate-300"
                      style={{ left: `${((i + 1) * 20)}%` }}
                    />
                  ))}
                  {[...Array(4)].map((_, i) => (
                    <div
                      key={`h${i}`}
                      className="absolute left-0 right-0 border-t border-slate-300"
                      style={{ top: `${((i + 1) * 20)}%` }}
                    />
                  ))}
                </div>
                <div className="text-xs">
                  <div className="font-semibold text-slate-700">Large square (bold lines)</div>
                  <div className="text-slate-600">{gridScale * 5}px = <span className="font-bold">5 cm × 5 cm</span></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Instructions overlay */}
      {fragments.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="bg-white/95 backdrop-blur-sm p-10 rounded-2xl shadow-2xl border border-slate-200 text-center max-w-lg">
            <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-5 shadow-lg">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 4a2 2 0 114 0v1a1 1 0 001 1h3a1 1 0 011 1v3a1 1 0 01-1 1h-1a2 2 0 100 4h1a1 1 0 011 1v3a1 1 0 01-1 1h-3a1 1 0 01-1-1v-1a2 2 0 10-4 0v1a1 1 0 01-1 1H7a1 1 0 01-1-1v-3a1 1 0 00-1-1H4a2 2 0 110-4h1a1 1 0 001-1V7a1 1 0 011-1h3a1 1 0 001-1V4z" />
              </svg>
            </div>
            <h3 className="text-2xl font-semibold text-slate-800 mb-3">
              Welcome to Fragment Canvas
            </h3>
            <p className="text-slate-600 mb-6 text-base">
              Drag and drop fragments from the sidebar to begin reconstructing manuscripts
            </p>
            <div className="space-y-2 text-sm text-slate-500">
              <div className="flex items-center justify-center gap-2">
                <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
                </svg>
                <span className="font-medium">Click to select • Shift/Ctrl+Click for multiple</span>
              </div>
              <div className="flex items-center justify-center gap-2">
                <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                </svg>
                <span className="font-medium">Drag to move • Use corner handles to resize</span>
              </div>
              <div className="flex items-center justify-center gap-2">
                <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                <span className="font-medium">Drag near corners to rotate fragments</span>
              </div>
              <div className="flex items-center justify-center gap-2">
                <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
                <span className="font-medium">Use Lock button to prevent accidental changes</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Canvas;

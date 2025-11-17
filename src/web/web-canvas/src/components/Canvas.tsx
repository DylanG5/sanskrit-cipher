import React, { useRef, useState, useEffect } from 'react';
import { Stage, Layer, Image as KonvaImage, Transformer } from 'react-konva';
import { CanvasFragment } from '../types/fragment';
import Konva from 'konva';

interface CanvasProps {
  fragments: CanvasFragment[];
  onFragmentsChange: (fragments: CanvasFragment[]) => void;
  selectedFragmentIds: string[];
  onSelectionChange: (ids: string[]) => void;
}

interface FragmentImageProps {
  fragment: CanvasFragment;
  isSelected: boolean;
  onSelect: () => void;
  onChange: (newAttrs: Partial<CanvasFragment>) => void;
}

const FragmentImage: React.FC<FragmentImageProps> = ({
  fragment,
  isSelected,
  onSelect,
  onChange,
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
        onClick={onSelect}
        onTap={onSelect}
        onDragEnd={(e) => {
          onChange({
            x: e.target.x(),
            y: e.target.y(),
          });
        }}
        onTransformEnd={() => {
          const node = imageRef.current;
          if (!node) return;

          const scaleX = node.scaleX();
          const scaleY = node.scaleY();

          // Reset scale and apply it to width/height
          node.scaleX(1);
          node.scaleY(1);

          onChange({
            x: node.x(),
            y: node.y(),
            width: Math.max(5, node.width() * scaleX),
            height: Math.max(5, node.height() * scaleY),
            rotation: node.rotation(),
          });
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
}) => {
  console.log('Canvas rendering with fragments:', fragments.length);

  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const layerRef = useRef<Konva.Layer>(null);
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });

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
        <Layer ref={layerRef}>
          {fragments.map((fragment) => (
            <FragmentImage
              key={fragment.id}
              fragment={fragment}
              isSelected={selectedFragmentIds.includes(fragment.id)}
              onSelect={(e) => handleSelect(fragment.id, e)}
              onChange={(newAttrs) => handleFragmentChange(fragment.id, newAttrs)}
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
                <span>Click to select • Shift/Ctrl+Click for multiple</span>
              </div>
              <div className="flex items-center justify-center gap-2">
                <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                </svg>
                <span>Drag to move • Use handles to resize & rotate</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Canvas;

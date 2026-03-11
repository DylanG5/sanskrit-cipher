import React, { useRef, useState, useEffect } from "react";
import {
  Stage,
  Layer,
  Image as KonvaImage,
  Transformer,
  Line,
} from "react-konva";
import { CanvasFragment } from "../types/fragment";
import { useFragmentImage } from "../hooks/useFragmentImage";
import Konva from "konva";

const MIN_SCALE = 0.1;
const MAX_SCALE = 5;
const KEYBOARD_ZOOM_FACTOR = 1.2;

interface CanvasProps {
  fragments: CanvasFragment[];
  onFragmentsChange: (fragments: CanvasFragment[]) => void;
  selectedFragmentIds: string[];
  onSelectionChange: (ids: string[]) => void;
  onEdgeMatch?: (fragmentId: string) => void;
  onFragmentDoubleClick?: (fragmentId: string) => void;
  isGridVisible?: boolean;
  gridScale?: number; // pixels per cm
  onViewportChange?: (scale: number, position: { x: number; y: number }) => void;
}

interface FragmentImageProps {
  fragment: CanvasFragment;
  isSelected: boolean;
  onSelect: (e: any) => void;
  onChange: (newAttrs: Partial<CanvasFragment>) => void;
  onDoubleClick?: () => void;
  onDragStart?: (e: Konva.KonvaEventObject<DragEvent>) => void;
  onDragMove?: (e: Konva.KonvaEventObject<DragEvent>) => void;
  onDragEnd?: (e: Konva.KonvaEventObject<DragEvent>) => void;
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
  onDragMove,
  onDragEnd,
  onTransformStart,
  onTransformEnd,
}) => {
  const imageRef = useRef<Konva.Image>(null);

  // Use custom hook for loading images with on-demand segmentation
  const { image, isLoading, error } = useFragmentImage({
    fragmentId: fragment.fragmentId,
    imagePath: fragment.imagePath,
    segmentationCoords: fragment.segmentationCoords,
    showSegmented: fragment.showSegmented,
    isMirrored: fragment.isMirrored,
  });

  // Log errors
  useEffect(() => {
    if (error) {
      console.error("Error loading fragment image:", error);
    }
  }, [error]);

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
        onDragStart={(e) => {
          if (onDragStart) onDragStart(e);
        }}
        onDragMove={(e) => {
          if (onDragMove) onDragMove(e);
        }}
        onDragEnd={(e) => {
          onChange({
            x: e.target.x(),
            y: e.target.y(),
          });
          if (onDragEnd) onDragEnd(e);
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
        shadowColor={isSelected ? "blue" : undefined}
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
  onViewportChange,
}) => {
  console.log("Canvas rendering with fragments:", fragments.length);

  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const layerRef = useRef<Konva.Layer>(null);
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });
  const [edgeMatchButtonPosition, setEdgeMatchButtonPosition] = useState<{
    x: number;
    y: number;
    fragmentId: string;
  } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const isTransformingRef = useRef(false);
  const [showSplash, setShowSplash] = useState(true);
  const splashTimerRef = useRef<NodeJS.Timeout | null>(null);
  const [showGridReference, setShowGridReference] = useState(false);
  const gridReferenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const [stageScale, setStageScale] = useState(1);
  const [stagePosition, setStagePosition] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStartRef = useRef<{ x: number; y: number } | null>(null);

  // Auto-dismiss splash after 4 seconds
  useEffect(() => {
    if (fragments.length === 0 && showSplash) {
      splashTimerRef.current = setTimeout(() => setShowSplash(false), 4000);
    }
    return () => {
      if (splashTimerRef.current) clearTimeout(splashTimerRef.current);
    };
  }, [fragments.length, showSplash]);

  // Auto-hide grid reference after 5 seconds
  useEffect(() => {
    if (gridReferenceTimerRef.current) {
      clearTimeout(gridReferenceTimerRef.current);
      gridReferenceTimerRef.current = null;
    }
    if (isGridVisible) {
      setShowGridReference(true);
      gridReferenceTimerRef.current = setTimeout(() => {
        setShowGridReference(false);
        gridReferenceTimerRef.current = null;
      }, 5000);
    } else {
      setShowGridReference(false);
    }
    return () => {
      if (gridReferenceTimerRef.current) {
        clearTimeout(gridReferenceTimerRef.current);
      }
    };
  }, [isGridVisible]);

  // Reset splash when canvas is cleared
  useEffect(() => {
    if (fragments.length === 0) {
      setShowSplash(true);
    }
  }, [fragments.length]);
  // Notify parent of viewport changes so it can convert drop coordinates
  useEffect(() => {
    if (onViewportChange) {
      onViewportChange(stageScale, stagePosition);
    }
  }, [stageScale, stagePosition, onViewportChange]);

  const dragContextRef = useRef<{
    anchorId: string;
    anchorStart: { x: number; y: number };
    memberStart: Record<string, { x: number; y: number }>;
  } | null>(null);

  // Update stage size
  useEffect(() => {
    const updateSize = () => {
      const container = containerRef.current;
      if (container) {
        const newSize = {
          width: container.clientWidth,
          height: container.clientHeight,
        };
        console.log("Stage size updated:", newSize);
        setStageSize(newSize);
      }
    };

    // Use setTimeout to ensure DOM is ready
    const timer = setTimeout(updateSize, 0);
    window.addEventListener("resize", updateSize);
    return () => {
      clearTimeout(timer);
      window.removeEventListener("resize", updateSize);
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
      const selectedFragment = fragments.find(
        (f) => f.id === selectedFragmentIds[0],
      );
      if (selectedFragment) {
        // Position button at the top-right corner of the fragment (screen coords)
        setEdgeMatchButtonPosition({
          x:
            (selectedFragment.x + selectedFragment.width) * stageScale +
            stagePosition.x +
            10,
          y: selectedFragment.y * stageScale + stagePosition.y,
          fragmentId: selectedFragment.fragmentId,
        });
      }
    } else {
      setEdgeMatchButtonPosition(null);
    }
  }, [selectedFragmentIds, fragments, stageScale, stagePosition]);

  const handleFragmentChange = (
    id: string,
    newAttrs: Partial<CanvasFragment>,
  ) => {
    const updatedFragments = fragments.map((f) =>
      f.id === id ? { ...f, ...newAttrs } : f,
    );
    onFragmentsChange(updatedFragments);
  };

  const getSelectionUnit = (id: string): string[] => {
    const target = fragments.find((f) => f.id === id);
    if (!target) return [id];
    if (!target.groupId) return [id];
    return fragments
      .filter((f) => f.groupId === target.groupId)
      .map((f) => f.id);
  };

  const handleSelect = (
    id: string,
    e?: React.MouseEvent | React.TouchEvent,
  ) => {
    // Allow selecting locked fragments so they can be unlocked
    const metaPressed = e && ("ctrlKey" in e ? e.ctrlKey || e.metaKey : false);
    const shiftPressed = e && ("shiftKey" in e ? e.shiftKey : false);
    const idsToSelect = getSelectionUnit(id);

    if (metaPressed || shiftPressed) {
      // Multi-select
      const allSelected = idsToSelect.every((fid) =>
        selectedFragmentIds.includes(fid),
      );
      if (allSelected) {
        onSelectionChange(
          selectedFragmentIds.filter((fid) => !idsToSelect.includes(fid)),
        );
      } else {
        onSelectionChange(
          Array.from(new Set([...selectedFragmentIds, ...idsToSelect])),
        );
      }
    } else {
      // Single select
      onSelectionChange(idsToSelect);
    }
  };

  const handleFragmentDragStart = (id: string) => {
    const dragIds = selectedFragmentIds.includes(id)
      ? selectedFragmentIds
      : [id];
    if (dragIds.length <= 1) {
      dragContextRef.current = null;
      return;
    }

    const anchor = fragments.find((f) => f.id === id);
    if (!anchor) {
      dragContextRef.current = null;
      return;
    }

    const memberStart: Record<string, { x: number; y: number }> = {};
    for (const fid of dragIds) {
      const fragment = fragments.find((f) => f.id === fid);
      if (fragment && !fragment.isLocked) {
        memberStart[fid] = { x: fragment.x, y: fragment.y };
      }
    }

    if (Object.keys(memberStart).length <= 1) {
      dragContextRef.current = null;
      return;
    }

    dragContextRef.current = {
      anchorId: id,
      anchorStart: { x: anchor.x, y: anchor.y },
      memberStart,
    };
  };

  const handleFragmentDragMove = (
    id: string,
    e: Konva.KonvaEventObject<DragEvent>,
  ) => {
    const dragCtx = dragContextRef.current;
    if (!dragCtx || dragCtx.anchorId !== id) return;

    const dx = e.target.x() - dragCtx.anchorStart.x;
    const dy = e.target.y() - dragCtx.anchorStart.y;

    const updated = fragments.map((fragment) => {
      const start = dragCtx.memberStart[fragment.id];
      if (!start) return fragment;
      return {
        ...fragment,
        x: start.x + dx,
        y: start.y + dy,
      };
    });

    onFragmentsChange(updated);
  };

  const handleFragmentDragEnd = () => {
    dragContextRef.current = null;
  };

  const handleStageClick = (e: Konva.KonvaEventObject<MouseEvent>) => {
    // Deselect when clicking on empty area
    if (e.target === e.target.getStage()) {
      onSelectionChange([]);
    }
  };

  // Click-drag panning
  const handleMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    // Only pan when clicking on the stage background (not on fragments)
    if (e.target !== e.target.getStage()) return;
    setIsPanning(true);
    panStartRef.current = { x: e.evt.clientX, y: e.evt.clientY };
  };

  const handleMouseMove = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (!isPanning || !panStartRef.current) return;
    const dx = e.evt.clientX - panStartRef.current.x;
    const dy = e.evt.clientY - panStartRef.current.y;
    panStartRef.current = { x: e.evt.clientX, y: e.evt.clientY };
    setStagePosition((prev) => ({ x: prev.x + dx, y: prev.y + dy }));
  };

  const handleMouseUp = () => {
    setIsPanning(false);
    panStartRef.current = null;
  };

  // Recenter view to the center of all fragments (keeps current zoom)
  const recenterToFragments = () => {
    if (fragments.length === 0) {
      setStagePosition({ x: 0, y: 0 });
      return;
    }

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const f of fragments) {
      minX = Math.min(minX, f.x);
      minY = Math.min(minY, f.y);
      maxX = Math.max(maxX, f.x + f.width * (f.scaleX ?? 1));
      maxY = Math.max(maxY, f.y + f.height * (f.scaleY ?? 1));
    }

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    setStagePosition({
      x: stageSize.width / 2 - centerX * stageScale,
      y: stageSize.height / 2 - centerY * stageScale,
    });
  };

  // Zoom toward the center of the viewport
  const zoomToCenter = (newScale: number) => {
    const clampedScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, newScale));
    const center = { x: stageSize.width / 2, y: stageSize.height / 2 };
    const pointTo = {
      x: (center.x - stagePosition.x) / stageScale,
      y: (center.y - stagePosition.y) / stageScale,
    };
    setStageScale(clampedScale);
    setStagePosition({
      x: center.x - pointTo.x * clampedScale,
      y: center.y - pointTo.y * clampedScale,
    });
  };

  // Handle wheel for zoom (pinch) and pan (scroll)
  const handleWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;

    const pointer = stage.getPointerPosition();
    if (!pointer) return;

    // Always zoom on wheel (scroll wheel, trackpad pinch, Ctrl/Cmd+scroll)
    const oldScale = stageScale;
    const newScale = Math.min(
      MAX_SCALE,
      Math.max(MIN_SCALE, oldScale * Math.exp(-e.evt.deltaY * 0.01)),
    );

    const mousePointTo = {
      x: (pointer.x - stagePosition.x) / oldScale,
      y: (pointer.y - stagePosition.y) / oldScale,
    };

    setStageScale(newScale);
    setStagePosition({
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    });
  };

  // Keyboard zoom: Cmd/Ctrl + Plus/Minus/Zero
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey)) return;

      let scaleFactor: number | null = null;

      if (e.key === "=" || e.key === "+") {
        e.preventDefault();
        scaleFactor = KEYBOARD_ZOOM_FACTOR;
      } else if (e.key === "-" || e.key === "_") {
        e.preventDefault();
        scaleFactor = 1 / KEYBOARD_ZOOM_FACTOR;
      } else if (e.key === "0") {
        e.preventDefault();
        setStageScale(1);
        setStagePosition({ x: 0, y: 0 });
        return;
      }

      if (scaleFactor === null) return;

      const newScale = Math.min(
        MAX_SCALE,
        Math.max(MIN_SCALE, stageScale * scaleFactor),
      );
      const center = { x: stageSize.width / 2, y: stageSize.height / 2 };
      const pointTo = {
        x: (center.x - stagePosition.x) / stageScale,
        y: (center.y - stagePosition.y) / stageScale,
      };

      setStageScale(newScale);
      setStagePosition({
        x: center.x - pointTo.x * newScale,
        y: center.y - pointTo.y * newScale,
      });
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [stageScale, stagePosition, stageSize]);

  // Generate grid lines covering the visible area at the current zoom level
  const generateGridLines = () => {
    const lines: JSX.Element[] = [];
    const { width, height } = stageSize;

    if (!isGridVisible || width === 0 || height === 0) {
      return lines;
    }

    // Calculate visible area in stage coordinates
    const visibleX = -stagePosition.x / stageScale;
    const visibleY = -stagePosition.y / stageScale;
    const visibleWidth = width / stageScale;
    const visibleHeight = height / stageScale;

    const startX = Math.floor(visibleX / gridScale) * gridScale;
    const endX = visibleX + visibleWidth + gridScale;
    const startY = Math.floor(visibleY / gridScale) * gridScale;
    const endY = visibleY + visibleHeight + gridScale;

    // Vertical lines
    for (let x = startX; x <= endX; x += gridScale) {
      const gridIndex = Math.round(x / gridScale);
      lines.push(
        <Line
          key={`v-${x}`}
          points={[x, startY, x, endY]}
          stroke="#cbd5e1"
          strokeWidth={gridIndex % 5 === 0 ? 1.5 : 0.5}
          opacity={gridIndex % 5 === 0 ? 0.4 : 0.25}
          listening={false}
        />,
      );
    }

    // Horizontal lines
    for (let y = startY; y <= endY; y += gridScale) {
      const gridIndex = Math.round(y / gridScale);
      lines.push(
        <Line
          key={`h-${y}`}
          points={[startX, y, endX, y]}
          stroke="#cbd5e1"
          strokeWidth={gridIndex % 5 === 0 ? 1.5 : 0.5}
          opacity={gridIndex % 5 === 0 ? 0.4 : 0.25}
          listening={false}
        />,
      );
    }

    return lines;
  };

  console.log("Rendering stage with size:", stageSize);
  console.log(
    "Fragments to render:",
    fragments.map((f) => ({
      id: f.id,
      x: f.x,
      y: f.y,
      width: f.width,
      height: f.height,
    })),
  );

  return (
    <div
      ref={containerRef}
      className="w-full h-full bg-gradient-to-br from-slate-100 via-slate-50 to-slate-100 relative overflow-hidden"
    >
      <Stage
        ref={stageRef}
        width={stageSize.width}
        height={stageSize.height}
        scaleX={stageScale}
        scaleY={stageScale}
        x={stagePosition.x}
        y={stagePosition.y}
        onClick={handleStageClick}
        onTap={handleStageClick}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: isPanning ? 'grabbing' : 'default' }}
      >
        {/* Grid Layer */}
        <Layer listening={false}>{generateGridLines()}</Layer>

        {/* Fragments Layer */}
        <Layer ref={layerRef}>
          {fragments.map((fragment) => (
            <FragmentImage
              key={fragment.id}
              fragment={fragment}
              isSelected={selectedFragmentIds.includes(fragment.id)}
              onSelect={(e) => handleSelect(fragment.id, e)}
              onChange={(newAttrs) =>
                handleFragmentChange(fragment.id, newAttrs)
              }
              onDoubleClick={() => {
                if (onFragmentDoubleClick) {
                  onFragmentDoubleClick(fragment.fragmentId);
                }
              }}
              onDragStart={() => {
                handleFragmentDragStart(fragment.id);
                setIsDragging(true);
              }}
              onDragMove={(e) => {
                handleFragmentDragMove(fragment.id, e);
              }}
              onDragEnd={() => {
                handleFragmentDragEnd();
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
      {edgeMatchButtonPosition && !isDragging && (
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
            transition: "background-color 0.2s, transform 0.2s",
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
      )}

      {/* Grid Scale Indicator */}
      {isGridVisible && showGridReference && (
        <div className="absolute bottom-4 left-4 bg-white/95 backdrop-blur-sm rounded-lg shadow-lg border-2 border-blue-400 p-3 z-20 max-w-sm">
          <div className="flex items-start gap-3">
            <svg
              className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 5a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1v-4zM14 5a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM14 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"
              />
            </svg>
            <div className="flex-1">
              <div className="text-xs font-bold text-slate-800 mb-2">
                Grid Scale Reference
              </div>

              {/* Small square explanation */}
              <div className="flex items-center gap-2 mb-2">
                <div
                  className="border-2 border-slate-400 bg-slate-50 flex-shrink-0"
                  style={{ width: `${gridScale}px`, height: `${gridScale}px` }}
                />
                <div className="text-xs">
                  <div className="font-semibold text-blue-600">
                    Small square (thin lines)
                  </div>
                  <div className="text-slate-600">
                    {gridScale}px ={" "}
                    <span className="font-bold">1 cm × 1 cm</span>
                  </div>
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
                      style={{ left: `${(i + 1) * 20}%` }}
                    />
                  ))}
                  {[...Array(4)].map((_, i) => (
                    <div
                      key={`h${i}`}
                      className="absolute left-0 right-0 border-t border-slate-300"
                      style={{ top: `${(i + 1) * 20}%` }}
                    />
                  ))}
                </div>
                <div className="text-xs">
                  <div className="font-semibold text-slate-700">
                    Large square (bold lines)
                  </div>
                  <div className="text-slate-600">
                    {gridScale * 5}px ={" "}
                    <span className="font-bold">5 cm × 5 cm</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Instructions overlay */}
      {fragments.length === 0 && showSplash && (
        <div
          className="absolute inset-0 flex items-center justify-center cursor-pointer"
          onClick={() => setShowSplash(false)}
        >
          <div className="bg-white/95 backdrop-blur-sm p-10 rounded-2xl shadow-2xl border border-slate-200 text-center max-w-lg pointer-events-none">
            <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-5 shadow-lg">
              <svg
                className="w-8 h-8 text-white"
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
            </div>
            <h3 className="text-2xl font-semibold text-slate-800 mb-3">
              Welcome to Fragment Canvas
            </h3>
            <p className="text-slate-600 mb-6 text-base">
              Drag and drop fragments from the sidebar to begin reconstructing
              manuscripts
            </p>
            <div className="space-y-2 text-sm text-slate-500">
              <div className="flex items-center justify-center gap-2">
                <svg
                  className="w-4 h-4 text-blue-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122"
                  />
                </svg>
                <span className="font-medium">
                  Click to select • Shift/Ctrl+Click for multiple
                </span>
              </div>
              <div className="flex items-center justify-center gap-2">
                <svg
                  className="w-4 h-4 text-blue-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"
                  />
                </svg>
                <span className="font-medium">
                  Drag to move • Use corner handles to resize
                </span>
              </div>
              <div className="flex items-center justify-center gap-2">
                <svg
                  className="w-4 h-4 text-blue-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                  />
                </svg>
                <span className="font-medium">
                  Drag near corners to rotate fragments
                </span>
              </div>
              <div className="flex items-center justify-center gap-2">
                <svg
                  className="w-4 h-4 text-blue-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                  />
                </svg>
                <span className="font-medium">
                  Use Lock button to prevent accidental changes
                </span>
              </div>
            </div>
            <p className="text-xs text-slate-400 mt-6">
              Click anywhere to dismiss
            </p>
          </div>
        </div>
      )}

      {/* Zoom Controls */}
      <div className="absolute bottom-4 right-4 flex flex-col items-stretch gap-1.5 z-20 select-none">
        <button
          onClick={recenterToFragments}
          className="bg-white/90 backdrop-blur-sm rounded-lg shadow-md border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-100 transition-colors"
        >
          Recenter
        </button>
        <div className="bg-white/90 backdrop-blur-sm rounded-lg shadow-md border border-slate-200 flex items-center gap-1 px-2 py-1.5">
          <button
            onClick={() => zoomToCenter(stageScale / KEYBOARD_ZOOM_FACTOR)}
            className="w-7 h-7 flex items-center justify-center rounded hover:bg-slate-100 text-slate-600 font-bold text-lg"
            title="Zoom out (Ctrl/Cmd + −)"
          >
            −
          </button>
          <button
            onClick={() => {
              setStageScale(1);
              setStagePosition({ x: 0, y: 0 });
            }}
            className="px-2 h-7 flex items-center justify-center rounded hover:bg-slate-100 text-xs font-medium text-slate-700 min-w-[3rem] tabular-nums"
            title="Reset zoom (Ctrl/Cmd + 0)"
          >
            {Math.round(stageScale * 100)}%
          </button>
          <button
            onClick={() => zoomToCenter(stageScale * KEYBOARD_ZOOM_FACTOR)}
            className="w-7 h-7 flex items-center justify-center rounded hover:bg-slate-100 text-slate-600 font-bold text-lg"
            title="Zoom in (Ctrl/Cmd + +)"
          >
            +
          </button>
        </div>
      </div>
    </div>
  );
};

export default Canvas;

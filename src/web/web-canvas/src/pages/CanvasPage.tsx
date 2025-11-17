import { useState, useRef, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import Sidebar from "../components/Sidebar";
import Canvas from "../components/Canvas";
import Toolbar from "../components/Toolbar";
import FilterPanel from "../components/FilterPanel";
import { fragments } from "../utils/fragments";
import { CanvasFragment, ManuscriptFragment } from "../types/fragment";
import { FragmentFilters, DEFAULT_FILTERS } from "../types/filters";
import { filterFragments, getAvailableScripts } from "../utils/filterFragments";

function CanvasPage() {
  const navigate = useNavigate();

  const [canvasFragments, setCanvasFragments] = useState<CanvasFragment[]>([]);
  const [selectedFragmentIds, setSelectedFragmentIds] = useState<string[]>([]);
  const [filters, setFilters] = useState<FragmentFilters>(DEFAULT_FILTERS);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isFilterPanelOpen, setIsFilterPanelOpen] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(280);
  const [filterPanelWidth, setFilterPanelWidth] = useState(300);
  const draggedFragmentRef = useRef<ManuscriptFragment | null>(null);
  const dropPositionRef = useRef<{ x: number; y: number } | null>(null);

  // Filter fragments based on current filters
  const filteredFragments = useMemo(
    () => filterFragments(fragments, filters),
    [filters]
  );

  // Get available script types for filter options
  const availableScripts = useMemo(() => getAvailableScripts(fragments), []);

  // Handle drag start from sidebar
  const handleDragStart = (
    fragment: ManuscriptFragment,
    e: React.DragEvent
  ) => {
    draggedFragmentRef.current = fragment;
    e.dataTransfer.effectAllowed = "copy";
  };

  // Handle drag over canvas
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";

    const rect = e.currentTarget.getBoundingClientRect();
    dropPositionRef.current = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  };

  // Handle drop on canvas
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();

    const fragment = draggedFragmentRef.current;
    const position = dropPositionRef.current;

    if (!fragment || !position) {
      return;
    }

    // Load the image to get its natural dimensions
    const img = new Image();
    img.src = fragment.imagePath;
    img.onload = () => {
      const maxSize = 300;
      let width = img.naturalWidth;
      let height = img.naturalHeight;

      if (width > height) {
        if (width > maxSize) {
          height = (height * maxSize) / width;
          width = maxSize;
        }
      } else {
        if (height > maxSize) {
          width = (width * maxSize) / height;
          height = maxSize;
        }
      }

      const newCanvasFragment: CanvasFragment = {
        id: `canvas-fragment-${Date.now()}-${Math.random()}`,
        fragmentId: fragment.id,
        name: fragment.name,
        imagePath: fragment.imagePath,
        x: position.x - width / 2,
        y: position.y - height / 2,
        width: width,
        height: height,
        rotation: 0,
        scaleX: 1,
        scaleY: 1,
        isLocked: false,
        isSelected: false,
      };

      setCanvasFragments([...canvasFragments, newCanvasFragment]);
    };

    draggedFragmentRef.current = null;
    dropPositionRef.current = null;
  };

  // Lock selected fragments
  const handleLockSelected = () => {
    const updatedFragments = canvasFragments.map((f) =>
      selectedFragmentIds.includes(f.id) ? { ...f, isLocked: true } : f
    );
    setCanvasFragments(updatedFragments);
  };

  // Unlock selected fragments
  const handleUnlockSelected = () => {
    const updatedFragments = canvasFragments.map((f) =>
      selectedFragmentIds.includes(f.id) ? { ...f, isLocked: false } : f
    );
    setCanvasFragments(updatedFragments);
  };

  // Delete selected fragments
  const handleDeleteSelected = () => {
    const updatedFragments = canvasFragments.filter(
      (f) => !selectedFragmentIds.includes(f.id)
    );
    setCanvasFragments(updatedFragments);
    setSelectedFragmentIds([]);
  };

  // Clear all fragments from canvas
  const handleClearCanvas = () => {
    if (canvasFragments.length === 0) return;

    if (
      window.confirm(
        "Are you sure you want to clear all fragments from the canvas?"
      )
    ) {
      setCanvasFragments([]);
      setSelectedFragmentIds([]);
    }
  };

  // Reset view
  const handleResetView = () => {
    setSelectedFragmentIds([]);
  };

  return (
    <div className="flex flex-col h-screen">
      <Toolbar
        selectedCount={selectedFragmentIds.length}
        onLockSelected={handleLockSelected}
        onUnlockSelected={handleUnlockSelected}
        onDeleteSelected={handleDeleteSelected}
        onClearCanvas={handleClearCanvas}
        onResetView={handleResetView}
      />

      <div className="flex flex-1 overflow-hidden relative min-w-0">
        <Sidebar
          fragments={filteredFragments}
          onDragStart={handleDragStart}
          width={sidebarWidth}
          onWidthChange={setSidebarWidth}
          isOpen={isSidebarOpen}
          onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
        />

        <div
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          className="flex-1 h-full min-w-0"
        >
          <Canvas
            fragments={canvasFragments}
            onFragmentsChange={setCanvasFragments}
            selectedFragmentIds={selectedFragmentIds}
            onSelectionChange={setSelectedFragmentIds}
          />
        </div>

        <FilterPanel
          filters={filters}
          onFiltersChange={setFilters}
          availableScripts={availableScripts}
          matchCount={filteredFragments.length}
          totalCount={fragments.length}
          isOpen={isFilterPanelOpen}
          onToggle={() => setIsFilterPanelOpen(!isFilterPanelOpen)}
          width={filterPanelWidth}
          onWidthChange={setFilterPanelWidth}
        />
      </div>
    </div>
  );
}

export default CanvasPage;

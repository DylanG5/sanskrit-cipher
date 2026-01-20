import { useState, useRef, useMemo, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import Sidebar from "../components/Sidebar";
import Canvas from "../components/Canvas";
import Toolbar from "../components/Toolbar";
import FilterPanel from "../components/FilterPanel";
import NotesPanel from "../components/NotesPanel";
import { CanvasFragment, ManuscriptFragment } from "../types/fragment";
import { FragmentFilters, DEFAULT_FILTERS } from "../types/filters";
import { getAllFragments, getFragmentCount } from "../services/fragment-service";
import { isElectron } from "../services/electron-api";

// Default page size for pagination
const PAGE_SIZE = 100;

function CanvasPage() {
  const navigate = useNavigate();

  // Fragment state - now loaded from database
  const [fragments, setFragments] = useState<ManuscriptFragment[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [currentOffset, setCurrentOffset] = useState(0);

  const [canvasFragments, setCanvasFragments] = useState<CanvasFragment[]>([]);
  const [selectedFragmentIds, setSelectedFragmentIds] = useState<string[]>([]);
  const [filters, setFilters] = useState<FragmentFilters>(DEFAULT_FILTERS);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isFilterPanelOpen, setIsFilterPanelOpen] = useState(false);
  const [isNotesPanelOpen, setIsNotesPanelOpen] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(280);
  const [filterPanelWidth, setFilterPanelWidth] = useState(300);
  const [notesPanelWidth, setNotesPanelWidth] = useState(400);
  const [notes, setNotes] = useState("");
  const [isGridVisible, setIsGridVisible] = useState(false);
  const [gridScale, setGridScale] = useState(25); // pixels per cm
  const draggedFragmentRef = useRef<ManuscriptFragment | null>(null);
  const dropPositionRef = useRef<{ x: number; y: number } | null>(null);

  // Load fragments from database (initial load)
  const loadFragments = useCallback(async () => {
    if (!isElectron()) {
      setLoadError("Not running in Electron environment");
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setLoadError(null);

    try {
      // Build API filters from UI filters
      const apiFilters: {
        lineCountMin?: number;
        lineCountMax?: number;
        scripts?: string[];
        isEdgePiece?: boolean | null;
        search?: string;
        limit?: number;
        offset?: number;
      } = {
        limit: PAGE_SIZE,
        offset: 0,
      };

      if (filters.lineCountMin !== undefined) {
        apiFilters.lineCountMin = filters.lineCountMin;
      }
      if (filters.lineCountMax !== undefined) {
        apiFilters.lineCountMax = filters.lineCountMax;
      }
      if (filters.scripts.length > 0) {
        apiFilters.scripts = filters.scripts;
      }
      if (filters.isEdgePiece !== null) {
        apiFilters.isEdgePiece = filters.isEdgePiece;
      }

      // Fetch fragments and count in parallel
      const [fragmentsResult, countResult] = await Promise.all([
        getAllFragments(apiFilters),
        getFragmentCount(apiFilters),
      ]);

      setFragments(fragmentsResult);
      setTotalCount(countResult);
      setCurrentOffset(PAGE_SIZE);
    } catch (error) {
      console.error("Failed to load fragments:", error);
      setLoadError(String(error));
    } finally {
      setIsLoading(false);
    }
  }, [filters]);

  // Load more fragments (infinite scroll)
  const loadMoreFragments = useCallback(async () => {
    if (!isElectron() || isLoadingMore || fragments.length >= totalCount) {
      return;
    }

    setIsLoadingMore(true);

    try {
      const apiFilters: {
        lineCountMin?: number;
        lineCountMax?: number;
        scripts?: string[];
        isEdgePiece?: boolean | null;
        search?: string;
        limit?: number;
        offset?: number;
      } = {
        limit: PAGE_SIZE,
        offset: currentOffset,
      };

      if (filters.lineCountMin !== undefined) {
        apiFilters.lineCountMin = filters.lineCountMin;
      }
      if (filters.lineCountMax !== undefined) {
        apiFilters.lineCountMax = filters.lineCountMax;
      }
      if (filters.scripts.length > 0) {
        apiFilters.scripts = filters.scripts;
      }
      if (filters.isEdgePiece !== null) {
        apiFilters.isEdgePiece = filters.isEdgePiece;
      }

      const moreFragments = await getAllFragments(apiFilters);

      setFragments((prev) => [...prev, ...moreFragments]);
      setCurrentOffset((prev) => prev + PAGE_SIZE);
    } catch (error) {
      console.error("Failed to load more fragments:", error);
    } finally {
      setIsLoadingMore(false);
    }
  }, [isLoadingMore, fragments.length, totalCount, currentOffset, filters]);

  // Load fragments on mount and when filters change
  useEffect(() => {
    loadFragments();
  }, [loadFragments]);

  // Available scripts - empty for now since ML hasn't populated them
  const availableScripts = useMemo(() => [] as string[], []);

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
  // Helper function to calculate display size based on scale data
  const calculateDisplaySize = (
    fragment: ManuscriptFragment,
    originalWidth: number,
    originalHeight: number
  ): { width: number; height: number } => {
    // If no scale data, use default sizing (max 300px)
    if (!fragment.metadata?.scale) {
      const maxSize = 300;
      let width = originalWidth;
      let height = originalHeight;

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

      return { width, height };
    }

    // Use scale data to calculate "actual size" based on grid scale
    const { unit, pixelsPerUnit } = fragment.metadata.scale;
    const MM_TO_CM = 0.1;

    // Calculate physical size in cm
    let widthInCm: number;
    let heightInCm: number;

    if (unit === 'mm') {
      // Convert from mm to cm
      widthInCm = (originalWidth / pixelsPerUnit) * MM_TO_CM;
      heightInCm = (originalHeight / pixelsPerUnit) * MM_TO_CM;
    } else {
      // Already in cm
      widthInCm = originalWidth / pixelsPerUnit;
      heightInCm = originalHeight / pixelsPerUnit;
    }

    // Convert physical size to screen pixels using grid scale
    // gridScale is pixels per cm (default 25)
    const displayWidth = widthInCm * gridScale;
    const displayHeight = heightInCm * gridScale;

    // Apply reasonable limits (min 50px, max 2000px)
    const clampedWidth = Math.max(50, Math.min(2000, Math.round(displayWidth)));
    const clampedHeight = Math.max(50, Math.min(2000, Math.round(displayHeight)));

    return {
      width: clampedWidth,
      height: clampedHeight,
    };
  };

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
      const originalWidth = img.naturalWidth;
      const originalHeight = img.naturalHeight;

      // Calculate display size (auto-scaled if scale data available)
      const { width, height } = calculateDisplaySize(
        fragment,
        originalWidth,
        originalHeight
      );

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

  // Save canvas progress (dummy function)
  const handleSave = () => {
    const canvasData = {
      fragments: canvasFragments,
      notes: notes,
      timestamp: new Date().toISOString(),
    };

    // Create a blob and download as JSON file
    const dataStr = JSON.stringify(canvasData, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `canvas-save-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    alert("Canvas saved successfully!");
  };

  // Toggle notes panel
  const handleToggleNotes = () => {
    setIsNotesPanelOpen(!isNotesPanelOpen);
  };

  // Toggle grid visibility
  const handleToggleGrid = () => {
    setIsGridVisible(!isGridVisible);
  };

  // Handle edge match request (dummy function for now)
  const handleEdgeMatch = (fragmentId: string) => {
    // TODO: Implement edge matching logic to filter sidebar fragments
    // For now, this is just a placeholder
    console.log('Edge match requested for fragment:', fragmentId);
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
        onSave={handleSave}
        onToggleNotes={handleToggleNotes}
        isGridVisible={isGridVisible}
        onToggleGrid={handleToggleGrid}
        isFilterPanelOpen={isFilterPanelOpen}
        onToggleFilters={() => setIsFilterPanelOpen(!isFilterPanelOpen)}
        hasActiveFilters={
          filters.lineCountMin !== undefined ||
          filters.lineCountMax !== undefined ||
          filters.scripts.length > 0 ||
          filters.isEdgePiece !== null
        }
      />

      <div className="flex flex-1 overflow-hidden relative min-w-0">
        <Sidebar
          fragments={fragments}
          onDragStart={handleDragStart}
          width={sidebarWidth}
          onWidthChange={setSidebarWidth}
          isOpen={isSidebarOpen}
          onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
          isLoading={isLoading}
          onLoadMore={loadMoreFragments}
          hasMore={fragments.length < totalCount}
          isLoadingMore={isLoadingMore}
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
            onEdgeMatch={handleEdgeMatch}
            isGridVisible={isGridVisible}
            gridScale={gridScale}
          />
        </div>

        <FilterPanel
          filters={filters}
          onFiltersChange={setFilters}
          availableScripts={availableScripts}
          matchCount={fragments.length}
          totalCount={totalCount}
          isOpen={isFilterPanelOpen}
          onToggle={() => setIsFilterPanelOpen(!isFilterPanelOpen)}
          width={filterPanelWidth}
          onWidthChange={setFilterPanelWidth}
        />

        <NotesPanel
          isOpen={isNotesPanelOpen}
          onToggle={handleToggleNotes}
          width={notesPanelWidth}
          onWidthChange={setNotesPanelWidth}
          notes={notes}
          onNotesChange={setNotes}
        />
      </div>
    </div>
  );
}

export default CanvasPage;

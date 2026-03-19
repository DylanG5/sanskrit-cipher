import { useState, useRef, useMemo, useEffect, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import Sidebar from "../components/Sidebar";
import Canvas from "../components/Canvas";
import Toolbar from "../components/Toolbar";
import FilterPanel from "../components/FilterPanel";
import FragmentMetadata from "../components/FragmentMetadata";
import BulkMetadataEditor from "../components/BulkMetadataEditor";
import UploadDialog from "../components/UploadDialog";
import { CanvasFragment, ManuscriptFragment } from "../types/fragment";
import { FragmentFilters, DEFAULT_FILTERS } from "../types/filters";
import {
  getAllFragments,
  getFragmentCount,
  enrichWithSegmentationStatus,
  getFragmentById,
} from "../services/fragment-service";
import {
  isElectron,
  getElectronAPISafe,
  CanvasFragmentData,
  CanvasStateData,
  EdgeMatchRecord,
} from "../services/electron-api";
import {
  sortBySearchRelevance,
  calculateCenteredPosition,
} from "../utils/fragments";
import { SCRIPT_TYPES, getScriptTypeDB } from "../types/constants";
import { CustomFilterDefinition } from "../types/customFilters";
import {
  getCustomFilters,
  createCustomFilter,
  deleteCustomFilter,
  updateCustomFilterOptions,
} from "../services/custom-filters";

// Default page size for pagination
const PAGE_SIZE = 100;

function CanvasPage() {
  const navigate = useNavigate();
  const location = useLocation();

  // Fragment state - now loaded from database
  const [fragments, setFragments] = useState<ManuscriptFragment[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [currentOffset, setCurrentOffset] = useState(0);

  const [canvasFragments, setCanvasFragments] = useState<CanvasFragment[]>([]);
  const [selectedFragmentIds, setSelectedFragmentIds] = useState<string[]>([]);
  const [selectedMetadataFragment, setSelectedMetadataFragment] =
    useState<ManuscriptFragment | null>(null);
  const [filters, setFilters] = useState<FragmentFilters>(DEFAULT_FILTERS);
  const [customFilters, setCustomFilters] = useState<CustomFilterDefinition[]>(
    [],
  );
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isFilterPanelOpen, setIsFilterPanelOpen] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(280);
  const [filterPanelWidth, setFilterPanelWidth] = useState(300);
  const [isGridVisible, setIsGridVisible] = useState(false);
  const [gridScale, setGridScale] = useState(25); // pixels per cm
  const draggedFragmentRef = useRef<ManuscriptFragment | null>(null);
  const draggedFragmentsRef = useRef<ManuscriptFragment[]>([]);
  const dropPositionRef = useRef<{ x: number; y: number } | null>(null);
  const viewportRef = useRef<{ scale: number; position: { x: number; y: number } }>({ scale: 1, position: { x: 0, y: 0 } });
  const draggedSelectionRef = useRef<ManuscriptFragment[] | null>(null);

  // Edge match state
  const [edgeMatchMode, setEdgeMatchMode] = useState(false);
  const [edgeMatchAnchorId, setEdgeMatchAnchorId] = useState<string | null>(
    null,
  );
  const [edgeMatches, setEdgeMatches] = useState<EdgeMatchRecord[]>([]);

  // Search state
  const [initialSearchQuery, setInitialSearchQuery] = useState<string | null>(
    null,
  );
  const [selectedFragmentId, setSelectedFragmentId] = useState<string | null>(
    null,
  );
  const [isSearchMode, setIsSearchMode] = useState(false);
  const [hasAutoPlaced, setHasAutoPlaced] = useState(false);
  const [sidebarSearchQuery, setSidebarSearchQuery] = useState<string>("");

  // Project state
  const [currentProjectId, setCurrentProjectId] = useState<number | null>(null);
  const [currentProjectName, setCurrentProjectName] = useState<string>("");
  const [saveStatus, setSaveStatus] = useState<"saved" | "saving" | "unsaved">(
    "saved",
  );
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const autoSaveTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isRestoringRef = useRef(false);

  // Upload dialog state
  const [isUploadDialogOpen, setIsUploadDialogOpen] = useState(false);

  // Bulk metadata editor state — holds fragment IDs to edit, or null when closed
  const [bulkEditFragmentIds, setBulkEditFragmentIds] = useState<
    string[] | null
  >(null);

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
        hasTopEdge?: boolean | null;
        hasBottomEdge?: boolean | null;
        hasLeftEdge?: boolean | null;
        hasRightEdge?: boolean | null;
        hasCircle?: boolean | null;
        search?: string;
        custom?: Record<string, string | null | undefined>;
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
        // Convert display values to database values
        apiFilters.scripts = filters.scripts.map(
          (s) => getScriptTypeDB(s) || s,
        );
      }
      if (filters.isEdgePiece !== null) {
        apiFilters.isEdgePiece = filters.isEdgePiece;
      }
      if (filters.hasTopEdge !== null) {
        apiFilters.hasTopEdge = filters.hasTopEdge;
      }
      if (filters.hasBottomEdge !== null) {
        apiFilters.hasBottomEdge = filters.hasBottomEdge;
      }
      if (filters.hasLeftEdge !== null) {
        apiFilters.hasLeftEdge = filters.hasLeftEdge;
      }
      if (filters.hasRightEdge !== null) {
        apiFilters.hasRightEdge = filters.hasRightEdge;
      }
      if (filters.hasCircle !== null) {
        apiFilters.hasCircle = filters.hasCircle;
      }
      if (filters.search) {
        apiFilters.search = filters.search;
        console.log("Loading fragments with search filter:", filters.search);
      }
      if (filters.custom) {
        const custom = Object.entries(filters.custom).reduce<
          Record<string, string | null | undefined>
        >((acc, [key, value]) => {
          if (value !== undefined && value !== null && value !== "") {
            acc[key] = value;
          }
          return acc;
        }, {});
        if (Object.keys(custom).length > 0) {
          apiFilters.custom = custom;
        }
      }

      // Fetch fragments and count in parallel
      const [fragmentsResult, countResult] = await Promise.all([
        getAllFragments(apiFilters, customFilters),
        getFragmentCount(apiFilters),
      ]);

      // Enrich fragments with segmentation status
      const enrichedFragments =
        await enrichWithSegmentationStatus(fragmentsResult);

      setFragments(enrichedFragments);
      setTotalCount(countResult);
      setCurrentOffset(PAGE_SIZE);
    } catch (error) {
      console.error("Failed to load fragments:", error);
      setLoadError(String(error));
    } finally {
      setIsLoading(false);
    }
  }, [filters, customFilters]);

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
        hasTopEdge?: boolean | null;
        hasBottomEdge?: boolean | null;
        hasLeftEdge?: boolean | null;
        hasRightEdge?: boolean | null;
        hasCircle?: boolean | null;
        search?: string;
        custom?: Record<string, string | null | undefined>;
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
        // Convert display values to database values
        apiFilters.scripts = filters.scripts.map(
          (s) => getScriptTypeDB(s) || s,
        );
      }
      if (filters.isEdgePiece !== null) {
        apiFilters.isEdgePiece = filters.isEdgePiece;
      }
      if (filters.hasTopEdge !== null) {
        apiFilters.hasTopEdge = filters.hasTopEdge;
      }
      if (filters.hasBottomEdge !== null) {
        apiFilters.hasBottomEdge = filters.hasBottomEdge;
      }
      if (filters.hasLeftEdge !== null) {
        apiFilters.hasLeftEdge = filters.hasLeftEdge;
      }
      if (filters.hasRightEdge !== null) {
        apiFilters.hasRightEdge = filters.hasRightEdge;
      }
      if (filters.hasCircle !== null) {
        apiFilters.hasCircle = filters.hasCircle;
      }
      if (filters.search) {
        apiFilters.search = filters.search;
      }
      if (filters.custom) {
        const custom = Object.entries(filters.custom).reduce<
          Record<string, string | null | undefined>
        >((acc, [key, value]) => {
          if (value !== undefined && value !== null && value !== "") {
            acc[key] = value;
          }
          return acc;
        }, {});
        if (Object.keys(custom).length > 0) {
          apiFilters.custom = custom;
        }
      }

      const moreFragments = await getAllFragments(apiFilters, customFilters);

      // Enrich with segmentation status
      const enrichedMore = await enrichWithSegmentationStatus(moreFragments);

      setFragments((prev) => [...prev, ...enrichedMore]);
      setCurrentOffset((prev) => prev + PAGE_SIZE);
    } catch (error) {
      console.error("Failed to load more fragments:", error);
    } finally {
      setIsLoadingMore(false);
    }
  }, [
    isLoadingMore,
    fragments.length,
    totalCount,
    currentOffset,
    filters,
    customFilters,
  ]);

  // Refs to track current values for auto-save without re-creating callback
  const canvasFragmentsRef = useRef(canvasFragments);
  const currentProjectIdRef = useRef(currentProjectId);

  // Keep refs in sync
  useEffect(() => {
    canvasFragmentsRef.current = canvasFragments;
  }, [canvasFragments]);

  useEffect(() => {
    currentProjectIdRef.current = currentProjectId;
  }, [currentProjectId]);

  // Auto-save function
  const saveProject = useCallback(
    async (projectId: number, fragmentsToSave: CanvasFragment[]) => {
      const api = getElectronAPISafe();
      if (!api) return;

      setSaveStatus("saving");
      try {
        const canvasState: CanvasStateData = {
          fragments: fragmentsToSave.map((f, index) => ({
            fragmentId: f.fragmentId,
            x: f.x,
            y: f.y,
            width: f.width,
            height: f.height,
            rotation: f.rotation,
            scaleX: f.scaleX,
            scaleY: f.scaleY,
            isLocked: f.isLocked,
            zIndex: index,
            showSegmented: f.showSegmented,
            isMirrored: f.isMirrored,
          })),
        };

        const response = await api.projects.save(projectId, canvasState, "");
        if (response.success) {
          setSaveStatus("saved");
          setHasUnsavedChanges(false);
        } else {
          console.error("Failed to save project:", response.error);
          setSaveStatus("unsaved");
        }
      } catch (error) {
        console.error("Failed to save project:", error);
        setSaveStatus("unsaved");
      }
    },
    [],
  );

  // Debounced auto-save - uses refs to avoid recreating on every state change
  const triggerAutoSave = useCallback(async () => {
    if (isRestoringRef.current) return;

    setHasUnsavedChanges(true);
    setSaveStatus("unsaved");

    // Clear existing timeout
    if (autoSaveTimeoutRef.current) {
      clearTimeout(autoSaveTimeoutRef.current);
    }

    // Set new timeout for auto-save (2 seconds)
    autoSaveTimeoutRef.current = setTimeout(async () => {
      let projectId = currentProjectIdRef.current;

      // Auto-create project if we don't have one yet
      if (!projectId) {
        const api = getElectronAPISafe();
        if (!api) return;

        try {
          const timestamp = new Date().toLocaleString();
          const response = await api.projects.create(
            `Untitled - ${timestamp}`,
            "Manuscript reconstruction",
          );

          if (response.success && response.projectId) {
            projectId = response.projectId;
            setCurrentProjectId(projectId);
            setCurrentProjectName(`Untitled - ${timestamp}`);
            currentProjectIdRef.current = projectId;
          }
        } catch (error) {
          console.error("Failed to auto-create project:", error);
          return;
        }
      }

      if (projectId) {
        saveProject(projectId, canvasFragmentsRef.current);
      }
    }, 2000);
  }, [saveProject]);

  // Create new project if needed
  const ensureProject = useCallback(async (): Promise<number | null> => {
    if (currentProjectId) return currentProjectId;

    const api = getElectronAPISafe();
    if (!api) return null;

    try {
      const timestamp = new Date().toLocaleString();
      const response = await api.projects.create(
        `Untitled - ${timestamp}`,
        "Manuscript reconstruction",
      );

      if (response.success && response.projectId) {
        setCurrentProjectId(response.projectId);
        setCurrentProjectName(`Untitled - ${timestamp}`);
        return response.projectId;
      }
    } catch (error) {
      console.error("Failed to create project:", error);
    }
    return null;
  }, [currentProjectId]);

  // Restore project from loaded data
  const restoreProject = useCallback(
    async (loadedProject: {
      project: { id: number; project_name: string };
      canvasState: { fragments: CanvasFragmentData[] };
      notes: string;
    }) => {
      isRestoringRef.current = true;

      setCurrentProjectId(loadedProject.project.id);
      setCurrentProjectName(loadedProject.project.project_name);

      // Restore canvas fragments - need to load image paths
      const api = getElectronAPISafe();
      if (!api || !loadedProject.canvasState.fragments.length) {
        isRestoringRef.current = false;
        return;
      }

      const restoredFragments: CanvasFragment[] = [];

      for (const frag of loadedProject.canvasState.fragments) {
        try {
          // Get fragment details from database
          const fragmentResponse = await api.fragments.getById(frag.fragmentId);
          if (fragmentResponse.success && fragmentResponse.data) {
            const record = fragmentResponse.data;

            // Determine showSegmented value - default to true if not specified
            const showSegmented =
              frag.showSegmented !== undefined ? frag.showSegmented : true;

            // Always use original image path - segmentation is handled on-demand by the hook
            const imagePath = `electron-image://${record.image_path}`;

            restoredFragments.push({
              id: `canvas-fragment-${Date.now()}-${Math.random()}`,
              fragmentId: frag.fragmentId,
              name: record.fragment_id,
              imagePath: imagePath,
              segmentationCoords: record.segmentation_coords ?? undefined,
              x: frag.x,
              y: frag.y,
              width: frag.width || 200,
              height: frag.height || 200,
              rotation: frag.rotation ?? record.ui_rotation ?? 0,
              scaleX: frag.scaleX,
              scaleY: frag.scaleY,
              isLocked: frag.isLocked,
              isSelected: false,
              showSegmented: showSegmented,
              isMirrored: frag.isMirrored ?? false,
              originalWidth: undefined, // Will be loaded when image loads
              originalHeight: undefined,
              hasBeenResized: false, // Reset on restore
            });
          }
        } catch (error) {
          console.error("Failed to restore fragment:", frag.fragmentId, error);
        }
      }

      setCanvasFragments(restoredFragments);
      setSaveStatus("saved");
      setHasUnsavedChanges(false);

      // Load original image dimensions for restored fragments
      // This is needed for the "Set Scale from Resize" feature to work
      for (const frag of restoredFragments) {
        const img = new Image();
        img.src = frag.imagePath;
        img.onload = () => {
          const originalWidth = img.naturalWidth;
          const originalHeight = img.naturalHeight;

          setCanvasFragments((prev) =>
            prev.map((f) =>
              f.id === frag.id ? { ...f, originalWidth, originalHeight } : f,
            ),
          );
        };
      }

      // Small delay before allowing auto-save
      setTimeout(() => {
        isRestoringRef.current = false;
      }, 500);
    },
    [],
  );

  // Initialize project from location state
  useEffect(() => {
    const projectIdFromLocation = location.state?.projectId;
    const loadedProjectFromLocation = location.state?.loadedProject;

    if (loadedProjectFromLocation) {
      restoreProject(loadedProjectFromLocation);
    } else if (projectIdFromLocation && !loadedProjectFromLocation) {
      setCurrentProjectId(projectIdFromLocation);
    }
  }, [
    location.state?.projectId,
    location.state?.loadedProject,
    restoreProject,
  ]);

  // Trigger auto-save when canvas fragments change
  useEffect(() => {
    // Only auto-save if there's actually content to save
    if (canvasFragments.length > 0) {
      triggerAutoSave();
    }
  }, [canvasFragments, triggerAutoSave]);

  // Cleanup auto-save timeout on unmount
  useEffect(() => {
    return () => {
      if (autoSaveTimeoutRef.current) {
        clearTimeout(autoSaveTimeoutRef.current);
      }
    };
  }, []);

  // Initialize search from location state
  useEffect(() => {
    const searchQueryFromLocation = location.state?.searchQuery;
    const selectedFragmentIdFromLocation = location.state?.selectedFragmentId;

    console.log("Location state received:", {
      searchQuery: searchQueryFromLocation,
      selectedFragmentId: selectedFragmentIdFromLocation,
    });

    // Always apply the search filter to ensure the selected fragment is loaded
    if (searchQueryFromLocation) {
      setInitialSearchQuery(searchQueryFromLocation);
      setIsSearchMode(true);
      setFilters((prev) => ({ ...prev, search: searchQueryFromLocation }));
    }

    if (selectedFragmentIdFromLocation) {
      setSelectedFragmentId(selectedFragmentIdFromLocation);
    }
  }, [location.state?.searchQuery, location.state?.selectedFragmentId]);

  // Load fragments on mount and when filters change
  useEffect(() => {
    loadFragments();
  }, [loadFragments]);

  const loadCustomFilters = useCallback(async () => {
    const list = await getCustomFilters();
    setCustomFilters(list);
  }, []);

  useEffect(() => {
    loadCustomFilters();
  }, [loadCustomFilters]);

  // Available scripts - use the defined script types
  const availableScripts = useMemo(() => [...SCRIPT_TYPES], []);

  // Handle drag start from sidebar (single fragment)
  const handleDragStart = (
    fragment: ManuscriptFragment,
    e: React.DragEvent,
  ) => {
    draggedFragmentRef.current = fragment;
    draggedFragmentsRef.current = [];
    e.dataTransfer.effectAllowed = "copy";
  };

  // Handle multi-drag start from sidebar
  const handleMultiDragStart = (
    selectedFragments: ManuscriptFragment[],
    e: React.DragEvent
  ) => {
    draggedFragmentRef.current = selectedFragments[0];
    draggedFragmentsRef.current = selectedFragments;
    e.dataTransfer.effectAllowed = "copy";
  };

  // When sidebar drag starts with a selection, store all selected fragments
  const handleDragStartSelected = useCallback(
    (selectedFragments: ManuscriptFragment[]) => {
      draggedSelectionRef.current = selectedFragments;
    },
    [],
  );

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
  const calculateDisplaySize = useCallback(
    (
      fragment: ManuscriptFragment,
      originalWidth: number,
      originalHeight: number,
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

      if (unit === "mm") {
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
      const clampedWidth = Math.max(
        50,
        Math.min(2000, Math.round(displayWidth)),
      );
      const clampedHeight = Math.max(
        50,
        Math.min(2000, Math.round(displayHeight)),
      );

      return {
        width: clampedWidth,
        height: clampedHeight,
      };
    },
    [gridScale],
  );

  // Auto-place fragment in center of canvas (for search results)
  const autoPlaceFragment = useCallback(
    async (fragment: ManuscriptFragment) => {
      const img = new Image();
      // Always use original image path - segmentation handled on-demand
      const imagePath = fragment.imagePath;
      img.src = imagePath;

      img.onload = () => {
        const originalWidth = img.naturalWidth;
        const originalHeight = img.naturalHeight;

        // Calculate display size (auto-scaled if scale data available)
        const { width, height } = calculateDisplaySize(
          fragment,
          originalWidth,
          originalHeight,
        );

        // Center on canvas (approximate canvas dimensions)
        const canvasWidth = 1200;
        const canvasHeight = 800;
        const { x, y } = calculateCenteredPosition(
          canvasWidth,
          canvasHeight,
          width,
          height,
        );

        const newCanvasFragment: CanvasFragment = {
          id: `canvas-fragment-${Date.now()}-${Math.random()}`,
          fragmentId: fragment.id,
          name: fragment.name,
          imagePath: imagePath,
          segmentationCoords: fragment.segmentationCoords,
          x,
          y,
          width,
          height,
          rotation: fragment.rotation ?? 0,
          scaleX: 1,
          scaleY: 1,
          isLocked: false,
          isSelected: true, // Select the auto-placed fragment
          showSegmented: true, // Default to showing segmented version
          originalWidth: originalWidth,
          originalHeight: originalHeight,
          hasBeenResized: false,
        };

        setCanvasFragments([newCanvasFragment]);
        setSelectedFragmentIds([newCanvasFragment.id]);
        setHasAutoPlaced(true);
      };

      img.onerror = () => {
        console.error("Failed to load fragment image:", fragment.id);
        setHasAutoPlaced(true); // Mark as attempted even on error
      };
    },
    [],
  );

  // Auto-place first matching fragment when search results load
  useEffect(() => {
    if (isSearchMode && fragments.length > 0 && !hasAutoPlaced) {
      let fragmentToPlace: ManuscriptFragment | null = null;

      console.log("Auto-placement triggered:", {
        selectedFragmentId,
        initialSearchQuery,
        fragmentCount: fragments.length,
        fragmentIds: fragments.slice(0, 10).map((f) => f.id),
      });

      // If user selected a specific fragment from autocomplete, use that
      if (selectedFragmentId) {
        fragmentToPlace =
          fragments.find((f) => f.id === selectedFragmentId) || null;
        console.log("Looking for selected fragment:", selectedFragmentId);
        console.log("Found fragment:", fragmentToPlace?.id);

        if (!fragmentToPlace) {
          // Check if it's a partial match issue
          const partialMatch = fragments.find(
            (f) =>
              f.id.includes(selectedFragmentId) ||
              selectedFragmentId.includes(f.id),
          );
          console.log("Partial match found:", partialMatch?.id);

          // Also check the exact IDs in the array
          const hasExactMatch = fragments.some(
            (f) => f.id === selectedFragmentId,
          );
          console.log("Has exact match in array:", hasExactMatch);
          console.log(
            "Fragment IDs containing search term:",
            fragments
              .filter((f) => f.id.toLowerCase().includes("or11878"))
              .map((f) => f.id),
          );
        }
      }

      // Otherwise, use the first result sorted by relevance
      if (!fragmentToPlace && initialSearchQuery) {
        const sortedFragments = sortBySearchRelevance(
          fragments,
          initialSearchQuery,
        );
        fragmentToPlace = sortedFragments[0] || null;
        console.log("Using first sorted fragment:", fragmentToPlace?.id);
      }

      if (fragmentToPlace) {
        console.log(
          "Auto-placing fragment:",
          fragmentToPlace.id,
          "imagePath:",
          fragmentToPlace.imagePath,
        );
        autoPlaceFragment(fragmentToPlace);
      } else {
        console.log("No fragment to place");
      }
    }
  }, [
    isSearchMode,
    fragments,
    hasAutoPlaced,
    initialSearchQuery,
    selectedFragmentId,
    autoPlaceFragment,
  ]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();

    const position = dropPositionRef.current;
    if (!position) return;

    // Convert screen-space drop coordinates to canvas coordinate space
    const { scale, position: stagePos } = viewportRef.current;
    const canvasPosition = {
      x: (position.x - stagePos.x) / scale,
      y: (position.y - stagePos.y) / scale,
    };

    const multiFragments = draggedFragmentsRef.current;
    const selectedFragments = draggedSelectionRef.current;
    const singleFragment = draggedFragmentRef.current;

    // Prefer multi-drag (from onMultiDragStart), fall back to selection drag, then single
    const fragmentsToDrop =
      multiFragments.length > 1
        ? multiFragments
        : selectedFragments && selectedFragments.length > 1
          ? selectedFragments
          : singleFragment
            ? [singleFragment]
            : [];

    if (fragmentsToDrop.length === 0) return;

    draggedFragmentRef.current = null;
    draggedFragmentsRef.current = [];
    draggedSelectionRef.current = null;
    dropPositionRef.current = null;

    // Load all images in parallel, then place them
    const loadOne = (fragment: ManuscriptFragment, offsetX: number, offsetY: number): Promise<CanvasFragment | null> => {
      return new Promise((resolve) => {
        const img = new Image();
        img.src = fragment.imagePath;
        img.onload = () => {
          const { width, height } = calculateDisplaySize(fragment, img.naturalWidth, img.naturalHeight);
          resolve({
            id: `canvas-fragment-${Date.now()}-${Math.random()}`,
            fragmentId: fragment.id,
            name: fragment.name,
            imagePath: fragment.imagePath,
            segmentationCoords: fragment.segmentationCoords,
            x: canvasPosition.x + offsetX - width / 2,
            y: canvasPosition.y + offsetY - height / 2,
            width,
            height,
            rotation: fragment.rotation ?? 0,
            scaleX: 1,
            scaleY: 1,
            isLocked: false,
            isSelected: false,
            showSegmented: true,
            originalWidth: img.naturalWidth,
            originalHeight: img.naturalHeight,
            hasBeenResized: false,
          });
        };
        img.onerror = () => resolve(null);
      });
    };

    // Stagger fragments so they don't land on top of each other
    const SPREAD = 30; // px offset per fragment
    Promise.all(
      fragmentsToDrop.map((frag, i) => loadOne(frag, i * SPREAD, i * SPREAD))
    ).then((results) => {
      const newFragments = results.filter((f): f is CanvasFragment => f !== null);
      setCanvasFragments(prev => [...prev, ...newFragments]);
    });
  };

  // Handle bulk add to canvas from sidebar multiselect
  const handleBulkAddToCanvas = useCallback(
    (selectedFragments: ManuscriptFragment[]) => {
      const OFFSET = 50; // px offset between staggered fragments
      let loadedCount = 0;
      const newFragments: CanvasFragment[] = [];

      selectedFragments.forEach((fragment, idx) => {
        const img = new Image();
        img.src = fragment.imagePath;
        img.onload = () => {
          const { width, height } = calculateDisplaySize(
            fragment,
            img.naturalWidth,
            img.naturalHeight,
          );
          const newFrag: CanvasFragment = {
            id: `canvas-fragment-${Date.now()}-${idx}-${Math.random()}`,
            fragmentId: fragment.id,
            name: fragment.name,
            imagePath: fragment.imagePath,
            segmentationCoords: fragment.segmentationCoords,
            x: 100 + idx * OFFSET,
            y: 100 + idx * OFFSET,
            width,
            height,
            rotation: fragment.rotation ?? 0,
            scaleX: 1,
            scaleY: 1,
            isLocked: false,
            isSelected: false,
            showSegmented: true,
            originalWidth: img.naturalWidth,
            originalHeight: img.naturalHeight,
            hasBeenResized: false,
          };
          newFragments.push(newFrag);
          loadedCount++;
          if (loadedCount === selectedFragments.length) {
            setCanvasFragments((prev) => [...prev, ...newFragments]);
          }
        };
      });
    },
    [calculateDisplaySize, gridScale],
  );

  // Handle bulk edit metadata from sidebar multiselect
  const handleBulkEditSidebarMetadata = useCallback((fragmentIds: string[]) => {
    if (fragmentIds.length >= 2) {
      setBulkEditFragmentIds(fragmentIds);
    }
  }, []);

  // Clear search and reset to normal mode
  const handleClearSearch = useCallback(() => {
    setIsSearchMode(false);
    setInitialSearchQuery(null);
    setSelectedFragmentId(null);
    setFilters((prev) => ({ ...prev, search: undefined }));
    setHasAutoPlaced(false);
  }, []);

  // Handle sidebar search
  const handleSidebarSearch = useCallback((query: string) => {
    setSidebarSearchQuery(query);
    // Update filters to trigger database search
    setFilters((prev) => ({ ...prev, search: query || undefined }));
    setCurrentOffset(0); // Reset pagination
  }, []);


  // Lock selected fragments
  const handleLockSelected = () => {
    const updatedFragments = canvasFragments.map((f) =>
      selectedFragmentIds.includes(f.id) ? { ...f, isLocked: true } : f,
    );
    setCanvasFragments(updatedFragments);
  };

  // Unlock selected fragments
  const handleUnlockSelected = () => {
    const updatedFragments = canvasFragments.map((f) =>
      selectedFragmentIds.includes(f.id) ? { ...f, isLocked: false } : f,
    );
    setCanvasFragments(updatedFragments);
  };

  // Rotate selected fragments by 180 degrees
  const handleRotate180Selected = () => {
    if (selectedFragmentIds.length === 0) return;

    const selectedSet = new Set(selectedFragmentIds);
    setCanvasFragments((prev) => {
      const nextCanvas = prev.map((fragment) => {
        if (!selectedSet.has(fragment.id)) return fragment;

        const currentRotationDeg = fragment.rotation ?? 0;
        const currentRotationRad = (currentRotationDeg * Math.PI) / 180;
        const cos = Math.cos(currentRotationRad);
        const sin = Math.sin(currentRotationRad);

        // Rotate around visual center (not top-left anchor).
        // Konva rotates around the node origin, so shifting origin by R(theta) * (w, h)
        // before adding 180° preserves center position.
        const effectiveWidth = (fragment.width ?? 0) * (fragment.scaleX ?? 1);
        const effectiveHeight = (fragment.height ?? 0) * (fragment.scaleY ?? 1);
        const dx = effectiveWidth * cos - effectiveHeight * sin;
        const dy = effectiveWidth * sin + effectiveHeight * cos;

        const nextRotation = ((fragment.rotation ?? 0) + 180) % 360;
        return {
          ...fragment,
          x: fragment.x + dx,
          y: fragment.y + dy,
          rotation: nextRotation,
        };
      });

      const rotationByFragmentId: Record<string, number> = {};
      for (const fragment of nextCanvas) {
        if (selectedSet.has(fragment.id)) {
          rotationByFragmentId[fragment.fragmentId] = fragment.rotation ?? 0;
        }
      }

      setFragments((prevFragments) =>
        prevFragments.map((fragment) => {
          const updated = rotationByFragmentId[fragment.id];
          return updated === undefined
            ? fragment
            : { ...fragment, rotation: updated };
        }),
      );

      const api = getElectronAPISafe();
      if (api && Object.keys(rotationByFragmentId).length > 0) {
        api.fragments.bulkUpdateRotation(rotationByFragmentId).catch((error) => {
          console.error('Failed to persist fragment rotation:', error);
        });
      }

      return nextCanvas;
    });
  };

  // Bring selected fragments to top of render stack
  const handleBringToFront = () => {
    if (selectedFragmentIds.length === 0) return;
    const selectedSet = new Set(selectedFragmentIds);
    const selected = canvasFragments.filter((f) => selectedSet.has(f.id));
    const rest = canvasFragments.filter((f) => !selectedSet.has(f.id));
    setCanvasFragments([...rest, ...selected]);
  };

  // Send selected fragments to bottom of render stack
  const handleSendToBack = () => {
    if (selectedFragmentIds.length === 0) return;
    const selectedSet = new Set(selectedFragmentIds);
    const selected = canvasFragments.filter((f) => selectedSet.has(f.id));
    const rest = canvasFragments.filter((f) => !selectedSet.has(f.id));
    setCanvasFragments([...selected, ...rest]);
  };

  // Assign a shared group ID so fragments can be selected and moved together
  const handleGroupSelected = () => {
    if (selectedFragmentIds.length < 2) return;
    const selectedSet = new Set(selectedFragmentIds);
    const newGroupId = `group-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    setCanvasFragments((prev) =>
      prev.map((fragment) =>
        selectedSet.has(fragment.id)
          ? { ...fragment, groupId: newGroupId }
          : fragment,
      ),
    );
  };

  // Ungroup any selected groups
  const handleUngroupSelected = () => {
    if (selectedFragmentIds.length === 0) return;
    const selectedSet = new Set(selectedFragmentIds);
    const selectedGroupIds = new Set(
      canvasFragments
        .filter((fragment) => selectedSet.has(fragment.id) && fragment.groupId)
        .map((fragment) => fragment.groupId as string),
    );

    if (selectedGroupIds.size === 0) return;

    setCanvasFragments((prev) =>
      prev.map((fragment) =>
        fragment.groupId && selectedGroupIds.has(fragment.groupId)
          ? { ...fragment, groupId: undefined }
          : fragment,
      ),
    );
  };

  // Mirror/flip selected fragment horizontally
  const handleMirrorSelected = () => {
    if (selectedFragmentIds.length !== 1) return;
    const selectedId = selectedFragmentIds[0];
    setCanvasFragments(prev =>
      prev.map(f =>
        f.id === selectedId ? { ...f, isMirrored: !f.isMirrored } : f
      )
    );
    setHasUnsavedChanges(true);
  };

  // Toggle segmentation for selected fragment
  const handleToggleSelectedFragmentSegmentation = async () => {
    if (selectedFragmentIds.length !== 1) return;

    const selectedId = selectedFragmentIds[0];
    const canvasFragment = canvasFragments.find((f) => f.id === selectedId);
    if (!canvasFragment) return;

    const newShowSegmented = !canvasFragment.showSegmented;

    // Simply toggle the flag - the useFragmentImage hook will handle loading
    // the appropriate version (segmented or original) based on this flag
    const updatedFragments = canvasFragments.map((f) =>
      f.id === selectedId
        ? {
            ...f,
            showSegmented: newShowSegmented,
          }
        : f,
    );
    setCanvasFragments(updatedFragments);
    setHasUnsavedChanges(true);
  };

  // Delete selected fragments
  const handleDeleteSelected = () => {
    const updatedFragments = canvasFragments.filter(
      (f) => !selectedFragmentIds.includes(f.id),
    );
    setCanvasFragments(updatedFragments);
    setSelectedFragmentIds([]);
  };

  // Clear all fragments from canvas
  const handleClearCanvas = () => {
    if (canvasFragments.length === 0) return;

    if (
      window.confirm(
        "Are you sure you want to clear all fragments from the canvas?",
      )
    ) {
      setCanvasFragments([]);
      setSelectedFragmentIds([]);
    }
  };

  // Save canvas progress to database
  const handleSave = async () => {
    // Cancel any pending auto-save
    if (autoSaveTimeoutRef.current) {
      clearTimeout(autoSaveTimeoutRef.current);
    }

    // Ensure we have a project
    const projectId = await ensureProject();
    if (!projectId) {
      console.error("Failed to create or get project");
      return;
    }

    await saveProject(projectId, canvasFragments);
  };

  // Toggle grid visibility
  const handleToggleGrid = () => {
    setIsGridVisible(!isGridVisible);
  };

  // Handle upload dialog
  const handleUploadClick = () => {
    setIsUploadDialogOpen(true);
  };

  const handleUploadComplete = () => {
    // Refresh fragment list to show newly uploaded fragments
    loadFragments();
  };

  const handleCreateCustomFilter = useCallback(
    async (payload: {
      label: string;
      type: "dropdown" | "text";
      options?: string[];
    }) => {
      const created = await createCustomFilter(payload);
      if (created) {
        setCustomFilters((prev) => [...prev, created]);
      }
      return created;
    },
    [],
  );

  const handleDeleteCustomFilter = useCallback(
    async (id: number) => {
      const target = customFilters.find((filter) => filter.id === id);
      if (!target) {
        return false;
      }

      const success = await deleteCustomFilter(id);
      if (success) {
        setCustomFilters((prev) => prev.filter((filter) => filter.id !== id));
        setFilters((prev) => {
          if (!prev.custom || !(target.filterKey in prev.custom)) {
            return prev;
          }
          const nextCustom = { ...prev.custom };
          delete nextCustom[target.filterKey];
          return { ...prev, custom: nextCustom };
        });
      }
      return success;
    },
    [customFilters],
  );

  const handleUpdateCustomFilterOptions = useCallback(
    async (id: number, options: string[]) => {
      const updated = await updateCustomFilterOptions(id, options);
      if (updated) {
        setCustomFilters((prev) =>
          prev.map((filter) => (filter.id === id ? updated : filter)),
        );
      }
      return updated;
    },
    [],
  );

  // Handle edge match request: query DB for matches and switch sidebar to match view
  const handleEdgeMatch = async (fragmentId: string) => {
    const api = getElectronAPISafe();
    if (!api?.edgeMatches) {
      console.warn("Edge match API not available");
      return;
    }

    // Check if edge match data exists
    const check = await api.edgeMatches.hasData();
    if (!check.hasData) {
      console.log(
        "No edge match data in database. Run the reconstruction pipeline first.",
      );
      return;
    }

    // Fetch best matches
    const result = await api.edgeMatches.getBestMatches(fragmentId, 20);
    if (result.success && result.data) {
      setEdgeMatchMode(true);
      setEdgeMatchAnchorId(fragmentId);
      setEdgeMatches(result.data);
    } else {
      setEdgeMatchMode(false);
      setEdgeMatchAnchorId(null);
      setEdgeMatches([]);
    }
  };

  // Handle placing a matched fragment on the canvas
  const handlePlaceEdgeMatch = async (match: EdgeMatchRecord) => {
    if (!edgeMatchAnchorId) return;

    // Find the anchor fragment's position on canvas
    const anchor = canvasFragments.find(
      (cf) => cf.fragmentId === edgeMatchAnchorId,
    );
    if (!anchor) {
      console.warn("Anchor fragment not on canvas");
      return;
    }

    // Convert cm offset to canvas pixels
    const relPx = (match.relative_x_cm ?? 0) * gridScale;
    const relPy = (match.relative_y_cm ?? 0) * gridScale;

    // Rotate the offset vector by the anchor's current rotation
    // so placement is correct even if the anchor has been rotated on canvas
    const anchorRotRad = ((anchor.rotation ?? 0) * Math.PI) / 180;
    const cosA = Math.cos(anchorRotRad);
    const sinA = Math.sin(anchorRotRad);
    const newX = anchor.x + relPx * cosA - relPy * sinA;
    const newY = anchor.y + relPx * sinA + relPy * cosA;

    // Combine pipeline rotation with anchor's canvas rotation
    const rotation = (match.rotation_deg ?? 0) + (anchor.rotation ?? 0);

    // Check if already on canvas
    const existing = canvasFragments.find(
      (cf) => cf.fragmentId === match.fragment_b_id,
    );
    if (existing) {
      // Update position of existing fragment
      setCanvasFragments((prev) =>
        prev.map((cf) =>
          cf.fragmentId === match.fragment_b_id
            ? { ...cf, x: newX, y: newY, rotation, showSegmented: true }
            : cf,
        ),
      );
      return;
    }

    // Load fragment image to get display size
    const api = getElectronAPISafe();
    if (!api) return;

    // Fetch the fragment from DB to get metadata for sizing
    const fragResult = await api.fragments.getById(match.fragment_b_id);
    if (!fragResult.success || !fragResult.data) return;
    const fragData = fragResult.data;

    // Build resolved image path using electron-image:// protocol (consistent with drag-drop and restore)
    const imagePath = `electron-image://${fragData.image_path}`;
    const img = new Image();
    img.src = imagePath;
    await new Promise<void>((resolve) => {
      img.onload = () => resolve();
      img.onerror = () => resolve();
    });

    // Build a ManuscriptFragment-like object for calculateDisplaySize
    const scaleInfo = fragData.pixels_per_unit
      ? {
          unit: (fragData.scale_unit ?? "cm") as "cm" | "mm",
          pixelsPerUnit: fragData.pixels_per_unit,
        }
      : undefined;
    const mockFragment: ManuscriptFragment = {
      id: match.fragment_b_id,
      name: match.fragment_b_id,
      imagePath: imagePath,
      thumbnailPath: imagePath,
      metadata: scaleInfo
        ? { scale: { ...scaleInfo, detectionStatus: "success" as const } }
        : undefined,
    };
    const { width, height } = calculateDisplaySize(
      mockFragment,
      img.naturalWidth || 300,
      img.naturalHeight || 300,
    );

    const newFragment: CanvasFragment = {
      id: `canvas-fragment-${Date.now()}-${Math.random()}`,
      fragmentId: match.fragment_b_id,
      name: match.fragment_b_id,
      imagePath: imagePath,
      segmentationCoords: fragData.segmentation_coords ?? undefined,
      x: newX,
      y: newY,
      width,
      height,
      rotation,
      scaleX: 1,
      scaleY: 1,
      isLocked: false,
      isSelected: false,
      showSegmented: true,
    };

    setCanvasFragments((prev) => [...prev, newFragment]);
  };

  // Exit edge match mode
  const handleExitEdgeMatch = () => {
    setEdgeMatchMode(false);
    setEdgeMatchAnchorId(null);
    setEdgeMatches([]);
  };

  // State to track the canvas fragment for metadata display
  const [
    selectedCanvasFragmentForMetadata,
    setSelectedCanvasFragmentForMetadata,
  ] = useState<CanvasFragment | null>(null);

  // Handle double-click on canvas fragment to show metadata
  const handleCanvasFragmentDoubleClick = async (fragmentId: string) => {
    // Find the canvas fragment
    const canvasFragment = canvasFragments.find(
      (cf) => cf.fragmentId === fragmentId,
    );

    // First try to find in already loaded fragments
    let manuscriptFragment = fragments.find((f) => f.id === fragmentId);

    // If not found in loaded fragments, fetch from database
    if (!manuscriptFragment) {
      manuscriptFragment = await getFragmentById(fragmentId, customFilters);
    }

    if (manuscriptFragment) {
      setSelectedMetadataFragment(manuscriptFragment);
      setSelectedCanvasFragmentForMetadata(canvasFragment || null);
    }
  };

  // Close metadata panel
  const handleCloseMetadata = () => {
    setSelectedMetadataFragment(null);
    setSelectedCanvasFragmentForMetadata(null);
  };

  // Get selected fragment info for segmentation toggle
  const selectedFragment =
    selectedFragmentIds.length === 1
      ? canvasFragments.find((f) => f.id === selectedFragmentIds[0])
      : undefined;

  // Find the corresponding ManuscriptFragment (used for metadata modal, scale info, etc.)
  const selectedManuscriptFragment = selectedFragment
    ? fragments.find((f) => f.id === selectedFragment.fragmentId)
    : undefined;

  // Derive hasSegmentation directly from the canvas fragment's stored coords so it
  // works even when the fragment isn't in the current sidebar page.
  const selectedFragmentHasSegmentation = selectedFragment
    ? Boolean(selectedFragment.segmentationCoords)
    : false;

  const canUngroupSelected = selectedFragmentIds.some((id) => {
    const fragment = canvasFragments.find((f) => f.id === id);
    return Boolean(fragment?.groupId);
  });

  return (
    <div className="flex flex-col h-screen w-full overflow-hidden">
      <Toolbar
        selectedCount={selectedFragmentIds.length}
        onLockSelected={handleLockSelected}
        onUnlockSelected={handleUnlockSelected}
        onRotate180Selected={handleRotate180Selected}
        onBringToFront={handleBringToFront}
        onSendToBack={handleSendToBack}
        onGroupSelected={handleGroupSelected}
        onUngroupSelected={handleUngroupSelected}
        canGroupSelected={selectedFragmentIds.length >= 2}
        canUngroupSelected={canUngroupSelected}
        onDeleteSelected={handleDeleteSelected}
        onClearCanvas={handleClearCanvas}
        onSave={handleSave}
        onUploadClick={handleUploadClick}
        isGridVisible={isGridVisible}
        onToggleGrid={handleToggleGrid}
        isFilterPanelOpen={isFilterPanelOpen}
        onToggleFilters={() => setIsFilterPanelOpen(!isFilterPanelOpen)}
        hasActiveFilters={
          filters.lineCountMin !== undefined ||
          filters.lineCountMax !== undefined ||
          filters.scripts.length > 0 ||
          filters.isEdgePiece !== null ||
          filters.hasTopEdge !== null ||
          filters.hasBottomEdge !== null ||
          filters.hasLeftEdge !== null ||
          filters.hasRightEdge !== null ||
          filters.hasCircle !== null ||
          (filters.search !== undefined && filters.search.trim().length > 0) ||
          Object.values(filters.custom || {}).some(
            (value) => value !== undefined && value !== null && value !== "",
          )
        }
        selectedFragmentHasSegmentation={selectedFragmentHasSegmentation}
        selectedFragmentShowSegmented={selectedFragment?.showSegmented}
        onToggleSelectedFragmentSegmentation={
          handleToggleSelectedFragmentSegmentation
        }
        onBulkEditMetadata={() => {
          const ids = selectedFragmentIds.map(
            (id) =>
              canvasFragments.find((cf) => cf.id === id)?.fragmentId ?? id,
          );
          setBulkEditFragmentIds(ids);
        }}
        selectedFragmentIsMirrored={selectedFragment?.isMirrored}
        onMirrorSelected={handleMirrorSelected}
        projectName={currentProjectName}
        saveStatus={saveStatus}
      />

      <div className="flex flex-1 overflow-hidden relative min-w-0">
        <Sidebar
          fragments={fragments}
          onDragStart={handleDragStart}
          onMultiDragStart={handleMultiDragStart}
          width={sidebarWidth}
          onWidthChange={setSidebarWidth}
          isOpen={isSidebarOpen}
          onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
          isLoading={isLoading}
          onLoadMore={loadMoreFragments}
          hasMore={fragments.length < totalCount}
          isLoadingMore={isLoadingMore}
          searchQuery={initialSearchQuery}
          onClearSearch={handleClearSearch}
          onSidebarSearch={handleSidebarSearch}
          onFragmentUpdate={loadFragments}
          canvasFragments={canvasFragments}
          gridScale={gridScale}
          customFilters={customFilters}
          edgeMatchMode={edgeMatchMode}
          edgeMatchAnchorId={edgeMatchAnchorId}
          edgeMatches={edgeMatches}
          onPlaceEdgeMatch={handlePlaceEdgeMatch}
          onExitEdgeMatch={handleExitEdgeMatch}
          onBulkAddToCanvas={handleBulkAddToCanvas}
          onBulkEditSidebarMetadata={handleBulkEditSidebarMetadata}
          onDragStartSelected={handleDragStartSelected}
          onFragmentsDeleted={loadFragments}
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
            onFragmentDoubleClick={handleCanvasFragmentDoubleClick}
            isGridVisible={isGridVisible}
            gridScale={gridScale}
            onViewportChange={(scale, position) => {
              viewportRef.current = { scale, position };
            }}
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
          customFilters={customFilters}
          onCreateCustomFilter={handleCreateCustomFilter}
          onDeleteCustomFilter={handleDeleteCustomFilter}
          onUpdateCustomFilterOptions={handleUpdateCustomFilterOptions}
        />
      </div>

      {/* Fragment Metadata Modal */}
      {selectedMetadataFragment && (
        <FragmentMetadata
          fragment={selectedMetadataFragment}
          onClose={handleCloseMetadata}
          onUpdate={loadFragments}
          canvasFragment={selectedCanvasFragmentForMetadata}
          gridScale={gridScale}
          customFilters={customFilters}
        />
      )}

      {/* Bulk Metadata Editor Modal */}
      {bulkEditFragmentIds && bulkEditFragmentIds.length >= 2 && (
        <BulkMetadataEditor
          fragmentIds={bulkEditFragmentIds}
          onClose={() => setBulkEditFragmentIds(null)}
          onUpdate={loadFragments}
          customFilters={customFilters}
        />
      )}

      {/* Upload Dialog */}
      <UploadDialog
        isOpen={isUploadDialogOpen}
        onClose={() => setIsUploadDialogOpen(false)}
        onUploadComplete={handleUploadComplete}
      />
    </div>
  );
}

export default CanvasPage;

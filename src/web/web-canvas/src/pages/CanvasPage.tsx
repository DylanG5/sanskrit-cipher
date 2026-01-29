import { useState, useRef, useMemo, useEffect, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import Sidebar from "../components/Sidebar";
import Canvas from "../components/Canvas";
import Toolbar from "../components/Toolbar";
import FilterPanel from "../components/FilterPanel";
import NotesPanel from "../components/NotesPanel";
import FragmentMetadata from "../components/FragmentMetadata";
import { CanvasFragment, ManuscriptFragment } from "../types/fragment";
import { FragmentFilters, DEFAULT_FILTERS } from "../types/filters";
import { getAllFragments, getFragmentCount, enrichWithSegmentationStatus, getFragmentById } from "../services/fragment-service";
import { isElectron, getElectronAPISafe, CanvasFragmentData, CanvasStateData } from "../services/electron-api";
import { sortBySearchRelevance, calculateCenteredPosition } from "../utils/fragments";

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
  const [selectedMetadataFragment, setSelectedMetadataFragment] = useState<ManuscriptFragment | null>(null);
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

  // Search state
  const [initialSearchQuery, setInitialSearchQuery] = useState<string | null>(null);
  const [selectedFragmentId, setSelectedFragmentId] = useState<string | null>(null);
  const [isSearchMode, setIsSearchMode] = useState(false);
  const [hasAutoPlaced, setHasAutoPlaced] = useState(false);
  const [sidebarSearchQuery, setSidebarSearchQuery] = useState<string>('');

  // Project state
  const [currentProjectId, setCurrentProjectId] = useState<number | null>(null);
  const [currentProjectName, setCurrentProjectName] = useState<string>('');
  const [saveStatus, setSaveStatus] = useState<'saved' | 'saving' | 'unsaved'>('saved');
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const autoSaveTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isRestoringRef = useRef(false);

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
      if (filters.search) {
        apiFilters.search = filters.search;
        console.log('Loading fragments with search filter:', filters.search);
      }

      // Fetch fragments and count in parallel
      const [fragmentsResult, countResult] = await Promise.all([
        getAllFragments(apiFilters),
        getFragmentCount(apiFilters),
      ]);

      // Enrich fragments with segmentation status
      const enrichedFragments = await enrichWithSegmentationStatus(fragmentsResult);

      setFragments(enrichedFragments);
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

      // Enrich with segmentation status
      const enrichedMore = await enrichWithSegmentationStatus(moreFragments);

      setFragments((prev) => [...prev, ...enrichedMore]);
      setCurrentOffset((prev) => prev + PAGE_SIZE);
    } catch (error) {
      console.error("Failed to load more fragments:", error);
    } finally {
      setIsLoadingMore(false);
    }
  }, [isLoadingMore, fragments.length, totalCount, currentOffset, filters]);

  // Refs to track current values for auto-save without re-creating callback
  const canvasFragmentsRef = useRef(canvasFragments);
  const notesRef = useRef(notes);
  const currentProjectIdRef = useRef(currentProjectId);

  // Keep refs in sync
  useEffect(() => {
    canvasFragmentsRef.current = canvasFragments;
  }, [canvasFragments]);

  useEffect(() => {
    notesRef.current = notes;
  }, [notes]);

  useEffect(() => {
    currentProjectIdRef.current = currentProjectId;
  }, [currentProjectId]);

  // Auto-save function
  const saveProject = useCallback(async (projectId: number, fragmentsToSave: CanvasFragment[], notesToSave: string) => {
    const api = getElectronAPISafe();
    if (!api) return;

    setSaveStatus('saving');
    try {
      const canvasState: CanvasStateData = {
        fragments: fragmentsToSave.map(f => ({
          fragmentId: f.fragmentId,
          x: f.x,
          y: f.y,
          width: f.width,
          height: f.height,
          rotation: f.rotation,
          scaleX: f.scaleX,
          scaleY: f.scaleY,
          isLocked: f.isLocked,
          zIndex: 0,
        })),
      };

      const response = await api.projects.save(projectId, canvasState, notesToSave);
      if (response.success) {
        setSaveStatus('saved');
        setHasUnsavedChanges(false);
      } else {
        console.error('Failed to save project:', response.error);
        setSaveStatus('unsaved');
      }
    } catch (error) {
      console.error('Failed to save project:', error);
      setSaveStatus('unsaved');
    }
  }, []);

  // Debounced auto-save - uses refs to avoid recreating on every state change
  const triggerAutoSave = useCallback(async () => {
    if (isRestoringRef.current) return;

    setHasUnsavedChanges(true);
    setSaveStatus('unsaved');

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
            'Manuscript reconstruction'
          );

          if (response.success && response.projectId) {
            projectId = response.projectId;
            setCurrentProjectId(projectId);
            setCurrentProjectName(`Untitled - ${timestamp}`);
            currentProjectIdRef.current = projectId;
          }
        } catch (error) {
          console.error('Failed to auto-create project:', error);
          return;
        }
      }

      if (projectId) {
        saveProject(projectId, canvasFragmentsRef.current, notesRef.current);
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
        'Manuscript reconstruction'
      );

      if (response.success && response.projectId) {
        setCurrentProjectId(response.projectId);
        setCurrentProjectName(`Untitled - ${timestamp}`);
        return response.projectId;
      }
    } catch (error) {
      console.error('Failed to create project:', error);
    }
    return null;
  }, [currentProjectId]);

  // Restore project from loaded data
  const restoreProject = useCallback(async (loadedProject: {
    project: { id: number; project_name: string };
    canvasState: { fragments: CanvasFragmentData[] };
    notes: string;
  }) => {
    isRestoringRef.current = true;

    setCurrentProjectId(loadedProject.project.id);
    setCurrentProjectName(loadedProject.project.project_name);
    setNotes(loadedProject.notes || '');

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
          // Use electron-image protocol, same as fragment-service.ts
          const imagePath = `electron-image://${record.image_path}`;

          restoredFragments.push({
            id: `canvas-fragment-${Date.now()}-${Math.random()}`,
            fragmentId: frag.fragmentId,
            name: record.fragment_id,
            imagePath: imagePath,
            x: frag.x,
            y: frag.y,
            width: frag.width || 200,
            height: frag.height || 200,
            rotation: frag.rotation,
            scaleX: frag.scaleX,
            scaleY: frag.scaleY,
            isLocked: frag.isLocked,
            isSelected: false,
          });
        }
      } catch (error) {
        console.error('Failed to restore fragment:', frag.fragmentId, error);
      }
    }

    setCanvasFragments(restoredFragments);
    setSaveStatus('saved');
    setHasUnsavedChanges(false);

    // Small delay before allowing auto-save
    setTimeout(() => {
      isRestoringRef.current = false;
    }, 500);
  }, []);

  // Initialize project from location state
  useEffect(() => {
    const projectIdFromLocation = location.state?.projectId;
    const loadedProjectFromLocation = location.state?.loadedProject;

    if (loadedProjectFromLocation) {
      restoreProject(loadedProjectFromLocation);
    } else if (projectIdFromLocation && !loadedProjectFromLocation) {
      setCurrentProjectId(projectIdFromLocation);
    }
  }, [location.state?.projectId, location.state?.loadedProject, restoreProject]);

  // Trigger auto-save when canvas fragments or notes change
  useEffect(() => {
    // Only auto-save if there's actually content to save
    if (canvasFragments.length > 0 || notes) {
      triggerAutoSave();
    }
  }, [canvasFragments, notes, triggerAutoSave]);

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

    console.log('Location state received:', {
      searchQuery: searchQueryFromLocation,
      selectedFragmentId: selectedFragmentIdFromLocation
    });

    // Always apply the search filter to ensure the selected fragment is loaded
    if (searchQueryFromLocation) {
      setInitialSearchQuery(searchQueryFromLocation);
      setIsSearchMode(true);
      setFilters(prev => ({ ...prev, search: searchQueryFromLocation }));
    }

    if (selectedFragmentIdFromLocation) {
      setSelectedFragmentId(selectedFragmentIdFromLocation);
    }
  }, [location.state?.searchQuery, location.state?.selectedFragmentId]);

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

  // Auto-place fragment in center of canvas (for search results)
  const autoPlaceFragment = useCallback(async (fragment: ManuscriptFragment) => {
    const img = new Image();
    // Default to segmented if available, otherwise use original
    const showSegmented = true; // Default preference
    const imagePath = (showSegmented && fragment.segmentedImagePath)
      ? fragment.segmentedImagePath
      : fragment.imagePath;
    img.src = imagePath;

    img.onload = () => {
      const originalWidth = img.naturalWidth;
      const originalHeight = img.naturalHeight;

      // Calculate display size (auto-scaled if scale data available)
      const { width, height } = calculateDisplaySize(
        fragment,
        originalWidth,
        originalHeight
      );

      // Center on canvas (approximate canvas dimensions)
      const canvasWidth = 1200;
      const canvasHeight = 800;
      const { x, y } = calculateCenteredPosition(
        canvasWidth,
        canvasHeight,
        width,
        height
      );

      const newCanvasFragment: CanvasFragment = {
        id: `canvas-fragment-${Date.now()}-${Math.random()}`,
        fragmentId: fragment.id,
        name: fragment.name,
        imagePath: imagePath,
        x,
        y,
        width,
        height,
        rotation: 0,
        scaleX: 1,
        scaleY: 1,
        isLocked: false,
        isSelected: true, // Select the auto-placed fragment
        showSegmented: true, // Default to showing segmented version
      };

      setCanvasFragments([newCanvasFragment]);
      setSelectedFragmentIds([newCanvasFragment.id]);
      setHasAutoPlaced(true);
    };

    img.onerror = () => {
      console.error('Failed to load fragment image:', fragment.id);
      setHasAutoPlaced(true); // Mark as attempted even on error
    };
  }, []);

  // Auto-place first matching fragment when search results load
  useEffect(() => {
    if (isSearchMode && fragments.length > 0 && !hasAutoPlaced) {
      let fragmentToPlace: ManuscriptFragment | null = null;

      console.log('Auto-placement triggered:', {
        selectedFragmentId,
        initialSearchQuery,
        fragmentCount: fragments.length,
        fragmentIds: fragments.slice(0, 10).map(f => f.id)
      });

      // If user selected a specific fragment from autocomplete, use that
      if (selectedFragmentId) {
        fragmentToPlace = fragments.find(f => f.id === selectedFragmentId) || null;
        console.log('Looking for selected fragment:', selectedFragmentId);
        console.log('Found fragment:', fragmentToPlace?.id);

        if (!fragmentToPlace) {
          // Check if it's a partial match issue
          const partialMatch = fragments.find(f =>
            f.id.includes(selectedFragmentId) || selectedFragmentId.includes(f.id)
          );
          console.log('Partial match found:', partialMatch?.id);

          // Also check the exact IDs in the array
          const hasExactMatch = fragments.some(f => f.id === selectedFragmentId);
          console.log('Has exact match in array:', hasExactMatch);
          console.log('Fragment IDs containing search term:',
            fragments.filter(f => f.id.toLowerCase().includes('or11878')).map(f => f.id)
          );
        }
      }

      // Otherwise, use the first result sorted by relevance
      if (!fragmentToPlace && initialSearchQuery) {
        const sortedFragments = sortBySearchRelevance(fragments, initialSearchQuery);
        fragmentToPlace = sortedFragments[0] || null;
        console.log('Using first sorted fragment:', fragmentToPlace?.id);
      }

      if (fragmentToPlace) {
        console.log('Auto-placing fragment:', fragmentToPlace.id, 'imagePath:', fragmentToPlace.imagePath);
        autoPlaceFragment(fragmentToPlace);
      } else {
        console.log('No fragment to place');
      }
    }
  }, [isSearchMode, fragments, hasAutoPlaced, initialSearchQuery, selectedFragmentId, autoPlaceFragment]);

  // Get image path - default to segmented if available, otherwise original
  const getImagePath = useCallback((fragment: ManuscriptFragment, preferSegmented: boolean = true) => {
    return (preferSegmented && fragment.segmentedImagePath)
      ? fragment.segmentedImagePath
      : fragment.imagePath;
  }, []);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();

    const fragment = draggedFragmentRef.current;
    const position = dropPositionRef.current;

    if (!fragment || !position) {
      return;
    }

    // Load the image to get its natural dimensions
    const img = new Image();
    const showSegmented = true; // Default to segmented version
    const imagePath = getImagePath(fragment, showSegmented);
    img.src = imagePath;
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
        imagePath: imagePath,
        x: position.x - width / 2,
        y: position.y - height / 2,
        width: width,
        height: height,
        rotation: 0,
        scaleX: 1,
        scaleY: 1,
        isLocked: false,
        isSelected: false,
        showSegmented: true, // Default to showing segmented version
      };

      setCanvasFragments([...canvasFragments, newCanvasFragment]);
    };

    draggedFragmentRef.current = null;
    dropPositionRef.current = null;
  };

  // Clear search and reset to normal mode
  const handleClearSearch = useCallback(() => {
    setIsSearchMode(false);
    setInitialSearchQuery(null);
    setSelectedFragmentId(null);
    setFilters(prev => ({ ...prev, search: undefined }));
    setHasAutoPlaced(false);
  }, []);

  // Handle sidebar search
  const handleSidebarSearch = useCallback((query: string) => {
    setSidebarSearchQuery(query);
    // Update filters to trigger database search
    setFilters(prev => ({ ...prev, search: query || undefined }));
    setCurrentOffset(0); // Reset pagination
  }, []);

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

  // Toggle segmentation for selected fragment
  const handleToggleSelectedFragmentSegmentation = async () => {
    if (selectedFragmentIds.length !== 1) return;

    const selectedId = selectedFragmentIds[0];
    const canvasFragment = canvasFragments.find(f => f.id === selectedId);
    if (!canvasFragment) return;

    // Find the manuscript fragment to get segmented/original paths
    const manuscriptFragment = fragments.find(f => f.id === canvasFragment.fragmentId);
    if (!manuscriptFragment) return;

    const newShowSegmented = !canvasFragment.showSegmented;
    const newImagePath = (newShowSegmented && manuscriptFragment.segmentedImagePath)
      ? manuscriptFragment.segmentedImagePath
      : manuscriptFragment.imagePath;

    // Load the new image to get its actual dimensions
    const img = new Image();
    img.src = newImagePath;

    img.onload = () => {
      const newImageWidth = img.naturalWidth;
      const newImageHeight = img.naturalHeight;

      // Calculate what the new display size should be using the same scaling logic
      const { width, height } = calculateDisplaySize(
        manuscriptFragment,
        newImageWidth,
        newImageHeight
      );

      const updatedFragments = canvasFragments.map((f) =>
        f.id === selectedId
          ? {
              ...f,
              showSegmented: newShowSegmented,
              imagePath: newImagePath,
              width,
              height,
            }
          : f
      );
      setCanvasFragments(updatedFragments);
      setHasUnsavedChanges(true);
    };

    img.onerror = () => {
      console.error('Failed to load new image:', newImagePath);
      // Fall back to just changing the path without resizing
      const updatedFragments = canvasFragments.map((f) =>
        f.id === selectedId ? { ...f, showSegmented: newShowSegmented, imagePath: newImagePath } : f
      );
      setCanvasFragments(updatedFragments);
      setHasUnsavedChanges(true);
    };
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

  // Save canvas progress to database
  const handleSave = async () => {
    // Cancel any pending auto-save
    if (autoSaveTimeoutRef.current) {
      clearTimeout(autoSaveTimeoutRef.current);
    }

    // Ensure we have a project
    const projectId = await ensureProject();
    if (!projectId) {
      console.error('Failed to create or get project');
      return;
    }

    await saveProject(projectId, canvasFragments, notes);
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

  // Handle double-click on canvas fragment to show metadata
  const handleCanvasFragmentDoubleClick = async (fragmentId: string) => {
    // First try to find in already loaded fragments
    let manuscriptFragment = fragments.find(f => f.id === fragmentId);

    // If not found in loaded fragments, fetch from database
    if (!manuscriptFragment) {
      manuscriptFragment = await getFragmentById(fragmentId);
    }

    if (manuscriptFragment) {
      setSelectedMetadataFragment(manuscriptFragment);
    }
  };

  // Close metadata panel
  const handleCloseMetadata = () => {
    setSelectedMetadataFragment(null);
  };

  // Get selected fragment info for segmentation toggle
  const selectedFragment = selectedFragmentIds.length === 1
    ? canvasFragments.find(f => f.id === selectedFragmentIds[0])
    : undefined;

  // Find the corresponding ManuscriptFragment to check if it has segmentation data
  const selectedManuscriptFragment = selectedFragment
    ? fragments.find(f => f.id === selectedFragment.fragmentId)
    : undefined;

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
        selectedFragmentHasSegmentation={selectedManuscriptFragment?.hasSegmentation}
        selectedFragmentShowSegmented={selectedFragment?.showSegmented}
        onToggleSelectedFragmentSegmentation={handleToggleSelectedFragmentSegmentation}
        projectName={currentProjectName}
        saveStatus={saveStatus}
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
          searchQuery={initialSearchQuery}
          onClearSearch={handleClearSearch}
          onSidebarSearch={handleSidebarSearch}
          onFragmentUpdate={loadFragments}
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

      {/* Fragment Metadata Modal */}
      {selectedMetadataFragment && (
        <FragmentMetadata
          fragment={selectedMetadataFragment}
          onClose={handleCloseMetadata}
          onUpdate={loadFragments}
        />
      )}
    </div>
  );
}

export default CanvasPage;

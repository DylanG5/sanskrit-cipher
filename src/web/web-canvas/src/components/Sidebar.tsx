import React, { useState, useRef, useCallback, useEffect } from "react";
import { getReverseSideId } from "../utils/fragments";
import { ManuscriptFragment, CanvasFragment } from "../types/fragment";
import { CustomFilterDefinition } from "../types/customFilters";
import { EdgeMatchRecord } from "../services/electron-api";
import { bulkDeleteFragments } from "../services/fragment-service";
import FragmentMetadata from "./FragmentMetadata";
import VirtualizedFragmentList from "./VirtualizedFragmentList";

interface SidebarProps {
  fragments: ManuscriptFragment[];
  onDragStart: (fragment: ManuscriptFragment, e: React.DragEvent) => void;
  onMultiDragStart: (fragments: ManuscriptFragment[], e: React.DragEvent) => void;
  width: number;
  onWidthChange: (width: number) => void;
  isOpen: boolean;
  onToggle: () => void;
  isLoading?: boolean;
  onLoadMore?: () => void;
  hasMore?: boolean;
  isLoadingMore?: boolean;
  searchQuery?: string | null;
  onClearSearch?: () => void;
  onSidebarSearch?: (query: string) => void;
  onFragmentUpdate?: () => void;
  canvasFragments?: CanvasFragment[];
  gridScale?: number;
  customFilters?: CustomFilterDefinition[];
  edgeMatchMode?: boolean;
  edgeMatchAnchorId?: string | null;
  edgeMatches?: EdgeMatchRecord[];
  onPlaceEdgeMatch?: (match: EdgeMatchRecord) => void;
  onExitEdgeMatch?: () => void;
  onBulkAddToCanvas?: (fragments: ManuscriptFragment[]) => void;
  onBulkEditSidebarMetadata?: (fragmentIds: string[]) => void;
  onDragStartSelected?: (selectedFragments: ManuscriptFragment[]) => void;
  onFragmentsDeleted?: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  fragments,
  onDragStart,
  onMultiDragStart,
  width,
  onWidthChange,
  isOpen,
  onToggle,
  isLoading = false,
  onLoadMore,
  hasMore = false,
  isLoadingMore = false,
  searchQuery = null,
  onClearSearch,
  onSidebarSearch,
  onFragmentUpdate,
  canvasFragments = [],
  gridScale = 25,
  customFilters = [],
  edgeMatchMode = false,
  edgeMatchAnchorId = null,
  edgeMatches = [],
  onPlaceEdgeMatch,
  onExitEdgeMatch,
  onBulkAddToCanvas,
  onBulkEditSidebarMetadata,
  onDragStartSelected,
  onFragmentsDeleted,
}) => {
  const [selectedFragment, setSelectedFragment] = useState<ManuscriptFragment | null>(null);

  // Multi-selection state
  const [selectedSidebarIds, setSelectedSidebarIds] = useState<Set<string>>(new Set());
  const [autoSelectReverse, setAutoSelectReverse] = useState(false);
  const autoSelectReverseRef = useRef(false);
  const fragmentsRef = useRef(fragments);

  const [isResizing, setIsResizing] = useState(false);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const [containerHeight, setContainerHeight] = useState(window.innerHeight - 120);
  const [scrollPosition, setScrollPosition] = useState(0);
  const [localSearchQuery, setLocalSearchQuery] = useState("");
  const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const clickTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Keep refs in sync so callbacks never go stale
  useEffect(() => { autoSelectReverseRef.current = autoSelectReverse; }, [autoSelectReverse]);
  useEffect(() => { fragmentsRef.current = fragments; }, [fragments]);

  // Clear sidebar selection when fragments list changes (filter/search)
  useEffect(() => {
    setSelectedSidebarIds(new Set());
  }, [fragments]);

  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const handleBulkDelete = useCallback(async () => {
    const ids = Array.from(selectedSidebarIds);
    const confirmed = window.confirm(
      `Permanently delete ${ids.length} fragment${ids.length !== 1 ? "s" : ""} from the database?\n\nThis cannot be undone.`
    );
    if (!confirmed) return;

    setIsDeleting(true);
    setDeleteError(null);
    try {
      const result = await bulkDeleteFragments(ids);
      if (result.success) {
        setSelectedSidebarIds(new Set());
        onFragmentsDeleted?.();
      } else {
        setDeleteError(result.error ?? "Delete failed");
      }
    } catch {
      setDeleteError("An unexpected error occurred");
    } finally {
      setIsDeleting(false);
    }
  }, [selectedSidebarIds, onFragmentsDeleted]);

  // Toggle select with auto-pair support
  const handleToggleSelect = useCallback((fragment: ManuscriptFragment) => {
    setSelectedSidebarIds(prev => {
      const next = new Set(prev);
      if (next.has(fragment.id)) {
        next.delete(fragment.id);
        const rev = getReverseSideId(fragment.id);
        if (rev) next.delete(rev);
      } else {
        next.add(fragment.id);
        if (autoSelectReverseRef.current) {
          const rev = getReverseSideId(fragment.id);
          if (rev && fragmentsRef.current.some(f => f.id === rev)) {
            next.add(rev);
          }
        }
      }
      return next;
    });
  }, []);

  const selectedSidebarIdsRef = useRef(selectedSidebarIds);
  useEffect(() => { selectedSidebarIdsRef.current = selectedSidebarIds; }, [selectedSidebarIds]);

  // Drag: if multiple selected and dragged fragment is in selection, use onMultiDragStart
  const handleDragStartInternal = useCallback((fragment: ManuscriptFragment, e: React.DragEvent) => {
    const ids = selectedSidebarIdsRef.current;
    if (ids.size > 1 && ids.has(fragment.id)) {
      const selected = fragmentsRef.current.filter(f => ids.has(f.id));
      onMultiDragStart(selected, e);
      if (onDragStartSelected) onDragStartSelected(selected);
    } else {
      onDragStart(fragment, e);
    }
  }, [onDragStart, onMultiDragStart, onDragStartSelected]);

  const handleFragmentClick = (fragment: ManuscriptFragment, e: React.MouseEvent) => {
    if ((e.target as HTMLElement).tagName === "IMG") return;
    e.stopPropagation();
    if (clickTimeoutRef.current) {
      clearTimeout(clickTimeoutRef.current);
      clickTimeoutRef.current = null;
    }
    if (e.detail > 1) return;
    clickTimeoutRef.current = setTimeout(() => {
      setSelectedFragment(fragment);
      clickTimeoutRef.current = null;
    }, 220);
  };

  const handleCloseMetadata = () => {
    setSelectedFragment(null);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  };

  const handleMouseMove = useCallback((e: MouseEvent) => {
    const newWidth = e.clientX;
    if (newWidth >= 220 && newWidth <= 500) {
      onWidthChange(newWidth);
    }
  }, [onWidthChange]);

  const handleMouseUp = useCallback(() => {
    setIsResizing(false);
  }, []);

  React.useEffect(() => {
    if (isResizing) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      return () => {
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
      };
    }
  }, [isResizing, handleMouseMove, handleMouseUp]);

  React.useEffect(() => {
    const handleResize = () => {
      setContainerHeight(window.innerHeight - 120);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    if (!onSidebarSearch) return;
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }
    if (!localSearchQuery.trim()) {
      onSidebarSearch("");
      return;
    }
    debounceTimeoutRef.current = setTimeout(() => {
      onSidebarSearch(localSearchQuery.trim());
    }, 300);
    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
    };
  }, [localSearchQuery, onSidebarSearch]);

  useEffect(() => {
    return () => {
      if (clickTimeoutRef.current) {
        clearTimeout(clickTimeoutRef.current);
      }
    };
  }, []);

  return (
    <>
      {/* Toggle button when closed */}
      {!isOpen && (
        <button
          onClick={onToggle}
          className="fixed left-0 top-1/2 -translate-y-1/2 text-white px-3 py-6 rounded-r-xl shadow-xl transition-all duration-300 z-30 group"
          style={{ background: "linear-gradient(90deg, #292524 0%, #44403c 100%)" }}
          onMouseEnter={(e) => { e.currentTarget.style.background = "linear-gradient(90deg, #ea580c 0%, #c2410c 100%)"; }}
          onMouseLeave={(e) => { e.currentTarget.style.background = "linear-gradient(90deg, #292524 0%, #44403c 100%)"; }}
          title="Open fragments sidebar"
        >
          <div className="flex flex-col items-center gap-2.5">
            <svg className="w-5 h-5 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            <span className="text-xs font-bold font-body" style={{ writingMode: "vertical-rl" }}>Fragments</span>
            <div className="text-xs font-bold rounded-full px-1.5 py-0.5 min-w-[22px] text-center" style={{ background: "#ea580c", color: "#ffffff" }}>
              {fragments.length}
            </div>
          </div>
        </button>
      )}

      {/* Sidebar panel */}
      {isOpen && (
        <div
          ref={sidebarRef}
          style={{
            width: `${width}px`,
            background: "linear-gradient(180deg, #fafaf9 0%, #f5f5f4 100%)",
            borderRight: "1px solid rgba(120, 113, 108, 0.2)",
          }}
          className="relative shadow-lg flex-shrink-0 overflow-hidden"
        >
          <div className="p-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold flex items-center gap-2.5 font-body" style={{ color: "#292524" }}>
                <svg className="w-5 h-5" style={{ color: "#ea580c" }} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                Fragments
              </h2>
              <button
                onClick={onToggle}
                className="rounded-xl p-2 transition-all duration-200"
                style={{ color: "#78716c", background: "transparent" }}
                onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(231, 229, 228, 0.8)"; e.currentTarget.style.color = "#292524"; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "#78716c"; }}
                title="Close sidebar"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Multi-select controls: auto-pair toggle */}
            <div className="mb-3 flex items-center gap-2 flex-wrap">
              <button
                onClick={() => setAutoSelectReverse(v => !v)}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold transition-all font-body border"
                style={{
                  background: autoSelectReverse ? 'rgba(234,88,12,0.12)' : 'rgba(231,229,228,0.5)',
                  borderColor: autoSelectReverse ? 'rgba(234,88,12,0.4)' : 'rgba(214,211,209,0.5)',
                  color: autoSelectReverse ? '#c2410c' : '#78716c',
                }}
                title="When selecting a fragment, also select its recto/verso pair"
              >
                <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M7 16V4m0 0L3 8m4-4l4 4M17 8v12m0 0l4-4m-4 4l-4-4" />
                </svg>
                Auto-pair
              </button>

              {selectedSidebarIds.size > 0 && (
                <div className="flex items-center gap-1.5">
                  <span className="px-2 py-1 rounded-lg text-xs font-bold font-body" style={{ background: 'rgba(59,130,246,0.12)', color: '#1d4ed8' }}>
                    {selectedSidebarIds.size} selected
                  </span>
                  <button
                    onClick={() => { setSelectedSidebarIds(new Set()); setDeleteError(null); }}
                    className="p-1 rounded transition-colors"
                    style={{ color: '#94a3b8' }}
                    onMouseEnter={(e) => { e.currentTarget.style.color = '#ef4444'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.color = '#94a3b8'; }}
                    title="Clear selection"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              )}
            </div>

            {/* Local search bar */}
            <div className="mb-3.5 relative">
              <div className="relative">
                <input
                  type="text"
                  value={localSearchQuery}
                  onChange={(e) => setLocalSearchQuery(e.target.value)}
                  placeholder="Search fragments..."
                  className="w-full pl-10 pr-10 py-2.5 text-sm bg-white border rounded-xl focus:outline-none focus:ring-2 placeholder-neutral-400 font-body transition-all"
                  style={{ borderColor: "rgba(214, 211, 209, 0.5)", color: "#292524" }}
                  onFocus={(e) => { e.currentTarget.style.borderColor = "rgba(234, 88, 12, 0.4)"; e.currentTarget.style.boxShadow = "0 0 0 3px rgba(234, 88, 12, 0.1)"; }}
                  onBlur={(e) => { e.currentTarget.style.borderColor = "rgba(214, 211, 209, 0.5)"; e.currentTarget.style.boxShadow = "none"; }}
                />
                <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none" style={{ color: "#a8a29e" }} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                {localSearchQuery && (
                  <button
                    onClick={() => { setLocalSearchQuery(""); if (onSidebarSearch) { onSidebarSearch(""); } }}
                    className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-lg transition-colors"
                    style={{ color: "#a8a29e" }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(231, 229, 228, 0.6)"; e.currentTarget.style.color = "#292524"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "#a8a29e"; }}
                    title="Clear search"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                )}
              </div>
              {localSearchQuery && (
                <div className="mt-1.5 text-xs font-medium font-body" style={{ color: "#78716c" }}>
                  Searching database...
                </div>
              )}
            </div>

            {/* Search indicator (from landing page search) */}
            {searchQuery && (
              <div className="mb-3.5 px-4 py-2.5 rounded-xl border" style={{ background: "#fff7ed", borderColor: "rgba(234, 88, 12, 0.3)" }}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <svg className="w-4 h-4" style={{ color: "#ea580c" }} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    <span className="text-sm font-semibold font-body" style={{ color: "#c2410c" }}>
                      Results for: "{searchQuery}"
                    </span>
                  </div>
                  {onClearSearch && (
                    <button
                      onClick={onClearSearch}
                      className="p-1.5 rounded-lg transition-colors"
                      style={{ color: "#ea580c" }}
                      onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(234, 88, 12, 0.2)"; e.currentTarget.style.color = "#c2410c"; }}
                      onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "#ea580c"; }}
                      title="Clear search"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  )}
                </div>
              </div>
            )}

            {edgeMatchMode ? (
              /* Edge Match Ranking View */
              <div>
                <div className="mb-3 px-3 py-2.5 rounded-xl border" style={{ background: "#eff6ff", borderColor: "rgba(59, 130, 246, 0.3)" }}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                      </svg>
                      <span className="text-sm font-semibold text-blue-800 font-body">
                        Edge Matches for {edgeMatchAnchorId}
                      </span>
                    </div>
                    {onExitEdgeMatch && (
                      <button onClick={onExitEdgeMatch} className="p-1.5 rounded-lg text-blue-600 hover:bg-blue-100 transition-colors" title="Exit edge match mode">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    )}
                  </div>
                  <div className="text-xs text-blue-600 mt-1 font-body">
                    {edgeMatches.length} match{edgeMatches.length !== 1 ? "es" : ""} found
                  </div>
                </div>

                {edgeMatches.length === 0 ? (
                  <div className="bg-white rounded-2xl p-6 shadow-md border text-center" style={{ borderColor: "rgba(214, 211, 209, 0.3)" }}>
                    <p className="text-sm font-bold mb-1 font-body" style={{ color: "#292524" }}>No matches found</p>
                    <p className="text-xs font-body" style={{ color: "#a8a29e" }}>No compatible edges detected for this fragment</p>
                  </div>
                ) : (
                  <div className="space-y-2 overflow-y-auto" style={{ maxHeight: containerHeight - 120 }}>
                    {edgeMatches.map((match, idx) => {
                      const confidence = match.confidence ?? 1 / (1 + match.score);
                      const confidencePct = Math.round(confidence * 100);
                      return (
                        <div key={match.id} className="bg-white rounded-xl p-3 shadow-sm border transition-all hover:shadow-md" style={{ borderColor: "rgba(214, 211, 209, 0.3)" }}>
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-1.5">
                                <span className="text-xs font-bold text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded font-body">#{idx + 1}</span>
                                <span className="text-sm font-semibold text-slate-800 truncate font-body">{match.fragment_b_id}</span>
                              </div>
                              <div className="mt-1 text-xs text-slate-500 font-body">{match.edge_a_name} ↔ {match.edge_b_name}</div>
                              <div className="mt-1 flex items-center gap-2">
                                <div className="flex-1 bg-slate-100 rounded-full h-1.5">
                                  <div className="h-1.5 rounded-full transition-all" style={{ width: `${confidencePct}%`, background: confidencePct > 70 ? "#16a34a" : confidencePct > 40 ? "#eab308" : "#dc2626" }} />
                                </div>
                                <span className="text-xs font-semibold text-slate-600 font-body">{confidencePct}%</span>
                              </div>
                            </div>
                            {onPlaceEdgeMatch && (
                              <button
                                onClick={() => onPlaceEdgeMatch(match)}
                                className="flex-shrink-0 px-3 py-1.5 text-xs font-semibold text-white rounded-lg transition-colors font-body"
                                style={{ background: "#2563eb" }}
                                onMouseEnter={(e) => { e.currentTarget.style.background = "#1d4ed8"; }}
                                onMouseLeave={(e) => { e.currentTarget.style.background = "#2563eb"; }}
                                title="Place matched fragment on canvas"
                              >
                                Place
                              </button>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            ) : isLoading ? (
              <div className="bg-white rounded-2xl p-6 shadow-md border text-center" style={{ borderColor: "rgba(214, 211, 209, 0.3)" }}>
                <div className="rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-3" style={{ background: "rgba(234, 88, 12, 0.1)" }}>
                  <svg className="w-8 h-8 animate-spin" style={{ color: "#ea580c" }} fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                </div>
                <p className="text-sm font-bold mb-1 font-body" style={{ color: "#292524" }}>Loading fragments...</p>
                <p className="text-xs font-body" style={{ color: "#a8a29e" }}>Please wait</p>
              </div>
            ) : fragments.length === 0 ? (
              <div className="bg-white rounded-2xl p-6 shadow-md border text-center" style={{ borderColor: "rgba(214, 211, 209, 0.3)" }}>
                <div className="rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-3" style={{ background: "rgba(168, 162, 158, 0.15)" }}>
                  <svg className="w-8 h-8" style={{ color: "#a8a29e" }} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
                  </svg>
                </div>
                <p className="text-sm font-bold mb-1 font-body" style={{ color: "#292524" }}>
                  {localSearchQuery ? "No matching fragments" : "No fragments available"}
                </p>
                <p className="text-xs font-body" style={{ color: "#a8a29e" }}>
                  {localSearchQuery ? "Try a different search term" : "Try adjusting your filters"}
                </p>
              </div>
            ) : (
              <VirtualizedFragmentList
                fragments={fragments}
                selectedIds={selectedSidebarIds}
                onDragStart={handleDragStartInternal}
                onFragmentClick={handleFragmentClick}
                onToggleSelect={handleToggleSelect}
                containerHeight={selectedSidebarIds.size > 0 ? containerHeight - 80 : containerHeight}
                onLoadMore={onLoadMore}
                hasMore={hasMore}
                isLoadingMore={isLoadingMore}
                scrollPosition={scrollPosition}
                onScrollPositionChange={setScrollPosition}
              />
            )}
          </div>

          {/* Multiselect Action Bar */}
          {selectedSidebarIds.size > 0 && (
            <div className="absolute bottom-0 left-0 right-0 bg-white border-t border-slate-200 px-3 pt-2 pb-2.5 z-10 shadow-[0_-2px_8px_rgba(0,0,0,0.06)]">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-slate-600">{selectedSidebarIds.size} selected</span>
                <button
                  onClick={() => { setSelectedSidebarIds(new Set()); setDeleteError(null); }}
                  className="text-xs text-slate-400 hover:text-slate-600 transition-colors"
                  title="Clear selection"
                >
                  Clear
                </button>
              </div>
              <div className="flex items-center gap-2">
                {onBulkAddToCanvas && (
                  <button
                    onClick={() => {
                      const selected = fragments.filter(f => selectedSidebarIds.has(f.id));
                      if (selected.length > 0) onBulkAddToCanvas(selected);
                    }}
                    className="flex-1 py-1.5 text-xs font-semibold text-white rounded-lg transition-colors"
                    style={{ background: "#2563eb" }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = "#1d4ed8"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "#2563eb"; }}
                    title="Add selected fragments to canvas"
                  >
                    Add to Canvas
                  </button>
                )}
                {onBulkEditSidebarMetadata && (
                  <button
                    onClick={() => onBulkEditSidebarMetadata(Array.from(selectedSidebarIds))}
                    className="flex-1 py-1.5 text-xs font-semibold text-white rounded-lg transition-colors"
                    style={{ background: "#ea580c" }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = "#c2410c"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "#ea580c"; }}
                    title="Edit metadata for selected fragments"
                  >
                    Edit Metadata
                  </button>
                )}
                <button
                  onClick={handleBulkDelete}
                  disabled={isDeleting}
                  className="flex-1 py-1.5 text-xs font-semibold text-white rounded-lg transition-colors disabled:opacity-50"
                  style={{ background: "#dc2626" }}
                  onMouseEnter={(e) => { if (!isDeleting) e.currentTarget.style.background = "#b91c1c"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = "#dc2626"; }}
                  title="Permanently delete selected fragments from database"
                >
                  {isDeleting ? "Deleting…" : "Delete"}
                </button>
              </div>
              {deleteError && (
                <p className="text-xs text-red-600 font-medium mt-1.5">{deleteError}</p>
              )}
            </div>
          )}

          {/* Resize Handle */}
          <div
            onMouseDown={handleMouseDown}
            className="absolute top-0 right-0 w-1 h-full cursor-col-resize transition-colors group"
            style={{ background: isResizing ? "#ea580c" : "transparent" }}
            onMouseEnter={(e) => { if (!isResizing) e.currentTarget.style.background = "#ea580c"; }}
            onMouseLeave={(e) => { if (!isResizing) e.currentTarget.style.background = "transparent"; }}
            title="Drag to resize"
          >
            <div className="absolute top-1/2 right-0 -translate-y-1/2 translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
              <div className="text-white rounded-full p-1 shadow-xl" style={{ background: "#ea580c" }}>
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M8 9l4-4 4 4m0 6l-4 4-4-4" />
                </svg>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Metadata panel */}
      {selectedFragment && (
        <>
          <div className="fixed inset-0 bg-black bg-opacity-30 z-40 backdrop-blur-sm" onClick={handleCloseMetadata} />
          <FragmentMetadata
            fragment={selectedFragment}
            onClose={handleCloseMetadata}
            onUpdate={onFragmentUpdate}
            canvasFragment={canvasFragments.find(cf => cf.fragmentId === selectedFragment.id)}
            gridScale={gridScale}
            customFilters={customFilters}
          />
        </>
      )}
    </>
  );
};

export default Sidebar;

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { ManuscriptFragment } from '../types/fragment';
import FragmentMetadata from './FragmentMetadata';
import VirtualizedFragmentList from './VirtualizedFragmentList';

interface SidebarProps {
  fragments: ManuscriptFragment[];
  onDragStart: (fragment: ManuscriptFragment, e: React.DragEvent) => void;
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
}

const Sidebar: React.FC<SidebarProps> = ({
  fragments,
  onDragStart,
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
}) => {
  const [selectedFragment, setSelectedFragment] = useState<ManuscriptFragment | null>(null);
  const [isResizing, setIsResizing] = useState(false);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const [containerHeight, setContainerHeight] = useState(window.innerHeight - 120); // Header + padding
  const [scrollPosition, setScrollPosition] = useState(0);
  const [localSearchQuery, setLocalSearchQuery] = useState('');
  const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleFragmentClick = (fragment: ManuscriptFragment, e: React.MouseEvent) => {
    // Don't show metadata if we're starting a drag
    if ((e.target as HTMLElement).tagName === 'IMG') return;

    e.stopPropagation();
    setSelectedFragment(fragment);
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
    // Constrain width between 220px and 500px
    if (newWidth >= 220 && newWidth <= 500) {
      onWidthChange(newWidth);
    }
  }, [onWidthChange]);

  const handleMouseUp = useCallback(() => {
    setIsResizing(false);
  }, []);

  React.useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isResizing, handleMouseMove, handleMouseUp]);

  // Update container height on window resize
  React.useEffect(() => {
    const handleResize = () => {
      setContainerHeight(window.innerHeight - 120);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Debounced database search when local search query changes
  useEffect(() => {
    if (!onSidebarSearch) return;

    // Clear existing timeout
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }

    // If search is empty, clear the search
    if (!localSearchQuery.trim()) {
      onSidebarSearch('');
      return;
    }

    // Debounce the search by 300ms
    debounceTimeoutRef.current = setTimeout(() => {
      onSidebarSearch(localSearchQuery.trim());
    }, 300);

    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
    };
  }, [localSearchQuery, onSidebarSearch]);

  return (
    <>
      {/* Toggle button when closed */}
      {!isOpen && (
        <button
          onClick={onToggle}
          className="fixed left-0 top-1/2 -translate-y-1/2 text-white px-3 py-6 rounded-r-xl shadow-xl transition-all duration-300 z-30 group"
          style={{
            background: 'linear-gradient(90deg, #292524 0%, #44403c 100%)'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'linear-gradient(90deg, #ea580c 0%, #c2410c 100%)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'linear-gradient(90deg, #292524 0%, #44403c 100%)';
          }}
          title="Open fragments sidebar"
        >
          <div className="flex flex-col items-center gap-2.5">
            <svg className="w-5 h-5 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            <span className="text-xs font-bold font-body" style={{ writingMode: 'vertical-rl' }}>Fragments</span>
            <div className="text-xs font-bold rounded-full px-1.5 py-0.5 min-w-[22px] text-center" style={{
              background: '#ea580c',
              color: '#ffffff'
            }}>
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
          background: 'linear-gradient(180deg, #fafaf9 0%, #f5f5f4 100%)',
          borderRight: '1px solid rgba(120, 113, 108, 0.2)'
        }}
        className="relative shadow-lg flex-shrink-0 overflow-hidden"
      >
        <div className="p-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold flex items-center gap-2.5 font-body" style={{ color: '#292524' }}>
              <svg className="w-5 h-5" style={{ color: '#ea580c' }} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              Fragments
            </h2>
            <button
              onClick={onToggle}
              className="rounded-xl p-2 transition-all duration-200"
              style={{
                color: '#78716c',
                background: 'transparent'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(231, 229, 228, 0.8)';
                e.currentTarget.style.color = '#292524';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent';
                e.currentTarget.style.color = '#78716c';
              }}
              title="Close sidebar"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
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
                style={{
                  borderColor: 'rgba(214, 211, 209, 0.5)',
                  color: '#292524'
                }}
                onFocus={(e) => {
                  e.currentTarget.style.borderColor = 'rgba(234, 88, 12, 0.4)';
                  e.currentTarget.style.boxShadow = '0 0 0 3px rgba(234, 88, 12, 0.1)';
                }}
                onBlur={(e) => {
                  e.currentTarget.style.borderColor = 'rgba(214, 211, 209, 0.5)';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              />
              <svg
                className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none"
                style={{ color: '#a8a29e' }}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                strokeWidth={2}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              {localSearchQuery && (
                <button
                  onClick={() => {
                    setLocalSearchQuery('');
                    if (onSidebarSearch) {
                      onSidebarSearch('');
                    }
                  }}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-lg transition-colors"
                  style={{ color: '#a8a29e' }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'rgba(231, 229, 228, 0.6)';
                    e.currentTarget.style.color = '#292524';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'transparent';
                    e.currentTarget.style.color = '#a8a29e';
                  }}
                  title="Clear search"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              )}
            </div>
            {localSearchQuery && (
              <div className="mt-1.5 text-xs font-medium font-body" style={{ color: '#78716c' }}>
                Searching database...
              </div>
            )}
          </div>

          {/* Search indicator (from landing page search) */}
          {searchQuery && (
            <div className="mb-3.5 px-4 py-2.5 rounded-xl border" style={{
              background: '#fff7ed',
              borderColor: 'rgba(234, 88, 12, 0.3)'
            }}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4" style={{ color: '#ea580c' }} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  <span className="text-sm font-semibold font-body" style={{ color: '#c2410c' }}>
                    Results for: "{searchQuery}"
                  </span>
                </div>
                {onClearSearch && (
                  <button
                    onClick={onClearSearch}
                    className="p-1.5 rounded-lg transition-colors"
                    style={{ color: '#ea580c' }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'rgba(234, 88, 12, 0.2)';
                      e.currentTarget.style.color = '#c2410c';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent';
                      e.currentTarget.style.color = '#ea580c';
                    }}
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

          {isLoading ? (
            <div className="bg-white rounded-2xl p-6 shadow-md border text-center" style={{
              borderColor: 'rgba(214, 211, 209, 0.3)'
            }}>
              <div className="rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-3" style={{
                background: 'rgba(234, 88, 12, 0.1)'
              }}>
                <svg className="w-8 h-8 animate-spin" style={{ color: '#ea580c' }} fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              </div>
              <p className="text-sm font-bold mb-1 font-body" style={{ color: '#292524' }}>Loading fragments...</p>
              <p className="text-xs font-body" style={{ color: '#a8a29e' }}>Please wait</p>
            </div>
          ) : fragments.length === 0 ? (
            <div className="bg-white rounded-2xl p-6 shadow-md border text-center" style={{
              borderColor: 'rgba(214, 211, 209, 0.3)'
            }}>
              <div className="rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-3" style={{
                background: 'rgba(168, 162, 158, 0.15)'
              }}>
                <svg className="w-8 h-8" style={{ color: '#a8a29e' }} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
                </svg>
              </div>
              <p className="text-sm font-bold mb-1 font-body" style={{ color: '#292524' }}>
                {localSearchQuery ? 'No matching fragments' : 'No fragments available'}
              </p>
              <p className="text-xs font-body" style={{ color: '#a8a29e' }}>
                {localSearchQuery ? 'Try a different search term' : 'Try adjusting your filters'}
              </p>
            </div>
          ) : (
          <VirtualizedFragmentList
            fragments={fragments}
            onDragStart={onDragStart}
            onFragmentClick={handleFragmentClick}
            containerHeight={containerHeight}
            onLoadMore={onLoadMore}
            hasMore={hasMore}
            isLoadingMore={isLoadingMore}
            scrollPosition={scrollPosition}
            onScrollPositionChange={setScrollPosition}
          />
          )}
        </div>

        {/* Resize Handle */}
        <div
          onMouseDown={handleMouseDown}
          className="absolute top-0 right-0 w-1 h-full cursor-col-resize transition-colors group"
          style={{
            background: isResizing ? '#ea580c' : 'transparent'
          }}
          onMouseEnter={(e) => {
            if (!isResizing) {
              e.currentTarget.style.background = '#ea580c';
            }
          }}
          onMouseLeave={(e) => {
            if (!isResizing) {
              e.currentTarget.style.background = 'transparent';
            }
          }}
          title="Drag to resize"
        >
          <div className="absolute top-1/2 right-0 -translate-y-1/2 translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
            <div className="text-white rounded-full p-1 shadow-xl" style={{
              background: '#ea580c'
            }}>
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
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-black bg-opacity-30 z-40 backdrop-blur-sm"
            onClick={handleCloseMetadata}
          />
          {/* Metadata component */}
          <FragmentMetadata
            fragment={selectedFragment}
            onClose={handleCloseMetadata}
          />
        </>
      )}
    </>
  );
};

export default Sidebar;

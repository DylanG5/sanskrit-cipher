import React, { useCallback, useRef, useEffect, useMemo } from 'react';
import { FixedSizeList, ListChildComponentProps } from 'react-window';
import { ManuscriptFragment } from '../types/fragment';

interface VirtualizedFragmentListProps {
  fragments: ManuscriptFragment[];
  selectedIds: Set<string>;
  onDragStart: (fragment: ManuscriptFragment, e: React.DragEvent) => void;
  onFragmentClick: (fragment: ManuscriptFragment, e: React.MouseEvent) => void;
  onToggleSelect: (fragment: ManuscriptFragment) => void;
  containerHeight: number;
  onLoadMore?: () => void;
  hasMore?: boolean;
  isLoadingMore?: boolean;
  scrollPosition?: number;
  onScrollPositionChange?: (position: number) => void;
}

const ITEM_HEIGHT = 180;

interface FragmentRowData {
  fragments: ManuscriptFragment[];
  selectedIds: Set<string>;
  onDragStart: (fragment: ManuscriptFragment, e: React.DragEvent) => void;
  onFragmentClick: (fragment: ManuscriptFragment, e: React.MouseEvent) => void;
  onToggleSelect: (fragment: ManuscriptFragment) => void;
}

const FragmentRow = ({ index, style, data }: ListChildComponentProps<FragmentRowData>) => {
  if (!data) return null;

  const { fragments, selectedIds, onDragStart, onFragmentClick, onToggleSelect } = data;
  const fragment = fragments[index];

  if (!fragment) return null;

  const isSelected = selectedIds.has(fragment.id);

  const handleClick = (e: React.MouseEvent) => {
    if (e.shiftKey || e.ctrlKey || e.metaKey) {
      e.preventDefault();
      onToggleSelect(fragment);
    } else {
      onFragmentClick(fragment, e);
    }
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    onToggleSelect(fragment);
  };

  return (
    <div style={style} className="px-4 pb-3">
      <div
        draggable
        onDragStart={(e) => onDragStart(fragment, e)}
        onClick={handleClick}
        onContextMenu={handleContextMenu}
        className="cursor-move rounded-lg shadow-sm transition-all duration-200 p-3 border relative h-full"
        style={{
          background: isSelected ? '#eff6ff' : '#ffffff',
          borderColor: isSelected ? '#3b82f6' : '#e2e8f0',
          boxShadow: isSelected ? '0 0 0 2px rgba(59,130,246,0.3)' : undefined,
        }}
      >
        {/* Selection checkbox indicator */}
        <div
          className="absolute top-2 left-2 z-10 w-5 h-5 rounded-full border-2 flex items-center justify-center transition-all"
          style={{
            background: isSelected ? '#3b82f6' : 'rgba(255,255,255,0.8)',
            borderColor: isSelected ? '#3b82f6' : '#cbd5e1',
          }}
        >
          {isSelected && (
            <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
          )}
        </div>

        <img
          src={fragment.thumbnailPath}
          alt={fragment.name}
          className="w-full h-32 object-contain mb-2 pointer-events-none"
          draggable={false}
        />
        <div className="flex justify-between items-center gap-2">
          <p
            className="text-xs truncate flex-1 font-medium"
            style={{ color: isSelected ? '#1d4ed8' : '#374151' }}
            title={fragment.name}
          >
            {fragment.name}
          </p>
          {/* Info button */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              onFragmentClick(fragment, e);
            }}
            className="flex-shrink-0 text-slate-400 hover:text-blue-600 transition-colors"
            title="View metadata"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>
        </div>

        {/* Badges */}
        <div className="absolute top-2 right-2 flex flex-col gap-1">
          {fragment.metadata && (
            <div className="bg-gradient-to-br from-emerald-500 to-emerald-600 text-white rounded-md px-2 py-1 flex items-center gap-1.5 shadow-md ring-2 ring-white">
              <svg className="w-3.5 h-3.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z"/>
                <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd"/>
              </svg>
              <span className="text-[10px] font-semibold uppercase tracking-wide">Metadata</span>
            </div>
          )}
        </div>

        {/* Shift-click hint */}
        <div className="absolute bottom-10 left-0 right-0 flex justify-center pointer-events-none">
          {!isSelected && (
            <span className="text-[9px] text-slate-300 font-medium">shift+click or right-click to select</span>
          )}
        </div>
      </div>
    </div>
  );
};

const VirtualizedFragmentList: React.FC<VirtualizedFragmentListProps> = ({
  fragments,
  selectedIds,
  onDragStart,
  onFragmentClick,
  onToggleSelect,
  containerHeight,
  onLoadMore,
  hasMore = false,
  isLoadingMore = false,
  scrollPosition = 0,
  onScrollPositionChange,
}) => {
  const listRef = useRef<FixedSizeList>(null);

  useEffect(() => {
    if (listRef.current && scrollPosition > 0) {
      listRef.current.scrollTo(scrollPosition);
    }
  }, [scrollPosition]);

  const handleItemsRendered = useCallback(
    ({ visibleStopIndex }: { visibleStopIndex: number }) => {
      if (hasMore && !isLoadingMore && visibleStopIndex >= fragments.length - 5) {
        onLoadMore?.();
      }
    },
    [hasMore, isLoadingMore, fragments.length, onLoadMore]
  );

  const handleScroll = useCallback(
    ({ scrollOffset }: { scrollOffset: number }) => {
      onScrollPositionChange?.(scrollOffset);
    },
    [onScrollPositionChange]
  );

  const itemData = useMemo<FragmentRowData>(
    () => ({ fragments, selectedIds, onDragStart, onFragmentClick, onToggleSelect }),
    [fragments, selectedIds, onDragStart, onFragmentClick, onToggleSelect]
  );

  const itemCount = Math.max(0, fragments.length);

  return (
    <div className="space-y-3">
      {itemCount > 0 ? (
        <FixedSizeList
          ref={listRef}
          height={containerHeight}
          itemCount={itemCount}
          itemSize={ITEM_HEIGHT}
          width="100%"
          onItemsRendered={handleItemsRendered}
          onScroll={handleScroll}
          itemData={itemData}
        >
          {FragmentRow}
        </FixedSizeList>
      ) : (
        <div className="text-center text-slate-500 py-8">No fragments to display</div>
      )}
      {isLoadingMore && (
        <div className="px-4 py-3">
          <div className="bg-white rounded-lg p-4 shadow-sm border border-slate-200 text-center">
            <div className="flex items-center justify-center gap-2">
              <svg className="w-5 h-5 text-slate-400 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <p className="text-slate-600 text-sm">Loading more...</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VirtualizedFragmentList;

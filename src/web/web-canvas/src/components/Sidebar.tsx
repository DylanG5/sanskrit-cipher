import React, { useState, useRef, useCallback } from 'react';
import { ManuscriptFragment } from '../types/fragment';
import FragmentMetadata from './FragmentMetadata';

interface SidebarProps {
  fragments: ManuscriptFragment[];
  onDragStart: (fragment: ManuscriptFragment, e: React.DragEvent) => void;
  width: number;
  onWidthChange: (width: number) => void;
  isOpen: boolean;
  onToggle: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ fragments, onDragStart, width, onWidthChange, isOpen, onToggle }) => {
  const [selectedFragment, setSelectedFragment] = useState<ManuscriptFragment | null>(null);
  const [isResizing, setIsResizing] = useState(false);
  const sidebarRef = useRef<HTMLDivElement>(null);

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

  return (
    <>
      {/* Toggle button when closed */}
      {!isOpen && (
        <button
          onClick={onToggle}
          className="fixed left-0 top-1/2 -translate-y-1/2 bg-gradient-to-r from-slate-700 to-slate-800 text-white px-3 py-6 rounded-r-lg shadow-lg hover:from-slate-600 hover:to-slate-700 transition-all duration-200 z-30 group"
          title="Open fragments sidebar"
        >
          <div className="flex flex-col items-center gap-2">
            <svg className="w-5 h-5 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            <span className="text-xs font-medium" style={{ writingMode: 'vertical-rl' }}>Fragments</span>
            <div className="text-xs font-semibold bg-blue-500 rounded-full px-1.5 py-0.5 min-w-[20px] text-center">
              {fragments.length}
            </div>
          </div>
        </button>
      )}

      {/* Sidebar panel */}
      {isOpen && (
      <div
        ref={sidebarRef}
        style={{ width: `${width}px` }}
        className="bg-gradient-to-b from-slate-50 to-slate-100 border-r border-slate-300 overflow-y-auto relative shadow-inner flex-shrink-0"
      >
        <div className="p-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
              <svg className="w-5 h-5 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              Fragments
            </h2>
            <button
              onClick={onToggle}
              className="text-slate-500 hover:text-slate-700 hover:bg-slate-200 rounded-lg p-1.5 transition-all duration-200"
              title="Close sidebar"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {fragments.length === 0 ? (
            <div className="bg-white rounded-lg p-6 shadow-sm border border-slate-200 text-center">
              <div className="bg-slate-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-3">
                <svg className="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
                </svg>
              </div>
              <p className="text-slate-600 text-sm font-medium mb-1">No matching fragments</p>
              <p className="text-slate-400 text-xs">Try adjusting your filters</p>
            </div>
          ) : (
          <div className="space-y-3">
            {fragments.map((fragment) => (
              <div
                key={fragment.id}
                draggable
                onDragStart={(e) => onDragStart(fragment, e)}
                onClick={(e) => handleFragmentClick(fragment, e)}
                className="cursor-move bg-white rounded-lg shadow-sm hover:shadow-lg transition-all duration-200 p-3 border border-slate-200 relative group hover:border-blue-300"
              >
                <img
                  src={fragment.thumbnailPath}
                  alt={fragment.name}
                  className="w-full h-32 object-contain mb-2 pointer-events-none"
                  draggable={false}
                />
                <div className="flex justify-between items-center gap-2">
                  <p className="text-xs text-slate-700 truncate flex-1 font-medium" title={fragment.name}>
                    {fragment.name}
                  </p>
                  {/* Info button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleFragmentClick(fragment, e);
                    }}
                    className="flex-shrink-0 text-slate-400 hover:text-blue-600 opacity-0 group-hover:opacity-100 transition-all duration-200 hover:scale-110"
                    title="View metadata"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </button>
                </div>
                {/* Metadata indicator badge - improved design */}
                {fragment.metadata && (
                  <div className="absolute top-2 right-2 group/badge">
                    <div className="bg-gradient-to-br from-emerald-500 to-emerald-600 text-white rounded-full w-6 h-6 flex items-center justify-center shadow-md ring-2 ring-white">
                      <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z"/>
                        <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd"/>
                      </svg>
                    </div>
                    <div className="absolute top-full right-0 mt-1 px-2 py-1 bg-slate-800 text-white text-xs rounded whitespace-nowrap opacity-0 group-hover/badge:opacity-100 transition-opacity pointer-events-none">
                      Has metadata
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
          )}
        </div>

        {/* Resize Handle */}
        <div
          onMouseDown={handleMouseDown}
          className={`absolute top-0 right-0 w-1 h-full cursor-col-resize hover:bg-blue-500 transition-colors group ${
            isResizing ? 'bg-blue-500' : 'bg-transparent'
          }`}
          title="Drag to resize"
        >
          <div className="absolute top-1/2 right-0 -translate-y-1/2 translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
            <div className="bg-blue-500 text-white rounded-full p-1 shadow-lg">
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l4-4 4 4m0 6l-4 4-4-4" />
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

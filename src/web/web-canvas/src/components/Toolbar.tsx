import React from "react";
import { useNavigate } from "react-router-dom";

interface ToolbarProps {
  selectedCount: number;
  onLockSelected: () => void;
  onUnlockSelected: () => void;
  onDeleteSelected: () => void;
  onClearCanvas: () => void;
  onResetView: () => void;
  onSave: () => void;
  onToggleNotes: () => void;
  isGridVisible: boolean;
  onToggleGrid: () => void;
  isFilterPanelOpen: boolean;
  onToggleFilters: () => void;
  hasActiveFilters: boolean;
  // Segmentation toggle
  showSegmented: boolean;
  onToggleSegmented: () => void;
  // Session management props
  projectName?: string;
  saveStatus?: 'saved' | 'saving' | 'unsaved';
}

const Toolbar: React.FC<ToolbarProps> = ({
  selectedCount,
  onLockSelected,
  onUnlockSelected,
  onDeleteSelected,
  onClearCanvas,
  onResetView,
  onSave,
  onToggleNotes,
  isGridVisible,
  onToggleGrid,
  isFilterPanelOpen,
  onToggleFilters,
  hasActiveFilters,
  showSegmented,
  onToggleSegmented,
  projectName,
  saveStatus = 'saved',
}) => {
  const navigate = useNavigate();
  return (
    <div className="h-16 border-b flex items-center px-6 gap-3 shadow-lg" style={{
      background: 'linear-gradient(90deg, #1c1917 0%, #292524 100%)',
      borderColor: 'rgba(120, 113, 108, 0.3)'
    }}>
      <div className="flex items-center gap-2.5">
        <svg
          className="w-6 h-6"
          style={{ color: '#ea580c' }}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          strokeWidth={2.5}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M11 4a2 2 0 114 0v1a1 1 0 001 1h3a1 1 0 011 1v3a1 1 0 01-1 1h-1a2 2 0 100 4h1a1 1 0 011 1v3a1 1 0 01-1 1h-3a1 1 0 01-1-1v-1a2 2 0 10-4 0v1a1 1 0 01-1 1H7a1 1 0 01-1-1v-3a1 1 0 00-1-1H4a2 2 0 110-4h1a1 1 0 001-1V7a1 1 0 011-1h3a1 1 0 001-1V4z"
          />
        </svg>
        <h1 className="text-lg font-bold text-white font-body">
          Fragment Reconstruction
        </h1>
        {projectName && (
          <>
            <span className="text-neutral-500 mx-2">—</span>
            <span className="text-sm font-medium text-neutral-300 font-body truncate max-w-[200px]" title={projectName}>
              {projectName}
            </span>
          </>
        )}
        {/* Save status indicator */}
        <div className="flex items-center gap-1.5 ml-3">
          {saveStatus === 'saving' && (
            <div className="flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium" style={{ color: '#fbbf24' }}>
              <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span>Saving...</span>
            </div>
          )}
          {saveStatus === 'saved' && (
            <div className="flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium" style={{ color: '#10b981' }}>
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              <span>Saved</span>
            </div>
          )}
          {saveStatus === 'unsaved' && (
            <div className="flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium" style={{ color: '#f97316' }}>
              <div className="w-2 h-2 rounded-full bg-current"></div>
              <span>Unsaved</span>
            </div>
          )}
        </div>
      </div>
      <button
        onClick={() => navigate("/")}
        className="px-3.5 py-2 rounded-lg transition-all duration-200 flex items-center gap-2 shadow-md hover:shadow-lg ml-4 font-body"
        style={{
          background: 'rgba(68, 64, 60, 0.8)',
          color: '#d6d3d1'
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'rgba(87, 83, 78, 0.9)';
          e.currentTarget.style.color = '#fafaf9';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = 'rgba(68, 64, 60, 0.8)';
          e.currentTarget.style.color = '#d6d3d1';
        }}
        title="Return to home"
      >
        <svg
          className="w-4 h-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          strokeWidth={2}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"
          />
        </svg>
        <span className="text-sm font-semibold">Home</span>
      </button>

      <div className="flex gap-2 ml-auto items-center">
        {selectedCount > 0 && (
          <>
            <div className="flex items-center gap-2 px-3.5 py-2 rounded-lg border mr-1.5 font-body" style={{
              background: 'rgba(234, 88, 12, 0.15)',
              borderColor: 'rgba(234, 88, 12, 0.4)',
              color: '#fed7aa'
            }}>
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span className="text-sm font-semibold">
                {selectedCount} {selectedCount === 1 ? "item" : "items"}
              </span>
            </div>

            <button
              onClick={onLockSelected}
              className="px-4 py-2.5 text-white rounded-lg transition-all duration-200 flex items-center gap-2 shadow-lg hover:shadow-xl hover:scale-105 group relative font-body"
              style={{
                background: 'linear-gradient(135deg, #ea580c 0%, #c2410c 100%)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'linear-gradient(135deg, #ea580c 0%, #c2410c 100%)';
              }}
              title="Lock selected fragments - prevents moving/editing"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                strokeWidth={2.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                />
              </svg>
              <span className="text-sm font-bold">Lock</span>
            </button>

            <button
              onClick={onUnlockSelected}
              className="px-4 py-2.5 text-white rounded-lg transition-all duration-200 flex items-center gap-2 shadow-lg hover:shadow-xl hover:scale-105 group relative font-body"
              style={{
                background: 'linear-gradient(135deg, #ea580c 0%, #c2410c 100%)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'linear-gradient(135deg, #ea580c 0%, #c2410c 100%)';
              }}
              title="Unlock selected fragments - allows moving/editing"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                strokeWidth={2.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M8 11V7a4 4 0 118 0m-4 8v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2z"
                />
              </svg>
              <span className="text-sm font-bold">Unlock</span>
            </button>

            <button
              onClick={onDeleteSelected}
              className="px-4 py-2.5 text-white rounded-lg transition-all duration-200 flex items-center gap-2 shadow-lg hover:shadow-xl hover:scale-105 font-body"
              style={{
                background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'linear-gradient(135deg, #f87171 0%, #ef4444 100%)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
              }}
              title="Delete selected fragments from canvas"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                strokeWidth={2.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
              <span className="text-sm font-bold">Delete</span>
            </button>

            <div className="w-px h-8 mx-1.5" style={{ background: 'rgba(120, 113, 108, 0.4)' }}></div>
          </>
        )}

        <button
          onClick={onResetView}
          className="px-3.5 py-2 rounded-lg transition-all duration-200 flex items-center gap-2 shadow-md hover:shadow-lg font-body"
          style={{
            background: 'rgba(68, 64, 60, 0.8)',
            color: '#d6d3d1'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(87, 83, 78, 0.9)';
            e.currentTarget.style.color = '#fafaf9';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'rgba(68, 64, 60, 0.8)';
            e.currentTarget.style.color = '#d6d3d1';
          }}
          title="Reset view"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
            />
          </svg>
          <span className="text-sm font-semibold">Reset View</span>
        </button>

        <button
          onClick={onToggleGrid}
          className="px-3.5 py-2 rounded-lg transition-all duration-200 flex items-center gap-2 shadow-md hover:shadow-lg font-body"
          style={{
            background: isGridVisible ? 'linear-gradient(135deg, #ea580c 0%, #c2410c 100%)' : 'rgba(68, 64, 60, 0.8)',
            color: isGridVisible ? '#ffffff' : '#d6d3d1'
          }}
          onMouseEnter={(e) => {
            if (isGridVisible) {
              e.currentTarget.style.background = 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)';
            } else {
              e.currentTarget.style.background = 'rgba(87, 83, 78, 0.9)';
              e.currentTarget.style.color = '#fafaf9';
            }
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = isGridVisible ? 'linear-gradient(135deg, #ea580c 0%, #c2410c 100%)' : 'rgba(68, 64, 60, 0.8)';
            e.currentTarget.style.color = isGridVisible ? '#ffffff' : '#d6d3d1';
          }}
          title={isGridVisible ? "Hide grid" : "Show grid (25px = 1cm)"}
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M4 5a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1v-4zM14 5a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM14 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"
            />
          </svg>
          <span className="text-sm font-semibold">Grid</span>
        </button>

        <button
          onClick={onToggleSegmented}
          className="px-3.5 py-2 rounded-lg transition-all duration-200 flex items-center gap-2 shadow-md hover:shadow-lg font-body"
          style={{
            background: showSegmented ? 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)' : 'rgba(68, 64, 60, 0.8)',
            color: showSegmented ? '#ffffff' : '#d6d3d1'
          }}
          onMouseEnter={(e) => {
            if (showSegmented) {
              e.currentTarget.style.background = 'linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%)';
            } else {
              e.currentTarget.style.background = 'rgba(87, 83, 78, 0.9)';
              e.currentTarget.style.color = '#fafaf9';
            }
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = showSegmented ? 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)' : 'rgba(68, 64, 60, 0.8)';
            e.currentTarget.style.color = showSegmented ? '#ffffff' : '#d6d3d1';
          }}
          title={showSegmented ? "Show original images" : "Show segmented images (transparent background)"}
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
          </svg>
          <span className="text-sm font-semibold">Segmented</span>
        </button>

        <button
          onClick={onToggleFilters}
          className="px-3.5 py-2 rounded-lg transition-all duration-200 flex items-center gap-2 shadow-md hover:shadow-lg relative font-body"
          style={{
            background: isFilterPanelOpen ? 'linear-gradient(135deg, #d97706 0%, #b45309 100%)' : 'rgba(68, 64, 60, 0.8)',
            color: isFilterPanelOpen ? '#ffffff' : '#d6d3d1'
          }}
          onMouseEnter={(e) => {
            if (isFilterPanelOpen) {
              e.currentTarget.style.background = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)';
            } else {
              e.currentTarget.style.background = 'rgba(87, 83, 78, 0.9)';
              e.currentTarget.style.color = '#fafaf9';
            }
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = isFilterPanelOpen ? 'linear-gradient(135deg, #d97706 0%, #b45309 100%)' : 'rgba(68, 64, 60, 0.8)';
            e.currentTarget.style.color = isFilterPanelOpen ? '#ffffff' : '#d6d3d1';
          }}
          title={isFilterPanelOpen ? "Close filters" : "Open filters"}
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
            />
          </svg>
          <span className="text-sm font-semibold">Filters</span>
          {hasActiveFilters && (
            <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full border-2 animate-pulse" style={{
              background: '#ea580c',
              borderColor: '#1c1917'
            }}></div>
          )}
        </button>

        <button
          onClick={onClearCanvas}
          className="px-3.5 py-2 rounded-lg transition-all duration-200 flex items-center gap-2 shadow-md hover:shadow-lg font-body"
          style={{
            background: 'rgba(68, 64, 60, 0.8)',
            color: '#d6d3d1'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(87, 83, 78, 0.9)';
            e.currentTarget.style.color = '#fafaf9';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'rgba(68, 64, 60, 0.8)';
            e.currentTarget.style.color = '#d6d3d1';
          }}
          title="Clear all fragments from canvas"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
          <span className="text-sm font-semibold">Clear Canvas</span>
        </button>

        <div className="w-px h-8 mx-1.5" style={{ background: 'rgba(120, 113, 108, 0.4)' }}></div>

        <button
          onClick={onSave}
          className="px-3.5 py-2 text-white rounded-lg transition-all duration-200 flex items-center gap-2 shadow-lg hover:shadow-xl hover:scale-105 font-body"
          style={{
            background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, #34d399 0%, #10b981 100%)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
          }}
          title="Save canvas progress"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4"
            />
          </svg>
          <span className="text-sm font-semibold">Save</span>
        </button>

        <button
          onClick={onToggleNotes}
          className="px-3.5 py-2 text-white rounded-lg transition-all duration-200 flex items-center gap-2 shadow-lg hover:shadow-xl hover:scale-105 font-body"
          style={{
            background: 'linear-gradient(135deg, #d97706 0%, #b45309 100%)'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, #d97706 0%, #b45309 100%)';
          }}
          title="Toggle notes panel"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
            />
          </svg>
          <span className="text-sm font-semibold">Notes</span>
        </button>
      </div>
    </div>
  );
};

export default Toolbar;

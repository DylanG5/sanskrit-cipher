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
}) => {
  const navigate = useNavigate();
  return (
    <div className="h-16 bg-gradient-to-r from-slate-800 to-slate-900 border-b border-slate-700 flex items-center px-6 gap-3 shadow-lg">
      <div className="flex items-center gap-2">
        <svg
          className="w-6 h-6 text-blue-400"
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
        <h1 className="text-lg font-semibold text-white">
          Fragment Reconstruction
        </h1>
      </div>
      <button
        onClick={() => navigate("/")}
        className="px-3 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-md transition-all duration-200 flex items-center gap-2 shadow-sm hover:shadow-md ml-4"
        title="Return to home"
      >
        <svg
          className="w-4 h-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"
          />
        </svg>
        <span className="text-sm font-medium">Home</span>
      </button>

      <div className="flex gap-2 ml-auto items-center">
        {selectedCount > 0 && (
          <>
            <div className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-500/20 text-blue-300 rounded-md border border-blue-500/30 mr-1">
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span className="text-sm font-medium">
                {selectedCount} {selectedCount === 1 ? "item" : "items"}
              </span>
            </div>

            <button
              onClick={onLockSelected}
              className="px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-all duration-200 flex items-center gap-2 shadow-md hover:shadow-lg hover:scale-105 group relative"
              title="Lock selected fragments - prevents moving/editing"
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
                  d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                />
              </svg>
              <span className="text-sm font-semibold">Lock</span>
            </button>

            <button
              onClick={onUnlockSelected}
              className="px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-all duration-200 flex items-center gap-2 shadow-md hover:shadow-lg hover:scale-105 group relative"
              title="Unlock selected fragments - allows moving/editing"
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
                  d="M8 11V7a4 4 0 118 0m-4 8v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2z"
                />
              </svg>
              <span className="text-sm font-semibold">Unlock</span>
            </button>

            <button
              onClick={onDeleteSelected}
              className="px-4 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-md transition-all duration-200 flex items-center gap-2 shadow-md hover:shadow-lg hover:scale-105"
              title="Delete selected fragments from canvas"
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
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
              <span className="text-sm font-semibold">Delete</span>
            </button>

            <div className="w-px h-8 bg-slate-600 mx-1"></div>
          </>
        )}

        <button
          onClick={onResetView}
          className="px-3 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-md transition-all duration-200 flex items-center gap-2 shadow-sm hover:shadow-md"
          title="Reset view"
        >
          <svg
            className="w-4 h-4"
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
          <span className="text-sm font-medium">Reset View</span>
        </button>

        <button
          onClick={onToggleGrid}
          className={`px-3 py-2 rounded-md transition-all duration-200 flex items-center gap-2 shadow-sm hover:shadow-md ${
            isGridVisible
              ? 'bg-blue-600 hover:bg-blue-700 text-white'
              : 'bg-slate-700 hover:bg-slate-600 text-slate-200'
          }`}
          title={isGridVisible ? "Hide grid" : "Show grid (25px = 1cm)"}
        >
          <svg
            className="w-4 h-4"
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
          <span className="text-sm font-medium">Grid</span>
        </button>

        <button
          onClick={onToggleFilters}
          className={`px-3 py-2 rounded-md transition-all duration-200 flex items-center gap-2 shadow-sm hover:shadow-md relative ${
            isFilterPanelOpen
              ? 'bg-purple-600 hover:bg-purple-700 text-white'
              : 'bg-slate-700 hover:bg-slate-600 text-slate-200'
          }`}
          title={isFilterPanelOpen ? "Close filters" : "Open filters"}
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
              d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
            />
          </svg>
          <span className="text-sm font-medium">Filters</span>
          {hasActiveFilters && (
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-blue-400 rounded-full border-2 border-slate-800 animate-pulse"></div>
          )}
        </button>

        <button
          onClick={onClearCanvas}
          className="px-3 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-md transition-all duration-200 flex items-center gap-2 shadow-sm hover:shadow-md"
          title="Clear all fragments from canvas"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
          <span className="text-sm font-medium">Clear Canvas</span>
        </button>

        <div className="w-px h-8 bg-slate-600 mx-1"></div>

        <button
          onClick={onSave}
          className="px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded-md transition-all duration-200 flex items-center gap-2 shadow-sm hover:shadow-md hover:scale-105"
          title="Save canvas progress"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4"
            />
          </svg>
          <span className="text-sm font-medium">Save</span>
        </button>

        <button
          onClick={onToggleNotes}
          className="px-3 py-2 bg-amber-600 hover:bg-amber-700 text-white rounded-md transition-all duration-200 flex items-center gap-2 shadow-sm hover:shadow-md hover:scale-105"
          title="Toggle notes panel"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
            />
          </svg>
          <span className="text-sm font-medium">Notes</span>
        </button>
      </div>
    </div>
  );
};

export default Toolbar;

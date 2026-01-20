import React from "react";
import { useNavigate } from "react-router-dom";

interface ToolbarProps {
  selectedCount: number;
  onLockSelected: () => void;
  onUnlockSelected: () => void;
  onDeleteSelected: () => void;
  onClearCanvas: () => void;
  onResetView: () => void;
}

const Toolbar: React.FC<ToolbarProps> = ({
  selectedCount,
  onLockSelected,
  onUnlockSelected,
  onDeleteSelected,
  onClearCanvas,
  onResetView,
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
              className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-all duration-200 flex items-center gap-2 shadow-sm hover:shadow-md hover:scale-105"
              title="Lock selected fragments"
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
                  d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                />
              </svg>
              <span className="text-sm font-medium">Lock</span>
            </button>

            <button
              onClick={onUnlockSelected}
              className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-all duration-200 flex items-center gap-2 shadow-sm hover:shadow-md hover:scale-105"
              title="Unlock selected fragments"
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
                  d="M8 11V7a4 4 0 118 0m-4 8v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2z"
                />
              </svg>
              <span className="text-sm font-medium">Unlock</span>
            </button>

            <button
              onClick={onDeleteSelected}
              className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md transition-all duration-200 flex items-center gap-2 shadow-sm hover:shadow-md hover:scale-105"
              title="Delete selected fragments"
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
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
              <span className="text-sm font-medium">Delete</span>
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
      </div>
    </div>
  );
};

export default Toolbar;
